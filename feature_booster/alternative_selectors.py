from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .catboost_xai_probe import run_catboost_xai_probe
from .config import CatBoostProbeConfig
from .types import OrderId, OrderLoader
from .utils import as_binary_target, benjamini_hochberg, cramers_v_from_table, infer_feature_columns, numeric_or_none

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency fallback
    scipy_stats = None


@dataclass(frozen=True)
class AlternativeSelectorConfig:
    """Config for comparison-only feature selection modules."""

    label_col: str = "target_bad_a"
    positive_label: Any = 1
    group_cols: tuple[str, ...] = ()
    sample_id_cols: tuple[str, ...] = ()
    exclude_cols: tuple[str, ...] = ()
    output_dir: Path | None = None
    methods: tuple[str, ...] = (
        "nonparametric_random",
        "distance",
        "catboost_shap_gap",
        "stability_consensus",
    )
    alpha: float = 0.05
    random_state: int = 42
    random_pool_size: int = 300
    max_categories: int = 200
    min_group_non_null_count: int = 3
    catboost_prefilter_top_n: int = 500
    stability_prefilter_top_n: int = 500
    stability_rounds: int = 12
    stability_top_multiplier: int = 3
    catboost: CatBoostProbeConfig = field(
        default_factory=lambda: CatBoostProbeConfig(
            enabled=True,
            max_features_per_order=500,
            iterations=220,
            depth=4,
            learning_rate=0.05,
            verbose=False,
        )
    )


class AlternativeFeatureSelector:
    """Run alternative Good/Bad feature selectors for method comparison.

    These selectors are intentionally simpler than the main evidence booster.
    They are useful as baselines or challenger methods, not as a replacement for
    the full evidence report.
    """

    VALID_METHODS = {
        "nonparametric_random",
        "distance",
        "catboost_shap_gap",
        "stability_consensus",
    }

    def __init__(self, order_loader: OrderLoader, config: AlternativeSelectorConfig | None = None):
        self.order_loader = order_loader
        self.config = config or AlternativeSelectorConfig()

    def run(
        self,
        order_list: list[OrderId],
        top_k_per_order: int = 10,
        methods: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        methods = tuple(methods or self.config.methods)
        invalid = sorted(set(methods) - self.VALID_METHODS)
        if invalid:
            raise ValueError(f"Unknown selector methods: {invalid}")

        results = []
        for order_id in order_list:
            raw = self.order_loader(order_id)
            X, y = self._split_order_frame(raw)
            profiles = compute_univariate_profiles(
                X,
                y,
                max_categories=self.config.max_categories,
                min_group_non_null_count=self.config.min_group_non_null_count,
            )
            for method in methods:
                selected = self._run_method(method, X, y, profiles, top_k_per_order, order_id)
                if not selected.empty:
                    results.append(selected)

        combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        if self.config.output_dir is not None and not combined.empty:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            combined.to_csv(
                self.config.output_dir / "combined_alternative_selectors.csv",
                index=False,
                encoding="utf-8-sig",
            )
        return combined

    def _run_method(
        self,
        method: str,
        X: pd.DataFrame,
        y: pd.Series,
        profiles: pd.DataFrame,
        top_k: int,
        order_id: OrderId,
    ) -> pd.DataFrame:
        if method == "nonparametric_random":
            selected = select_nonparametric_random(profiles, top_k, self.config)
        elif method == "distance":
            selected = select_distance(profiles, top_k)
        elif method == "catboost_shap_gap":
            selected = select_catboost_shap_gap(X, y, profiles, top_k, self.config)
        elif method == "stability_consensus":
            selected = select_stability_consensus(X, y, profiles, top_k, self.config)
        else:  # pragma: no cover - guarded by VALID_METHODS
            raise ValueError(f"Unknown selector method: {method}")

        selected = selected.copy()
        selected.insert(0, "order_id", order_id)
        selected.insert(1, "method", method)
        selected.insert(2, "rank", range(1, len(selected) + 1))
        return _order_selector_columns(selected)

    def _split_order_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if self.config.label_col not in df.columns:
            raise ValueError(f"label_col '{self.config.label_col}' is missing from order data")

        y = as_binary_target(df[self.config.label_col], self.config.positive_label)
        group_cols = tuple(col for col in self.config.group_cols if col in df.columns)
        feature_cols = infer_feature_columns(
            df,
            self.config.label_col,
            group_cols,
            self.config.sample_id_cols,
            self.config.exclude_cols,
        )
        return df[feature_cols].copy(), y


def compute_univariate_profiles(
    X: pd.DataFrame,
    y: pd.Series,
    max_categories: int = 200,
    min_group_non_null_count: int = 3,
) -> pd.DataFrame:
    """Compute nonparametric and distance evidence for all candidate features."""

    y = y.astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    bad_mask = y == 1
    good_mask = y == 0
    records: list[dict[str, Any]] = []

    for feature in X.columns:
        series = X[feature]
        numeric = numeric_or_none(series)
        presence = _presence_profile(series, bad_mask, good_mask)

        if numeric is not None and numeric.dropna().nunique() > 5:
            record = _numeric_profile(
                feature,
                numeric,
                bad_mask,
                good_mask,
                presence,
                min_group_non_null_count,
            )
        else:
            record = _categorical_profile(
                feature,
                series,
                y,
                bad_mask,
                good_mask,
                presence,
                max_categories,
            )
        records.append(record)

    profiles = pd.DataFrame.from_records(records)
    if profiles.empty:
        return profiles

    profiles["fdr_pvalue"] = benjamini_hochberg(profiles["p_value"])
    profiles["nonparametric_score"] = (
        _safe_neg_log10(profiles["p_value"]) * 0.55
        + profiles["effect_score"].fillna(0.0) * 0.25
        + profiles["presence_delta_abs"].fillna(0.0) * 0.20
    )
    return profiles.sort_values("distance_score", ascending=False).reset_index(drop=True)


def select_nonparametric_random(
    profiles: pd.DataFrame,
    top_k: int,
    config: AlternativeSelectorConfig,
) -> pd.DataFrame:
    """Filter by raw p-value <= alpha, then random-sample top candidates."""

    if profiles.empty:
        return profiles

    rng = np.random.default_rng(config.random_state)
    significant = profiles[profiles["p_value"].fillna(1.0) <= config.alpha].copy()
    if significant.empty:
        significant = profiles.sort_values("p_value", na_position="last").head(max(top_k, config.random_pool_size)).copy()
        warning = f"no_raw_pvalue_under_{config.alpha}; used_lowest_pvalue_pool"
    else:
        warning = ""

    pool = significant.sort_values("nonparametric_score", ascending=False).head(config.random_pool_size).copy()
    if pool.empty:
        return pool

    weights = pool["nonparametric_score"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = None
    else:
        weights = weights / weights.sum()

    take = min(top_k, len(pool))
    picked = rng.choice(np.arange(len(pool)), size=take, replace=False, p=weights)
    selected = pool.iloc[picked].copy()
    selected = selected.sort_values("nonparametric_score", ascending=False)
    selected["score"] = selected["nonparametric_score"]
    selected["selection_reason"] = "pvalue_filter_then_weighted_random_sample"
    selected["selection_warning"] = warning
    selected["sampling_pool_size"] = len(pool)
    return selected


def select_distance(profiles: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Select by distribution-distance score."""

    selected = profiles.sort_values("distance_score", ascending=False).head(top_k).copy()
    selected["score"] = selected["distance_score"]
    selected["selection_reason"] = "largest_good_bad_distribution_distance"
    selected["selection_warning"] = ""
    return selected


def select_catboost_shap_gap(
    X: pd.DataFrame,
    y: pd.Series,
    profiles: pd.DataFrame,
    top_k: int,
    config: AlternativeSelectorConfig,
) -> pd.DataFrame:
    """Select features with the largest Bad-vs-Good mean SHAP gap."""

    candidates = (
        profiles.sort_values("distance_score", ascending=False)
        .head(config.catboost_prefilter_top_n)["feature_name"]
        .tolist()
    )
    if not candidates:
        return pd.DataFrame()

    xai = run_catboost_xai_probe(X, y, candidates, config.catboost).reset_index()
    xai["shap_gap_bad_minus_good"] = xai["shap_bad_mean"] - xai["shap_good_mean"]
    xai["shap_gap_abs"] = xai["shap_gap_bad_minus_good"].abs()

    merged = profiles.merge(xai, on="feature_name", how="inner")
    available = merged[merged["xai_available"]].copy()
    if available.empty:
        fallback = profiles[profiles["feature_name"].isin(candidates)].head(top_k).copy()
        warning = str(xai["xai_warning"].dropna().iloc[0]) if not xai.empty else "catboost_unavailable"
        fallback["score"] = 0.0
        fallback["shap_gap_abs"] = np.nan
        fallback["shap_gap_bad_minus_good"] = np.nan
        fallback["selection_reason"] = "catboost_shap_gap_unavailable"
        fallback["selection_warning"] = warning
        return fallback

    selected = available.sort_values("shap_gap_abs", ascending=False).head(top_k).copy()
    selected["score"] = selected["shap_gap_abs"]
    selected["selection_reason"] = "largest_abs_mean_shap_bad_minus_good"
    selected["selection_warning"] = selected["xai_warning"].fillna("")
    return selected


def select_stability_consensus(
    X: pd.DataFrame,
    y: pd.Series,
    profiles: pd.DataFrame,
    top_k: int,
    config: AlternativeSelectorConfig,
) -> pd.DataFrame:
    """Bootstrap distance ranking and select features that repeatedly survive."""

    if profiles.empty:
        return profiles

    rng = np.random.default_rng(config.random_state + 7919)
    candidates = (
        profiles.sort_values("distance_score", ascending=False)
        .head(config.stability_prefilter_top_n)["feature_name"]
        .tolist()
    )
    if not candidates:
        return pd.DataFrame()

    y_reset = y.astype(int).reset_index(drop=True)
    X_reset = X[candidates].reset_index(drop=True)
    bad_idx = np.flatnonzero(y_reset.to_numpy() == 1)
    good_idx = np.flatnonzero(y_reset.to_numpy() == 0)
    if len(bad_idx) < 2 or len(good_idx) < 2:
        selected = profiles[profiles["feature_name"].isin(candidates)].head(top_k).copy()
        selected["score"] = selected["distance_score"]
        selected["selection_reason"] = "stability_consensus_fallback_distance"
        selected["selection_warning"] = "not_enough_good_bad_rows_for_bootstrap"
        return selected

    selection_counts = pd.Series(0.0, index=candidates)
    rank_credit = pd.Series(0.0, index=candidates)
    per_round_top_n = max(top_k, top_k * config.stability_top_multiplier)

    for _ in range(max(config.stability_rounds, 1)):
        sample_bad = rng.choice(bad_idx, size=len(bad_idx), replace=True)
        sample_good = rng.choice(good_idx, size=len(good_idx), replace=True)
        sample_idx = np.r_[sample_good, sample_bad]
        rng.shuffle(sample_idx)

        sampled_profiles = compute_univariate_profiles(
            X_reset.iloc[sample_idx].reset_index(drop=True),
            y_reset.iloc[sample_idx].reset_index(drop=True),
            max_categories=config.max_categories,
            min_group_non_null_count=config.min_group_non_null_count,
        )
        round_top = sampled_profiles.sort_values("distance_score", ascending=False).head(per_round_top_n)
        for rank, feature_name in enumerate(round_top["feature_name"], start=1):
            selection_counts.loc[feature_name] += 1.0
            rank_credit.loc[feature_name] += 1.0 / rank

    stability = pd.DataFrame(
        {
            "feature_name": candidates,
            "stability_selection_rate": selection_counts / max(config.stability_rounds, 1),
            "stability_rank_credit": rank_credit / max(config.stability_rounds, 1),
        }
    )
    merged = profiles.merge(stability, on="feature_name", how="inner")
    merged["stability_consensus_score"] = (
        merged["stability_selection_rate"] * 0.75
        + merged["stability_rank_credit"].rank(pct=True).fillna(0.0) * 0.25
    )
    selected = merged.sort_values("stability_consensus_score", ascending=False).head(top_k).copy()
    selected["score"] = selected["stability_consensus_score"]
    selected["selection_reason"] = "bootstrap_distance_ranking_consensus"
    selected["selection_warning"] = ""
    return selected


def summarize_method_overlap(result: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise Jaccard overlap between selector methods per order."""

    if result.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for order_id, order_df in result.groupby("order_id"):
        method_sets = {
            method: set(method_df["feature_name"])
            for method, method_df in order_df.groupby("method")
        }
        methods = sorted(method_sets)
        for left in methods:
            for right in methods:
                union = method_sets[left] | method_sets[right]
                intersection = method_sets[left] & method_sets[right]
                rows.append(
                    {
                        "order_id": order_id,
                        "method_left": left,
                        "method_right": right,
                        "intersection_count": len(intersection),
                        "union_count": len(union),
                        "jaccard": len(intersection) / len(union) if union else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _numeric_profile(
    feature: str,
    series: pd.Series,
    bad_mask: pd.Series,
    good_mask: pd.Series,
    presence: dict[str, float],
    min_group_non_null_count: int,
) -> dict[str, Any]:
    bad = pd.to_numeric(series[bad_mask], errors="coerce").dropna()
    good = pd.to_numeric(series[good_mask], errors="coerce").dropna()

    p_value = np.nan
    p_source = ""
    cliff_delta = np.nan
    ks_stat = np.nan
    ks_p_value = np.nan
    effect_score = 0.0
    distance_score = presence["presence_delta_abs"] * 0.35
    direction = "flat"

    if len(bad) >= min_group_non_null_count and len(good) >= min_group_non_null_count:
        p_value, cliff_delta = _mann_whitney_pvalue_and_cliff(bad, good)
        p_source = "mann_whitney_u"
        ks_stat, ks_p_value = _ks_stat_and_pvalue(bad, good)
        scale = _robust_scale(pd.concat([bad, good], ignore_index=True))
        median_gap = float(bad.median() - good.median())
        median_gap_scaled = abs(median_gap) / scale
        wasserstein = _wasserstein_distance(bad, good) / scale
        threshold_lift = _max_threshold_lift(pd.concat([good, bad], ignore_index=True), len(good))

        effect_score = abs(float(cliff_delta)) if np.isfinite(cliff_delta) else min(median_gap_scaled, 3.0) / 3.0
        distance_score = (
            0.32 * _squash(wasserstein)
            + 0.24 * float(np.nan_to_num(ks_stat, nan=0.0))
            + 0.22 * _squash(median_gap_scaled)
            + 0.14 * threshold_lift
            + 0.08 * presence["presence_delta_abs"]
        )
        direction = "bad_higher" if median_gap > 0 else "good_higher" if median_gap < 0 else "flat"

    presence_p = _presence_p_value(presence)
    if np.isfinite(presence_p) and (not np.isfinite(p_value) or presence_p < p_value):
        p_value = presence_p
        p_source = "presence_fisher_or_z"

    return {
        "feature_name": feature,
        "stat_type": "numeric",
        "p_value": p_value,
        "p_value_source": p_source,
        "ks_p_value": ks_p_value,
        "ks_statistic": ks_stat,
        "effect_score": effect_score,
        "cliff_delta": cliff_delta,
        "distance_score": float(distance_score),
        "direction": direction,
        **presence,
    }


def _categorical_profile(
    feature: str,
    series: pd.Series,
    y: pd.Series,
    bad_mask: pd.Series,
    good_mask: pd.Series,
    presence: dict[str, float],
    max_categories: int,
) -> dict[str, Any]:
    clean = series.astype("object").where(series.notna(), "__MISSING__")
    if clean.nunique(dropna=False) > max_categories:
        top_values = set(clean.value_counts(dropna=False).head(max_categories).index)
        clean = clean.where(clean.isin(top_values), "__OTHER__")

    table = pd.crosstab(clean, y)
    p_value = _chi_square_p_value(table)
    presence_p = _presence_p_value(presence)
    p_source = "chi_square"
    if np.isfinite(presence_p) and (not np.isfinite(p_value) or presence_p < p_value):
        p_value = presence_p
        p_source = "presence_fisher_or_z"

    bad_dist = clean[bad_mask].value_counts(normalize=True, dropna=False)
    good_dist = clean[good_mask].value_counts(normalize=True, dropna=False)
    categories = sorted(set(bad_dist.index) | set(good_dist.index), key=str)
    total_variation = 0.0
    if categories:
        total_variation = 0.5 * sum(abs(float(bad_dist.get(cat, 0.0)) - float(good_dist.get(cat, 0.0))) for cat in categories)

    effect = cramers_v_from_table(table) if not table.empty else 0.0
    distance_score = 0.72 * total_variation + 0.20 * effect + 0.08 * presence["presence_delta_abs"]

    return {
        "feature_name": feature,
        "stat_type": "categorical",
        "p_value": p_value,
        "p_value_source": p_source,
        "ks_p_value": np.nan,
        "ks_statistic": np.nan,
        "effect_score": float(effect),
        "cliff_delta": np.nan,
        "distance_score": float(distance_score),
        "direction": "category_shift",
        **presence,
    }


def _presence_profile(series: pd.Series, bad_mask: pd.Series, good_mask: pd.Series) -> dict[str, float]:
    present = series.notna()
    bad_total = int(bad_mask.sum())
    good_total = int(good_mask.sum())
    bad_present = int((present & bad_mask).sum())
    good_present = int((present & good_mask).sum())
    bad_rate = bad_present / bad_total if bad_total else 0.0
    good_rate = good_present / good_total if good_total else 0.0
    delta = bad_rate - good_rate
    return {
        "bad_non_null_count": bad_present,
        "good_non_null_count": good_present,
        "bad_coverage": bad_rate,
        "good_coverage": good_rate,
        "presence_delta_bad_minus_good": delta,
        "presence_delta_abs": abs(delta),
        "bad_null_count": bad_total - bad_present,
        "good_null_count": good_total - good_present,
    }


def _mann_whitney_pvalue_and_cliff(bad: pd.Series, good: pd.Series) -> tuple[float, float]:
    if scipy_stats is not None:
        try:
            mw = scipy_stats.mannwhitneyu(bad, good, alternative="two-sided")
            cliff = (2.0 * float(mw.statistic) / (len(bad) * len(good))) - 1.0
            return float(mw.pvalue), float(cliff)
        except Exception:
            pass

    values = pd.concat([bad, good], ignore_index=True)
    ranks = values.rank(method="average").to_numpy(dtype=float)
    n_bad = len(bad)
    n_good = len(good)
    rank_bad_sum = float(ranks[:n_bad].sum())
    u = rank_bad_sum - n_bad * (n_bad + 1) / 2.0
    mean_u = n_bad * n_good / 2.0
    var_u = n_bad * n_good * (n_bad + n_good + 1) / 12.0
    z = (u - mean_u) / math.sqrt(max(var_u, 1e-12))
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    cliff = (2.0 * u / max(n_bad * n_good, 1)) - 1.0
    return float(p_value), float(cliff)


def _ks_stat_and_pvalue(bad: pd.Series, good: pd.Series) -> tuple[float, float]:
    if scipy_stats is not None:
        try:
            ks = scipy_stats.ks_2samp(bad, good)
            return float(ks.statistic), float(ks.pvalue)
        except Exception:
            pass
    values = np.sort(np.unique(np.r_[bad.to_numpy(dtype=float), good.to_numpy(dtype=float)]))
    if len(values) == 0:
        return np.nan, np.nan
    bad_cdf = np.searchsorted(np.sort(bad), values, side="right") / len(bad)
    good_cdf = np.searchsorted(np.sort(good), values, side="right") / len(good)
    return float(np.max(np.abs(bad_cdf - good_cdf))), np.nan


def _wasserstein_distance(bad: pd.Series, good: pd.Series) -> float:
    if scipy_stats is not None:
        try:
            return float(scipy_stats.wasserstein_distance(bad, good))
        except Exception:
            pass
    quantiles = np.linspace(0.05, 0.95, 19)
    return float(np.mean(np.abs(np.quantile(bad, quantiles) - np.quantile(good, quantiles))))


def _presence_p_value(presence: dict[str, float]) -> float:
    table = np.array(
        [
            [presence["bad_non_null_count"], presence["bad_null_count"]],
            [presence["good_non_null_count"], presence["good_null_count"]],
        ],
        dtype=float,
    )
    if table.sum() <= 0 or table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        return np.nan
    if scipy_stats is not None:
        try:
            return float(scipy_stats.fisher_exact(table)[1])
        except Exception:
            pass
    bad_total = table[0].sum()
    good_total = table[1].sum()
    p_bad = table[0, 0] / max(bad_total, 1.0)
    p_good = table[1, 0] / max(good_total, 1.0)
    pooled = table[:, 0].sum() / max(table.sum(), 1.0)
    se = math.sqrt(max(pooled * (1.0 - pooled) * (1.0 / bad_total + 1.0 / good_total), 1e-12))
    return float(math.erfc(abs(p_bad - p_good) / se / math.sqrt(2.0)))


def _chi_square_p_value(table: pd.DataFrame) -> float:
    if table.shape[0] <= 1 or table.shape[1] <= 1:
        return np.nan
    if scipy_stats is not None:
        try:
            return float(scipy_stats.chi2_contingency(table, correction=False).pvalue)
        except Exception:
            return np.nan
    return np.nan


def _max_threshold_lift(values: pd.Series, good_count: int) -> float:
    values = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    y = pd.Series(np.r_[np.zeros(good_count, dtype=int), np.ones(len(values) - good_count, dtype=int)])
    valid = values.notna()
    values = values[valid]
    y = y[valid]
    if values.nunique() < 5 or y.nunique() < 2:
        return 0.0

    lifts = []
    for q in np.linspace(0.60, 0.95, 8):
        threshold = float(values.quantile(q))
        high = values >= threshold
        low = values <= float(values.quantile(1.0 - q))
        for mask in (high, low):
            if mask.sum() < 3:
                continue
            bad_rate_in = float(y[mask].mean())
            bad_rate_out = float(y[~mask].mean()) if (~mask).sum() else 0.0
            lifts.append(abs(bad_rate_in - bad_rate_out))
    return float(max(lifts)) if lifts else 0.0


def _robust_scale(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 1.0
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    scale = iqr / 1.349 if iqr > 1e-12 else std
    return scale if np.isfinite(scale) and scale > 1e-12 else 1.0


def _squash(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value / (1.0 + abs(value)))


def _safe_neg_log10(series: pd.Series) -> pd.Series:
    return -np.log10(pd.to_numeric(series, errors="coerce").clip(lower=1e-300)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _order_selector_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "order_id",
        "method",
        "rank",
        "feature_name",
        "score",
        "selection_reason",
        "selection_warning",
        "stat_type",
        "direction",
        "p_value",
        "fdr_pvalue",
        "p_value_source",
        "distance_score",
        "nonparametric_score",
        "stability_selection_rate",
        "stability_rank_credit",
        "shap_gap_abs",
        "shap_gap_bad_minus_good",
        "shap_bad_mean",
        "shap_good_mean",
        "xai_consistency_score",
        "bad_coverage",
        "good_coverage",
        "presence_delta_bad_minus_good",
        "effect_score",
        "cliff_delta",
        "ks_statistic",
        "ks_p_value",
        "sampling_pool_size",
    ]
    existing = [col for col in preferred if col in df.columns]
    extras = [col for col in df.columns if col not in existing]
    return df[existing + extras]
