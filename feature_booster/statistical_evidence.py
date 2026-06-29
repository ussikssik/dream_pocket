from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import StatisticalConfig
from .utils import benjamini_hochberg, cramers_v_from_table, effect_direction, numeric_or_none, stable_random

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency fallback
    scipy_stats = None


def compute_statistical_evidence(
    X: pd.DataFrame,
    y: pd.Series,
    config: StatisticalConfig,
    presence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute contrast evidence between Good and Bad groups."""

    records: list[dict[str, Any]] = []
    y = y.astype(int)
    bad_mask = y == 1
    good_mask = y == 0

    for feature in X.columns:
        series = X[feature]
        numeric = numeric_or_none(series)
        if numeric is not None and series.dropna().nunique() > config.numeric_unique_threshold:
            record = _numeric_evidence(feature, numeric, bad_mask, good_mask)
        else:
            record = _categorical_evidence(feature, series, y, config)

        presence_type = str(presence_df.loc[feature, "presence_type"])
        record["direction"] = effect_direction(record.get("effect_size"), presence_type)
        records.append(record)

    evidence = pd.DataFrame.from_records(records).set_index("feature_name")
    evidence["fdr_pvalue"] = benjamini_hochberg(evidence["p_value"])
    evidence["bootstrap_stability_score"] = bootstrap_direction_stability(
        X[evidence.index.tolist()],
        y,
        evidence["direction"],
        config,
    )
    evidence["direction_consistency_score"] = evidence["bootstrap_stability_score"].fillna(0.0)
    return evidence


def _numeric_evidence(
    feature: str,
    series: pd.Series,
    bad_mask: pd.Series,
    good_mask: pd.Series,
) -> dict[str, Any]:
    bad = series[bad_mask].dropna().astype(float)
    good = series[good_mask].dropna().astype(float)
    n_bad = len(bad)
    n_good = len(good)

    effect = np.nan
    p_value = np.nan
    ks_p_value = np.nan
    cliff_delta = np.nan
    bad_median = np.nan
    good_median = np.nan

    if n_bad > 0:
        bad_median = float(bad.median())
    if n_good > 0:
        good_median = float(good.median())

    if n_bad >= 2 and n_good >= 2:
        effect = _robust_effect_size(bad, good)
        if scipy_stats is not None:
            try:
                mw = scipy_stats.mannwhitneyu(bad, good, alternative="two-sided")
                p_value = float(mw.pvalue)
                cliff_delta = float((2.0 * mw.statistic / (n_bad * n_good)) - 1.0)
            except Exception:
                p_value = np.nan
            try:
                ks_p_value = float(scipy_stats.ks_2samp(bad, good).pvalue)
            except Exception:
                ks_p_value = np.nan
        else:
            p_value = _normal_approx_pvalue(effect, n_bad, n_good)

    return {
        "feature_name": feature,
        "stat_type": "numeric",
        "bad_center": bad_median,
        "good_center": good_median,
        "center_delta_bad_minus_good": bad_median - good_median
        if np.isfinite(bad_median) and np.isfinite(good_median)
        else np.nan,
        "effect_size": effect,
        "cliff_delta": cliff_delta,
        "p_value": p_value,
        "ks_p_value": ks_p_value,
    }


def _categorical_evidence(
    feature: str,
    series: pd.Series,
    y: pd.Series,
    config: StatisticalConfig,
) -> dict[str, Any]:
    clean = series.astype("object").where(series.notna(), "__MISSING__")
    if clean.nunique() > config.max_categories:
        top_values = set(clean.value_counts().head(config.max_categories).index)
        clean = clean.where(clean.isin(top_values), "__OTHER__")

    table = pd.crosstab(clean, y)
    p_value = np.nan
    if table.shape[0] > 1 and table.shape[1] > 1 and scipy_stats is not None:
        try:
            p_value = float(scipy_stats.chi2_contingency(table, correction=False).pvalue)
        except Exception:
            p_value = np.nan

    effect = cramers_v_from_table(table)
    bad_rates = _target_rates(clean, y)
    if bad_rates.empty:
        center_delta = np.nan
    else:
        center_delta = float(bad_rates.max() - bad_rates.min())

    return {
        "feature_name": feature,
        "stat_type": "categorical",
        "bad_center": np.nan,
        "good_center": np.nan,
        "center_delta_bad_minus_good": center_delta,
        "effect_size": effect,
        "cliff_delta": np.nan,
        "p_value": p_value,
        "ks_p_value": np.nan,
    }


def bootstrap_direction_stability(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_direction: pd.Series,
    config: StatisticalConfig,
) -> pd.Series:
    if config.bootstrap_rounds <= 0 or len(X) < 10:
        return pd.Series(np.nan, index=X.columns)

    rng = stable_random(config.random_state)
    n_rows = len(X)
    sample_size = max(2, int(n_rows * config.bootstrap_sample_frac))
    same_direction_counts = pd.Series(0, index=X.columns, dtype=float)
    valid_counts = pd.Series(0, index=X.columns, dtype=float)

    for _ in range(config.bootstrap_rounds):
        sample_idx = rng.choice(n_rows, size=sample_size, replace=True)
        y_sample = y.iloc[sample_idx].reset_index(drop=True).astype(int)
        bad_mask = y_sample == 1
        good_mask = y_sample == 0
        if bad_mask.sum() < 2 or good_mask.sum() < 2:
            continue

        X_sample = X.iloc[sample_idx].reset_index(drop=True)
        for feature in X.columns:
            numeric = numeric_or_none(X_sample[feature])
            if numeric is not None and numeric.dropna().nunique() > config.numeric_unique_threshold:
                bad = numeric[bad_mask].dropna()
                good = numeric[good_mask].dropna()
                if len(bad) < 2 or len(good) < 2:
                    continue
                effect = _robust_effect_size(bad, good)
            else:
                effect = _categorical_effect_fast(X_sample[feature], y_sample, config)

            if not np.isfinite(effect) or abs(effect) < config.bootstrap_min_abs_effect:
                continue
            direction = "bad_higher" if effect > 0 else "good_higher"
            expected = baseline_direction.get(feature)
            if expected in {"bad_presence", "bad_higher"}:
                same = direction == "bad_higher"
            elif expected in {"good_presence", "good_higher"}:
                same = direction == "good_higher"
            else:
                same = False
            same_direction_counts.loc[feature] += 1.0 if same else 0.0
            valid_counts.loc[feature] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        stability = same_direction_counts / valid_counts.replace(0.0, np.nan)
    return stability.fillna(0.0).clip(0.0, 1.0)


def _robust_effect_size(bad: pd.Series, good: pd.Series) -> float:
    bad = pd.to_numeric(bad, errors="coerce").dropna()
    good = pd.to_numeric(good, errors="coerce").dropna()
    if len(bad) < 2 or len(good) < 2:
        return np.nan
    delta = float(bad.median() - good.median())
    pooled_iqr = (float(bad.quantile(0.75) - bad.quantile(0.25)) + float(good.quantile(0.75) - good.quantile(0.25))) / 2.0
    pooled_std = math.sqrt(max(float(bad.var(ddof=1)), 0.0) + max(float(good.var(ddof=1)), 0.0)) / math.sqrt(2.0)
    scale = pooled_iqr / 1.349 if pooled_iqr > 0 else pooled_std
    if not np.isfinite(scale) or scale <= 1e-12:
        if abs(delta) <= 1e-12:
            return 0.0
        return float(np.sign(delta))
    return float(delta / scale)


def _normal_approx_pvalue(effect: float, n_bad: int, n_good: int) -> float:
    if not np.isfinite(effect):
        return np.nan
    effective_n = (n_bad * n_good) / max(n_bad + n_good, 1)
    z = abs(effect) * math.sqrt(max(effective_n, 1.0) / 2.0)
    return float(math.erfc(z / math.sqrt(2.0)))


def _target_rates(values: pd.Series, y: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"value": values, "target": y})
    grouped = frame.groupby("value", observed=True)["target"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 2]
    return grouped["mean"]


def _categorical_effect_fast(series: pd.Series, y: pd.Series, config: StatisticalConfig) -> float:
    clean = series.astype("object").where(series.notna(), "__MISSING__")
    if clean.nunique() > config.max_categories:
        top_values = set(clean.value_counts().head(config.max_categories).index)
        clean = clean.where(clean.isin(top_values), "__OTHER__")
    rates = _target_rates(clean, y)
    if rates.empty:
        return 0.0
    spread = float(rates.max() - rates.min())
    return spread if rates.idxmax() != "__MISSING__" else -spread
