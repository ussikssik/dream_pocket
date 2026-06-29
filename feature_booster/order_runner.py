from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from .asymmetric_signal_detector import detect_asymmetric_signals
from .catboost_xai_probe import run_catboost_xai_probe
from .config import BoosterConfig
from .confounding_checker import check_group_confounding, compute_groupwise_stability
from .evidence_ranker import rank_evidence
from .quality_profiler import profile_quality
from .redundancy_filter import assign_redundancy_groups
from .report import save_combined_report, save_order_report
from .statistical_evidence import compute_statistical_evidence
from .types import OrderId, OrderLoader
from .utils import as_binary_target, infer_feature_columns


class DefectAFeatureEvidenceBooster:
    """Order-wise evidence booster for weakly indexed defect labels.

    CatBoost is used only as an optional XAI probe. Final ranking is based on
    contrast, stability, asymmetric presence, SHAP consistency, and confounding
    risk instead of pure classifier performance.
    """

    def __init__(self, order_loader: OrderLoader, config: BoosterConfig | None = None):
        self.order_loader = order_loader
        self.config = config or BoosterConfig()

    def run(
        self,
        order_list: list[OrderId],
        top_k_per_order: int | None = None,
        label_col: str | None = None,
        group_cols: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        config = self._with_overrides(label_col=label_col, group_cols=group_cols)
        top_k = top_k_per_order or config.ranking.final_top_k_per_order
        results = []

        for order_id in order_list:
            order_result = self.run_order(order_id, top_k_per_order=top_k, config=config)
            results.append(order_result)

        combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        if config.output_dir is not None and not combined.empty:
            save_combined_report(combined, config.output_dir)
        return combined

    def run_order(
        self,
        order_id: OrderId,
        top_k_per_order: int | None = None,
        config: BoosterConfig | None = None,
    ) -> pd.DataFrame:
        config = config or self.config
        top_k = top_k_per_order or config.ranking.final_top_k_per_order

        raw = self.order_loader(order_id)
        X, y, groups = self._split_order_frame(raw, config)
        if X.empty:
            return pd.DataFrame()

        presence = detect_asymmetric_signals(X, y, config.asymmetric)
        quality = profile_quality(X, y, config.quality, presence)
        keep_features = quality.index[quality["quality_keep"]].tolist()
        if not keep_features:
            return self._empty_order_result(order_id)

        X_keep = X[keep_features]
        evidence = quality.join(presence, how="left")
        stats = compute_statistical_evidence(X_keep, y, config.statistical, presence.loc[keep_features])
        evidence = evidence.join(stats, how="left")
        evidence = evidence.loc[keep_features]

        preliminary = rank_evidence(evidence, config)
        candidate_features = preliminary.head(config.ranking.preliminary_top_n).index.tolist()

        groupwise = compute_groupwise_stability(X_keep, y, groups, candidate_features)
        confounding = check_group_confounding(X_keep, y, groups, candidate_features, config.confounding)
        xai_features = candidate_features[: config.catboost.max_features_per_order]
        xai = run_catboost_xai_probe(X_keep, y, xai_features, config.catboost)

        evidence = evidence.join(groupwise, how="left")
        evidence = evidence.join(confounding, how="left")
        evidence = evidence.join(xai, how="left", rsuffix="_xai")
        reranked = rank_evidence(evidence, config)

        redundancy = assign_redundancy_groups(X_keep, reranked, config.redundancy)
        evidence = evidence.drop(columns=[col for col in redundancy.columns if col in evidence.columns], errors="ignore")
        evidence = evidence.join(redundancy, how="left")
        final = rank_evidence(evidence, config).head(top_k).copy()

        final.insert(0, "order_id", order_id)
        final.insert(1, "final_rank", range(1, len(final) + 1))
        final = final.reset_index().rename(columns={"index": "feature_name"})
        final = _order_output_columns(final)

        if config.output_dir is not None and not final.empty:
            save_order_report(final, config.output_dir, order_id)
        return final

    def _with_overrides(
        self,
        label_col: str | None = None,
        group_cols: tuple[str, ...] | None = None,
    ) -> BoosterConfig:
        config = self.config
        if label_col is not None:
            config = replace(config, label_col=label_col)
        if group_cols is not None:
            config = replace(config, group_cols=group_cols)
        return config

    @staticmethod
    def _split_order_frame(
        df: pd.DataFrame,
        config: BoosterConfig,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
        if config.label_col not in df.columns:
            raise ValueError(f"label_col '{config.label_col}' is missing from order data")

        y = as_binary_target(df[config.label_col], config.positive_label)
        group_cols = tuple(col for col in config.group_cols if col in df.columns)
        groups = df[list(group_cols)].copy() if group_cols else None
        feature_cols = infer_feature_columns(
            df,
            config.label_col,
            group_cols,
            config.sample_id_cols,
            config.exclude_cols,
        )
        X = df[feature_cols].copy()
        return X, y, groups

    @staticmethod
    def _empty_order_result(order_id: Any) -> pd.DataFrame:
        return pd.DataFrame({"order_id": [order_id], "warning": ["no_features_after_quality_filter"]})


def _order_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "order_id",
        "final_rank",
        "feature_name",
        "final_score",
        "evidence_reason",
        "presence_type",
        "bad_coverage",
        "good_coverage",
        "direction",
        "effect_size",
        "fdr_pvalue",
        "bootstrap_stability_score",
        "groupwise_stability_score",
        "xai_consistency_score",
        "shap_abs_mean",
        "confounding_risk_score",
        "redundancy_group",
        "redundancy_penalty_score",
        "data_quality_penalty_score",
        "quality_warning",
        "confounding_warning",
        "xai_warning",
    ]
    existing = [col for col in preferred if col in df.columns]
    extras = [col for col in df.columns if col not in existing]
    return df[existing + extras]
