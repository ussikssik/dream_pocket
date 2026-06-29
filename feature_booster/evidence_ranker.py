from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BoosterConfig
from .utils import robust_minmax, signed_log10_pvalue


def rank_evidence(evidence_df: pd.DataFrame, config: BoosterConfig) -> pd.DataFrame:
    """Combine evidence columns into an interpretable final score."""

    df = evidence_df.copy()
    ranking = config.ranking

    df["abs_effect_score"] = robust_minmax(df.get("effect_size", 0.0).abs())
    df["significance_score"] = robust_minmax(signed_log10_pvalue(df.get("fdr_pvalue", np.nan)))
    df["contrast_strength_score"] = (
        0.65 * df["abs_effect_score"] + 0.35 * df["significance_score"]
    ).clip(0.0, 1.0)
    df["direction_consistency_score"] = _column(df, "direction_consistency_score")
    df["bootstrap_stability_score"] = _column(df, "bootstrap_stability_score")
    df["groupwise_stability_score"] = _column(df, "groupwise_stability_score")
    df["xai_consistency_score"] = _column(df, "xai_consistency_score")
    df["asymmetric_presence_score"] = _column(df, "asymmetric_presence_score")
    df["domain_metadata_score"] = _domain_metadata_score(df.index, config.feature_metadata)
    df["confounding_risk_score"] = _column(df, "confounding_risk_score")
    df["redundancy_penalty_score"] = _column(df, "redundancy_penalty_score")
    df["data_quality_penalty_score"] = _column(df, "data_quality_penalty_score")

    df["final_score"] = (
        ranking.contrast_strength_weight * df["contrast_strength_score"]
        + ranking.direction_consistency_weight * df["direction_consistency_score"]
        + ranking.bootstrap_stability_weight * df["bootstrap_stability_score"]
        + ranking.groupwise_stability_weight * df["groupwise_stability_score"]
        + ranking.xai_consistency_weight * df["xai_consistency_score"]
        + ranking.asymmetric_presence_weight * df["asymmetric_presence_score"]
        + ranking.domain_metadata_weight * df["domain_metadata_score"]
        - ranking.confounding_penalty_weight * df["confounding_risk_score"]
        - ranking.redundancy_penalty_weight * df["redundancy_penalty_score"]
        - ranking.data_quality_penalty_weight * df["data_quality_penalty_score"]
    )
    df["final_score"] = df["final_score"].clip(lower=0.0)
    df["evidence_reason"] = [_reason(row) for _, row in df.iterrows()]
    return df.sort_values(["final_score", "contrast_strength_score"], ascending=False)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _domain_metadata_score(index: pd.Index, metadata: dict[str, dict[str, object]]) -> pd.Series:
    scores = pd.Series(0.0, index=index)
    for feature in index:
        item = metadata.get(feature, {})
        score = item.get("domain_prior_score", 0.0)
        try:
            scores.loc[feature] = float(score)
        except Exception:
            scores.loc[feature] = 0.0
    return scores.clip(0.0, 1.0)


def _reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("asymmetric_presence_score", 0.0) >= 0.7:
        reasons.append(str(row.get("presence_type", "asymmetric_presence")))
    if row.get("contrast_strength_score", 0.0) >= 0.7:
        reasons.append("strong_good_bad_contrast")
    if row.get("bootstrap_stability_score", 0.0) >= 0.7:
        reasons.append("bootstrap_stable")
    if row.get("groupwise_stability_score", 0.0) >= 0.7:
        reasons.append("groupwise_stable")
    if row.get("xai_consistency_score", 0.0) >= 0.7:
        reasons.append("shap_consistent")
    if row.get("confounding_risk_score", 0.0) >= 0.6:
        reasons.append("confounding_warning")
    if row.get("redundancy_penalty_score", 0.0) >= 0.7:
        reasons.append("redundant_with_stronger_feature")
    return "|".join(reasons) if reasons else "moderate_evidence"
