from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_toy_feature_metadata(metadata_path: Path | str) -> pd.DataFrame:
    """Load toyset feature metadata used as synthetic ground truth."""

    metadata = pd.read_csv(metadata_path)
    if "feature_name" not in metadata.columns:
        raise ValueError("metadata must contain a feature_name column")
    if "signal_subtype" not in metadata.columns:
        metadata["signal_subtype"] = metadata["feature_name"].map(infer_signal_subtype)
    return metadata


def infer_signal_subtype(feature_name: str) -> str:
    """Infer a more detailed planted-signal subtype from feature names."""

    name = str(feature_name)
    if name.startswith("sensor_a_"):
        return "defect_a_sensor"
    if name.startswith("measure_a_"):
        return "defect_a_measure"
    if name.startswith("midproc_a_"):
        return "defect_a_midproc_count"
    if name.startswith("nonlinear_a_elbow"):
        return "defect_a_nonlinear_elbow"
    if name.startswith("nonlinear_a_saturation"):
        return "defect_a_nonlinear_saturation"
    if name.startswith("nonlinear_a_u_shape"):
        return "defect_a_nonlinear_u_shape"
    if name.startswith("interaction_a_"):
        return "defect_a_interaction"
    if name.startswith("bad_only_a_"):
        return "defect_a_bad_only_sparse"
    if name.startswith("good_only_stable_signature_"):
        return "defect_a_good_only_sparse"
    if name.startswith("tool_confounded"):
        return "tool_confounded"
    if name.startswith("defect_b_"):
        return "other_defect_b"
    if name.startswith("defect_c_"):
        return "other_defect_c"
    if name.startswith("defect_d_"):
        return "other_defect_d"
    if "noise" in name:
        return "noise"
    return "unknown"


def annotate_with_toy_truth(result: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach toy truth labels to a selector or SHAP-delta result table."""

    if result.empty:
        return result.copy()
    if "feature_name" not in result.columns:
        raise ValueError("result must contain a feature_name column")

    meta_cols = ["feature_name", "feature_family", "domain_prior_score", "signal_subtype"]
    available_cols = [col for col in meta_cols if col in metadata.columns]
    merged = result.merge(metadata[available_cols], on="feature_name", how="left")
    merged["signal_subtype"] = merged["signal_subtype"].fillna(merged["feature_name"].map(infer_signal_subtype))
    merged["feature_family"] = merged["feature_family"].fillna(merged["signal_subtype"].map(_family_from_subtype))

    merged["toy_truth_is_defect_a"] = merged["feature_family"].eq("planted_defect_a")
    merged["toy_truth_is_nonlinear_a"] = merged["signal_subtype"].isin(
        {
            "defect_a_nonlinear_elbow",
            "defect_a_nonlinear_saturation",
            "defect_a_nonlinear_u_shape",
            "defect_a_interaction",
        }
    )
    merged["toy_truth_is_sparse_a"] = merged["signal_subtype"].isin(
        {"defect_a_bad_only_sparse", "defect_a_good_only_sparse"}
    )
    merged["toy_truth_is_other_defect"] = merged["feature_family"].eq("other_defect_signal")
    merged["toy_truth_is_tool_confounded"] = merged["feature_family"].eq("tool_confounded")
    merged["toy_truth_is_noise"] = merged["feature_family"].eq("noise")
    merged["toy_truth_label"] = np.select(
        [
            merged["toy_truth_is_defect_a"],
            merged["toy_truth_is_other_defect"],
            merged["toy_truth_is_tool_confounded"],
            merged["toy_truth_is_noise"],
        ],
        ["target_defect_a", "other_defect", "tool_confounded", "noise"],
        default="unknown",
    )
    return merged


def summarize_toy_truth_hits(
    result: pd.DataFrame,
    metadata: pd.DataFrame,
    top_k: int | None = None,
    rank_col: str | None = None,
) -> pd.DataFrame:
    """Summarize synthetic truth hits per order/method."""

    annotated = result.copy() if "toy_truth_label" in result.columns else annotate_with_toy_truth(result, metadata)
    if annotated.empty:
        return pd.DataFrame()

    rank_col = rank_col or _guess_rank_col(annotated)
    if top_k is not None and rank_col in annotated.columns:
        annotated = annotated[pd.to_numeric(annotated[rank_col], errors="coerce") <= top_k]

    group_cols = [col for col in ["order_id", "method"] if col in annotated.columns]
    if not group_cols:
        annotated = annotated.assign(method="all")
        group_cols = ["method"]

    records: list[dict[str, object]] = []
    for keys, group in annotated.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        selected_count = len(group)
        defect_a_count = int(group["toy_truth_is_defect_a"].sum())
        other_defect_count = int(group["toy_truth_is_other_defect"].sum())
        tool_confounded_count = int(group["toy_truth_is_tool_confounded"].sum())
        noise_count = int(group["toy_truth_is_noise"].sum())

        record.update(
            {
                "evaluated_top_k": top_k if top_k is not None else selected_count,
                "selected_count": selected_count,
                "target_defect_a_hit_count": defect_a_count,
                "target_defect_a_precision": defect_a_count / selected_count if selected_count else np.nan,
                "nonlinear_a_hit_count": int(group["toy_truth_is_nonlinear_a"].sum()),
                "sparse_a_hit_count": int(group["toy_truth_is_sparse_a"].sum()),
                "other_defect_hit_count": other_defect_count,
                "tool_confounded_hit_count": tool_confounded_count,
                "noise_hit_count": noise_count,
                "unknown_hit_count": int(group["toy_truth_label"].eq("unknown").sum()),
                "toy_truth_penalty_count": other_defect_count + tool_confounded_count + noise_count,
                "toy_truth_score": _toy_truth_score(group),
                "dominant_signal_subtype": _dominant(group["signal_subtype"]),
            }
        )
        if "shap_delta_abs" in group.columns:
            record["mean_shap_delta_abs"] = float(pd.to_numeric(group["shap_delta_abs"], errors="coerce").mean())
        records.append(record)

    columns = [
        *group_cols,
        "evaluated_top_k",
        "selected_count",
        "target_defect_a_hit_count",
        "target_defect_a_precision",
        "nonlinear_a_hit_count",
        "sparse_a_hit_count",
        "other_defect_hit_count",
        "tool_confounded_hit_count",
        "noise_hit_count",
        "unknown_hit_count",
        "toy_truth_penalty_count",
        "toy_truth_score",
        "dominant_signal_subtype",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)


def save_toy_truth_evaluation(
    result: pd.DataFrame,
    metadata_path: Path | str,
    output_dir: Path | str,
    prefix: str,
    top_k: int | None = None,
    rank_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate and summarize a result table, then save both CSVs."""

    metadata = load_toy_feature_metadata(metadata_path)
    annotated = annotate_with_toy_truth(result, metadata)
    summary = summarize_toy_truth_hits(result, metadata, top_k=top_k, rank_col=rank_col)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_dir / f"{prefix}_toy_truth_annotated.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / f"{prefix}_toy_truth_summary.csv", index=False, encoding="utf-8-sig")
    return annotated, summary


def _family_from_subtype(subtype: str) -> str:
    subtype = str(subtype)
    if subtype.startswith("defect_a_"):
        return "planted_defect_a"
    if subtype.startswith("other_defect_"):
        return "other_defect_signal"
    if subtype == "tool_confounded":
        return "tool_confounded"
    if subtype == "noise":
        return "noise"
    return "unknown"


def _guess_rank_col(df: pd.DataFrame) -> str:
    for col in ["shap_rank", "rank", "final_rank"]:
        if col in df.columns:
            return col
    return ""


def _toy_truth_score(group: pd.DataFrame) -> float:
    if group.empty:
        return np.nan
    reward = group["toy_truth_is_defect_a"].astype(float)
    reward += group["toy_truth_is_nonlinear_a"].astype(float) * 0.20
    reward += group["toy_truth_is_sparse_a"].astype(float) * 0.10
    penalty = group["toy_truth_is_other_defect"].astype(float) * 0.40
    penalty += group["toy_truth_is_tool_confounded"].astype(float) * 0.60
    penalty += group["toy_truth_is_noise"].astype(float) * 1.00
    return float((reward - penalty).mean())


def _dominant(series: pd.Series) -> str:
    counts = series.dropna().astype(str).value_counts()
    return str(counts.index[0]) if not counts.empty else ""
