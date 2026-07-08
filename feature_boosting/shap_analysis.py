from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_shap_summary(
    model,
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    id_col: str,
    defects: dict[str, dict[str, set[str]]],
    selected_features: pd.DataFrame,
    max_samples: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    if not hasattr(model, "get_feature_importance"):
        return _skipped_summary(selected_features, "model_has_no_shap_api")

    selected_lookup = _selected_lookup(selected_features)
    groups = [("global", "global", np.ones(len(df), dtype=bool))]
    ids = df[id_col].astype(str)
    for defect_id, defect_groups in defects.items():
        groups.append((defect_id, "bad", ids.isin(defect_groups.get("bad", set())).to_numpy()))
        groups.append((defect_id, "good", ids.isin(defect_groups.get("good", set())).to_numpy()))

    rows: list[pd.DataFrame] = []
    for defect_id, group_name, mask in groups:
        if not mask.any():
            continue
        subset = df.loc[mask, feature_cols]
        if len(subset) > max_samples:
            subset = subset.sample(max_samples, random_state=random_seed)
        try:
            values = _catboost_shap_values(model, subset)
        except Exception:
            rows.append(_group_skipped(feature_cols, defect_id, group_name, "shap_failed", selected_lookup))
            continue
        mean_abs = np.nanmean(np.abs(values), axis=0)
        report = pd.DataFrame(
            {
                "defect_id": defect_id,
                "group": group_name,
                "feature_name": feature_cols,
                "mean_abs_shap": mean_abs,
            }
        )
        report = report.sort_values("mean_abs_shap", ascending=False, na_position="last").reset_index(drop=True)
        report["shap_rank"] = np.arange(1, len(report) + 1)
        report["is_selected_feature"] = report["feature_name"].isin(selected_lookup)
        report["selected_round"] = report["feature_name"].map(selected_lookup)
        report["status"] = "ok"
        rows.append(report)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _catboost_shap_values(model, frame: pd.DataFrame) -> np.ndarray:
    try:
        from catboost import Pool
    except Exception:
        values = model.get_feature_importance(frame, type="ShapValues")
    else:
        cat_features = [
            idx
            for idx, col in enumerate(frame.columns)
            if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_categorical_dtype(frame[col])
        ]
        prepared = frame.copy()
        for col in prepared.columns:
            if pd.api.types.is_object_dtype(prepared[col]) or pd.api.types.is_categorical_dtype(prepared[col]):
                prepared[col] = prepared[col].astype("object").where(prepared[col].notna(), "__MISSING__")
        values = model.get_feature_importance(Pool(prepared, cat_features=cat_features), type="ShapValues")
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    if values.shape[1] == len(frame.columns) + 1:
        values = values[:, :-1]
    return values


def _selected_lookup(selected_features: pd.DataFrame) -> dict[str, Any]:
    if selected_features is None or selected_features.empty:
        return {}
    result = {}
    for _, row in selected_features.iterrows():
        feature = str(row.get("feature_name", ""))
        if feature and feature not in result:
            result[feature] = row.get("round", np.nan)
    return result


def _skipped_summary(selected_features: pd.DataFrame, reason: str) -> pd.DataFrame:
    if selected_features is None or selected_features.empty:
        return pd.DataFrame([{"status": reason}])
    frame = selected_features[["defect_id", "feature_name", "round"]].copy()
    frame["group"] = "global"
    frame["mean_abs_shap"] = np.nan
    frame["shap_rank"] = np.nan
    frame["is_selected_feature"] = True
    frame["selected_round"] = frame["round"]
    frame["status"] = reason
    return frame.drop(columns=["round"])


def _group_skipped(
    feature_cols: list[str],
    defect_id: str,
    group_name: str,
    reason: str,
    selected_lookup: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.DataFrame({"feature_name": feature_cols})
    frame["defect_id"] = defect_id
    frame["group"] = group_name
    frame["mean_abs_shap"] = np.nan
    frame["shap_rank"] = np.nan
    frame["is_selected_feature"] = frame["feature_name"].isin(selected_lookup)
    frame["selected_round"] = frame["feature_name"].map(selected_lookup)
    frame["status"] = reason
    return frame
