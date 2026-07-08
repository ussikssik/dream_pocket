from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import regression_metrics
from .modeling import fit_regressor, predict_regressor


def train_final_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    catboost_params: dict[str, Any] | None = None,
):
    return fit_regressor(
        train_df,
        train_df[target_col].to_numpy(dtype=float),
        valid_df,
        valid_df[target_col].to_numpy(dtype=float),
        feature_cols,
        catboost_params,
    )


def predict_final(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return predict_regressor(model, df, feature_cols)


def evaluate_model_by_groups(
    df: pd.DataFrame,
    *,
    model_name: str,
    pred_col: str,
    target_col: str,
    split_col: str,
    id_col: str,
    defects: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, split_df in df.groupby(split_col, sort=False):
        rows.append(_metric_row(model_name, "global", "global", split, split_df, target_col, pred_col))
        ids = split_df[id_col].astype(str)
        for defect_id, groups in defects.items():
            bad_df = split_df[ids.isin(groups.get("bad", set()))]
            good_df = split_df[ids.isin(groups.get("good", set()))]
            rows.append(_metric_row(model_name, defect_id, "bad", split, bad_df, target_col, pred_col))
            rows.append(_metric_row(model_name, defect_id, "good", split, good_df, target_col, pred_col))
    return pd.DataFrame(rows)


def _metric_row(
    model_name: str,
    defect_id: str,
    group: str,
    split: object,
    frame: pd.DataFrame,
    target_col: str,
    pred_col: str,
) -> dict[str, object]:
    metrics = regression_metrics(frame[target_col], frame[pred_col]) if len(frame) else {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    return {
        "model_name": model_name,
        "defect_id": defect_id,
        "group": group,
        "split": split,
        "n_samples": len(frame),
        **metrics,
    }
