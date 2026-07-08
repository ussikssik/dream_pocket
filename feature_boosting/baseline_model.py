from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import regression_metrics
from .modeling import fit_regressor, predict_regressor


def train_baseline_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    catboost_params: dict | None = None,
):
    return fit_regressor(
        train_df,
        train_df[target_col].to_numpy(dtype=float),
        valid_df,
        valid_df[target_col].to_numpy(dtype=float),
        feature_cols,
        catboost_params,
    )


def predict_baseline(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return predict_regressor(model, df, feature_cols)


def add_baseline_predictions(
    model,
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    pred_col: str = "baseline_pred",
    residual_col: str = "baseline_residual",
) -> pd.DataFrame:
    result = df.copy()
    pred = predict_baseline(model, result, feature_cols)
    result[pred_col] = pred
    result[residual_col] = result[target_col].to_numpy(dtype=float) - pred
    return result


def metrics_by_split(df: pd.DataFrame, *, target_col: str, pred_col: str, split_col: str) -> pd.DataFrame:
    rows = []
    for split, group in df.groupby(split_col, sort=False):
        row = {"split": split, "n_samples": len(group)}
        row.update(regression_metrics(group[target_col], group[pred_col]))
        rows.append(row)
    return pd.DataFrame(rows)
