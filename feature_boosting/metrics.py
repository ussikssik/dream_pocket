from __future__ import annotations

import numpy as np


def _finite_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def mae(y_true, y_pred) -> float:
    true, pred = _finite_pair(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    true, pred = _finite_pair(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def r2(y_true, y_pred) -> float:
    true, pred = _finite_pair(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    denom = float(np.sum((true - np.mean(true)) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((true - pred) ** 2) / denom)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}


def reduction(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    return float(before - after)


def residual_reduction_metrics(y_true, pred_before, pred_after) -> dict[str, float]:
    before_mae = mae(y_true, pred_before)
    after_mae = mae(y_true, pred_after)
    before_rmse = rmse(y_true, pred_before)
    after_rmse = rmse(y_true, pred_after)
    return {
        "mae_before": before_mae,
        "mae_after": after_mae,
        "mae_reduction": reduction(before_mae, after_mae),
        "rmse_before": before_rmse,
        "rmse_after": after_rmse,
        "rmse_reduction": reduction(before_rmse, after_rmse),
    }
