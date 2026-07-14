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


def residual_reduction_from_residuals(residual_before, residual_after) -> dict[str, float]:
    before = np.asarray(residual_before, dtype=float)
    after = np.asarray(residual_after, dtype=float)
    mask = np.isfinite(before) & np.isfinite(after)
    before = before[mask]
    after = after[mask]
    if before.size == 0:
        return {
            "mae_before": float("nan"),
            "mae_after": float("nan"),
            "mae_reduction": float("nan"),
            "rmse_before": float("nan"),
            "rmse_after": float("nan"),
            "rmse_reduction": float("nan"),
        }

    before_mae = float(np.mean(np.abs(before)))
    after_mae = float(np.mean(np.abs(after)))
    before_rmse = float(np.sqrt(np.mean(before**2)))
    after_rmse = float(np.sqrt(np.mean(after**2)))
    return {
        "mae_before": before_mae,
        "mae_after": after_mae,
        "mae_reduction": reduction(before_mae, after_mae),
        "rmse_before": before_rmse,
        "rmse_after": after_rmse,
        "rmse_reduction": reduction(before_rmse, after_rmse),
    }


def residual_reduction_metrics(y_true, pred_before, pred_after) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    before = np.asarray(pred_before, dtype=float)
    after = np.asarray(pred_after, dtype=float)
    mask = np.isfinite(true) & np.isfinite(before) & np.isfinite(after)
    return residual_reduction_from_residuals(true[mask] - before[mask], true[mask] - after[mask])
