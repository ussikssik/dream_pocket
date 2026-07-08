from __future__ import annotations

from typing import Any


def recommend_overfit_safe_settings(
    residual_model_params: dict[str, Any],
    *,
    n_train_rows: int,
    n_candidate_features: int,
    n_base_features: int = 0,
) -> dict[str, Any]:
    """Return conservative residual-model and guard settings for early trials.

    The residual booster fits one small model per candidate feature. Even so,
    many candidate features create a multiple-comparison risk: train residual
    can shrink while validation/test residual gets worse. This helper lowers
    residual-model capacity and tightens validation guards when the dataset is
    small or the candidate count is high relative to train rows.
    """
    params = dict(residual_model_params or {})
    n_train = max(int(n_train_rows), 1)
    n_candidates = max(int(n_candidate_features), 0)
    pressure = n_candidates / n_train

    if n_train < 1000:
        profile = "tiny_train"
        limits = {"iterations": 120, "depth": 2, "learning_rate": 0.03, "l2_leaf_reg": 20.0, "valid_train_gap": 0.15}
    elif n_train < 5000 or pressure >= 1.0:
        profile = "small_or_high_pressure"
        limits = {"iterations": 180, "depth": 2, "learning_rate": 0.035, "l2_leaf_reg": 15.0, "valid_train_gap": 0.20}
    elif pressure >= 0.30:
        profile = "moderate_candidate_pressure"
        limits = {"iterations": 240, "depth": 3, "learning_rate": 0.04, "l2_leaf_reg": 10.0, "valid_train_gap": 0.25}
    else:
        profile = "standard"
        limits = {"iterations": 300, "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 6.0, "valid_train_gap": 0.25}

    changes: list[str] = []
    _cap_int(params, "iterations", int(limits["iterations"]), changes)
    _cap_int(params, "depth", int(limits["depth"]), changes)
    _cap_float(params, "learning_rate", float(limits["learning_rate"]), changes)
    _floor_regularization(params, float(limits["l2_leaf_reg"]), changes)
    params.setdefault("early_stopping_rounds", 30)
    params.setdefault("thread_count", 1)
    params.setdefault("verbose", False)

    guard = {
        "overfit_guard_enabled": True,
        "overfit_guard_metric_scope": "bad",
        "overfit_guard_min_valid_rmse_reduction": 0.0,
        "overfit_guard_max_valid_after_over_baseline": 1.0,
        "overfit_guard_max_valid_train_gap": float(limits["valid_train_gap"]),
        "overfit_guard_use_test": False,
        "overfit_guard_max_test_after_over_baseline": 1.05,
    }

    return {
        "profile": profile,
        "train_rows": n_train,
        "candidate_features": n_candidates,
        "base_features": int(n_base_features),
        "candidate_features_per_train_row": pressure,
        "residual_model_params": params,
        "guard": guard,
        "summary_rows": [
            {"setting": "profile", "value": profile},
            {"setting": "train_rows", "value": n_train},
            {"setting": "candidate_features", "value": n_candidates},
            {"setting": "base_features", "value": int(n_base_features)},
            {"setting": "candidate_features_per_train_row", "value": round(pressure, 4)},
            {"setting": "residual_param_changes", "value": "; ".join(changes) if changes else "none"},
            {"setting": "overfit_guard_max_valid_after_over_baseline", "value": guard["overfit_guard_max_valid_after_over_baseline"]},
            {"setting": "overfit_guard_max_valid_train_gap", "value": guard["overfit_guard_max_valid_train_gap"]},
            {"setting": "overfit_guard_use_test", "value": guard["overfit_guard_use_test"]},
            {"setting": "overfit_guard_max_test_after_over_baseline", "value": guard["overfit_guard_max_test_after_over_baseline"]},
        ],
    }


def _cap_int(params: dict[str, Any], key: str, cap: int, changes: list[str]) -> None:
    current = _to_float(params.get(key))
    if current is None or current > cap:
        old = params.get(key, "<unset>")
        params[key] = cap
        changes.append(f"{key}: {old} -> {cap}")


def _cap_float(params: dict[str, Any], key: str, cap: float, changes: list[str]) -> None:
    current = _to_float(params.get(key))
    if current is None or current > cap:
        old = params.get(key, "<unset>")
        params[key] = cap
        changes.append(f"{key}: {old} -> {cap}")


def _floor_regularization(params: dict[str, Any], floor: float, changes: list[str]) -> None:
    backend = str(params.get("backend", "auto")).lower()
    key = "l2" if backend in {"numpy", "numpy_ridge", "linear"} else "l2_leaf_reg"
    current = _to_float(params.get(key))
    if current is None or current < floor:
        old = params.get(key, "<unset>")
        params[key] = floor
        changes.append(f"{key}: {old} -> {floor}")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
