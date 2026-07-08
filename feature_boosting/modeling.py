from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 3000,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "early_stopping_rounds": 100,
    "verbose": 200,
}


class NumpyRidgeRegressor:
    """Small deterministic fallback used when CatBoost is unavailable."""

    def __init__(self, l2: float = 1e-6):
        self.l2 = float(l2)
        self.columns_: list[str] = []
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.dummy_columns_: list[str] = []
        self.coef_: np.ndarray | None = None
        self.backend = "numpy_ridge"

    def fit(self, X: pd.DataFrame, y) -> "NumpyRidgeRegressor":
        self.columns_ = list(X.columns)
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        for col in self.columns_:
            converted = pd.to_numeric(X[col], errors="coerce")
            non_null = int(X[col].notna().sum())
            if non_null == 0 or converted.notna().sum() >= max(1, int(0.95 * non_null)):
                self.numeric_cols_.append(col)
                median = float(converted.median()) if converted.notna().any() else 0.0
                self.medians_[col] = median
            else:
                self.categorical_cols_.append(col)
        matrix = self._matrix(X, fit=True)
        target = np.asarray(y, dtype=float)
        mask = np.isfinite(target)
        matrix = matrix[mask]
        target = target[mask]
        if matrix.shape[0] == 0:
            raise ValueError("no finite target rows")
        penalty = self.l2 * np.eye(matrix.shape[1])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(matrix.T @ matrix + penalty) @ matrix.T @ target
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        matrix = self._matrix(X, fit=False)
        return matrix @ self.coef_

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "backend": self.backend,
            "columns": self.columns_,
            "numeric_cols": self.numeric_cols_,
            "categorical_cols": self.categorical_cols_,
            "medians": self.medians_,
            "dummy_columns": self.dummy_columns_,
            "coef": self.coef_.tolist() if self.coef_ is not None else None,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _matrix(self, X: pd.DataFrame, *, fit: bool) -> np.ndarray:
        frame = pd.DataFrame(index=X.index)
        for col in self.numeric_cols_:
            values = pd.to_numeric(X[col], errors="coerce") if col in X.columns else pd.Series(np.nan, index=X.index)
            frame[col] = values.fillna(self.medians_.get(col, 0.0)).astype(float)
        if self.categorical_cols_:
            cat = X.reindex(columns=self.categorical_cols_).astype("object").where(
                X.reindex(columns=self.categorical_cols_).notna(), "__MISSING__"
            )
            dummies = pd.get_dummies(cat, columns=self.categorical_cols_, dummy_na=False, dtype=float)
            if fit:
                self.dummy_columns_ = list(dummies.columns)
            dummies = dummies.reindex(columns=self.dummy_columns_, fill_value=0.0)
            frame = pd.concat([frame, dummies], axis=1)
        values = frame.to_numpy(dtype=float)
        return np.column_stack([np.ones(len(frame), dtype=float), values])


def fit_regressor(
    train_df: pd.DataFrame,
    y_train,
    valid_df: pd.DataFrame | None,
    y_valid,
    feature_cols: list[str],
    params: dict[str, Any] | None = None,
):
    params = dict(params or {})
    backend = str(params.pop("backend", "auto")).lower()
    if backend in {"numpy", "numpy_ridge", "linear"}:
        return NumpyRidgeRegressor(l2=_numpy_l2(params)).fit(train_df[feature_cols], y_train)

    try:
        from catboost import CatBoostRegressor, Pool
    except Exception:
        if backend == "catboost":
            raise RuntimeError("catboost is not installed") from None
        return NumpyRidgeRegressor(l2=_numpy_l2(params)).fit(train_df[feature_cols], y_train)

    model_params = dict(DEFAULT_CATBOOST_PARAMS)
    model_params.update(params)
    model_params.setdefault("allow_writing_files", False)
    cat_features = _categorical_feature_indices(train_df[feature_cols])
    train_pool = Pool(_prepare_catboost_frame(train_df[feature_cols]), y_train, cat_features=cat_features)
    eval_set = None
    if valid_df is not None and y_valid is not None:
        eval_set = Pool(_prepare_catboost_frame(valid_df[feature_cols]), y_valid, cat_features=cat_features)
    model = CatBoostRegressor(**model_params)
    model.fit(train_pool, eval_set=eval_set, use_best_model=eval_set is not None)
    model.backend = "catboost"
    return model


def predict_regressor(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if getattr(model, "backend", "") == "catboost":
        try:
            from catboost import Pool
        except Exception:
            return np.asarray(model.predict(_prepare_catboost_frame(df[feature_cols])), dtype=float)
        cat_features = _categorical_feature_indices(df[feature_cols])
        return np.asarray(model.predict(Pool(_prepare_catboost_frame(df[feature_cols]), cat_features=cat_features)), dtype=float)
    return np.asarray(model.predict(df[feature_cols]), dtype=float)


def _categorical_feature_indices(frame: pd.DataFrame) -> list[int]:
    return [
        idx
        for idx, col in enumerate(frame.columns)
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_categorical_dtype(frame[col])
    ]


def _numpy_l2(params: dict[str, Any]) -> float:
    return float(params.pop("l2", params.pop("l2_leaf_reg", 1e-6)))


def _prepare_catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for col in prepared.columns:
        if pd.api.types.is_object_dtype(prepared[col]) or pd.api.types.is_categorical_dtype(prepared[col]):
            prepared[col] = prepared[col].astype("object").where(prepared[col].notna(), "__MISSING__")
    return prepared
