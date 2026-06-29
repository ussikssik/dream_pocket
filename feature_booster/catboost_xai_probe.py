from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CatBoostProbeConfig


def run_catboost_xai_probe(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_features: list[str],
    config: CatBoostProbeConfig,
) -> pd.DataFrame:
    """Train a small CatBoost probe and summarize SHAP direction consistency."""

    base = pd.DataFrame(index=pd.Index(candidate_features, name="feature_name"))
    base["xai_available"] = False
    base["xai_consistency_score"] = 0.0
    base["shap_abs_mean"] = np.nan
    base["shap_bad_mean"] = np.nan
    base["shap_good_mean"] = np.nan
    base["shap_bad_positive_rate"] = np.nan
    base["shap_good_negative_rate"] = np.nan
    base["xai_warning"] = ""

    if not config.enabled or not candidate_features:
        base["xai_warning"] = "catboost_probe_disabled"
        return base

    try:
        from catboost import CatBoostClassifier, Pool
    except Exception:
        base["xai_warning"] = "catboost_not_installed"
        return base

    y = y.astype(int)
    if y.nunique() < 2:
        base["xai_warning"] = "single_class_target"
        return base

    features = candidate_features[: config.max_features_per_order]
    work = X[features].copy()
    if len(work) > config.max_rows:
        work = work.sample(n=config.max_rows, random_state=config.random_seed)
        y_work = y.loc[work.index]
    else:
        y_work = y

    cat_features = [
        idx
        for idx, col in enumerate(work.columns)
        if pd.api.types.is_object_dtype(work[col]) or pd.api.types.is_categorical_dtype(work[col])
    ]
    work = _prepare_for_catboost(work, cat_features)

    try:
        pool = Pool(work, y_work, cat_features=cat_features)
        model = CatBoostClassifier(
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            loss_function="Logloss",
            random_seed=config.random_seed,
            thread_count=config.thread_count,
            verbose=config.verbose,
            allow_writing_files=False,
        )
        model.fit(pool)
    except Exception as exc:
        base.loc[features, "xai_warning"] = f"catboost_fit_failed:{type(exc).__name__}"
        return base

    shap_work = work
    y_shap = y_work
    if len(shap_work) > config.shap_sample_size:
        shap_work = shap_work.sample(n=config.shap_sample_size, random_state=config.random_seed + 1)
        y_shap = y_work.loc[shap_work.index]

    try:
        shap_values = model.get_feature_importance(
            Pool(shap_work, y_shap, cat_features=cat_features),
            type="ShapValues",
        )
    except Exception as exc:
        base.loc[features, "xai_warning"] = f"shap_failed:{type(exc).__name__}"
        return base

    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape[1] == len(features) + 1:
        values = values[:, :-1]

    bad_mask = y_shap.to_numpy(dtype=int) == 1
    good_mask = ~bad_mask
    for idx, feature in enumerate(features):
        feature_shap = values[:, idx]
        bad_values = feature_shap[bad_mask]
        good_values = feature_shap[good_mask]
        bad_positive_rate = _rate(bad_values > 0)
        good_negative_rate = _rate(good_values < 0)
        consistency = np.nanmean([bad_positive_rate, good_negative_rate])

        base.loc[feature, "xai_available"] = True
        base.loc[feature, "xai_consistency_score"] = float(np.nan_to_num(consistency, nan=0.0))
        base.loc[feature, "shap_abs_mean"] = float(np.nanmean(np.abs(feature_shap)))
        base.loc[feature, "shap_bad_mean"] = float(np.nanmean(bad_values)) if len(bad_values) else np.nan
        base.loc[feature, "shap_good_mean"] = float(np.nanmean(good_values)) if len(good_values) else np.nan
        base.loc[feature, "shap_bad_positive_rate"] = bad_positive_rate
        base.loc[feature, "shap_good_negative_rate"] = good_negative_rate

    return base


def _prepare_for_catboost(X: pd.DataFrame, cat_feature_indices: list[int]) -> pd.DataFrame:
    prepared = X.copy()
    cat_cols = {prepared.columns[idx] for idx in cat_feature_indices}
    for col in prepared.columns:
        if col in cat_cols:
            prepared[col] = prepared[col].astype("object").where(prepared[col].notna(), "__MISSING__")
        else:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    return prepared


def _rate(mask: np.ndarray) -> float:
    if len(mask) == 0:
        return np.nan
    return float(np.mean(mask))
