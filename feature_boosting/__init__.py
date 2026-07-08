"""Residual-based CatBoost feature boosting PoC."""

from .baseline_model import predict_baseline, train_baseline_model
from .config import ExperimentConfig, load_config
from .final_model import evaluate_model_by_groups, train_final_model
from .metrics import mae, r2, rmse
from .overfit import recommend_overfit_safe_settings
from .residual_boosting import ResidualFeatureBooster, ResidualFeatureBoosterConfig

__all__ = [
    "ExperimentConfig",
    "ResidualFeatureBooster",
    "ResidualFeatureBoosterConfig",
    "evaluate_model_by_groups",
    "load_config",
    "mae",
    "predict_baseline",
    "recommend_overfit_safe_settings",
    "r2",
    "rmse",
    "train_baseline_model",
    "train_final_model",
]
