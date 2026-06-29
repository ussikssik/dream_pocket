"""Evidence-first feature boosting for mixed-defect semiconductor data."""

from .config import (
    AsymmetricSignalConfig,
    BoosterConfig,
    CatBoostProbeConfig,
    ConfoundingConfig,
    QualityConfig,
    RankingConfig,
    RedundancyConfig,
    StatisticalConfig,
)
from .alternative_selectors import (
    AlternativeFeatureSelector,
    AlternativeSelectorConfig,
    summarize_method_overlap,
)
from .catboost_feature_set_evaluator import (
    CatBoostFeatureSetEvalConfig,
    evaluate_catboost_feature_sets,
    evaluate_catboost_global_feature_sets,
    plot_shap_delta_top_features,
)
from .toy_truth_evaluator import (
    annotate_with_toy_truth,
    load_toy_feature_metadata,
    save_toy_truth_evaluation,
    summarize_toy_truth_hits,
)
from .order_runner import DefectAFeatureEvidenceBooster

__all__ = [
    "AlternativeFeatureSelector",
    "AlternativeSelectorConfig",
    "AsymmetricSignalConfig",
    "BoosterConfig",
    "CatBoostFeatureSetEvalConfig",
    "CatBoostProbeConfig",
    "ConfoundingConfig",
    "DefectAFeatureEvidenceBooster",
    "QualityConfig",
    "RankingConfig",
    "RedundancyConfig",
    "StatisticalConfig",
    "annotate_with_toy_truth",
    "evaluate_catboost_feature_sets",
    "evaluate_catboost_global_feature_sets",
    "load_toy_feature_metadata",
    "plot_shap_delta_top_features",
    "save_toy_truth_evaluation",
    "summarize_toy_truth_hits",
    "summarize_method_overlap",
]
