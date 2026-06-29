from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityConfig:
    """Feature quality rules that avoid dropping one-sided signals."""

    min_total_non_null_count: int = 20
    min_group_non_null_count: int = 5
    min_unique_non_null: int = 2
    near_constant_top_ratio: float = 0.995
    warn_missing_rate: float = 0.95
    drop_all_null: bool = True
    drop_constant: bool = True


@dataclass(frozen=True)
class AsymmetricSignalConfig:
    """Rules for presence-only or sparse asymmetric signals."""

    sparse_presence_rate: float = 0.01
    min_presence_delta: float = 0.10
    min_presence_ratio: float = 3.0


@dataclass(frozen=True)
class StatisticalConfig:
    """Statistical evidence settings."""

    alpha: float = 0.05
    numeric_unique_threshold: int = 10
    max_categories: int = 200
    bootstrap_rounds: int = 30
    bootstrap_sample_frac: float = 0.80
    bootstrap_min_abs_effect: float = 0.10
    random_state: int = 42


@dataclass(frozen=True)
class CatBoostProbeConfig:
    """CatBoost is a probe for XAI evidence, not a hard performance gate."""

    enabled: bool = True
    max_features_per_order: int = 500
    max_rows: int = 20000
    shap_sample_size: int = 5000
    iterations: int = 300
    learning_rate: float = 0.05
    depth: int = 4
    l2_leaf_reg: float = 6.0
    random_seed: int = 42
    thread_count: int = -1
    verbose: bool = False


@dataclass(frozen=True)
class ConfoundingConfig:
    """Group-level checks for lot/tool/product/chamber confounding."""

    enabled: bool = True
    high_risk_threshold: float = 0.60
    medium_risk_threshold: float = 0.35
    max_groups: int = 200


@dataclass(frozen=True)
class RedundancyConfig:
    """Greedy correlation clustering for near-duplicate candidates."""

    enabled: bool = True
    max_features: int = 500
    correlation_threshold: float = 0.95


@dataclass(frozen=True)
class RankingConfig:
    """Weights for evidence ranking. Model performance is intentionally absent."""

    contrast_strength_weight: float = 0.30
    direction_consistency_weight: float = 0.12
    bootstrap_stability_weight: float = 0.12
    groupwise_stability_weight: float = 0.10
    xai_consistency_weight: float = 0.16
    asymmetric_presence_weight: float = 0.12
    domain_metadata_weight: float = 0.04
    confounding_penalty_weight: float = 0.14
    redundancy_penalty_weight: float = 0.08
    data_quality_penalty_weight: float = 0.10
    preliminary_top_n: int = 2000
    final_top_k_per_order: int = 10


@dataclass(frozen=True)
class BoosterConfig:
    """Top-level configuration for order-wise evidence boosting."""

    label_col: str = "target_bad_a"
    positive_label: Any = 1
    group_cols: tuple[str, ...] = ()
    sample_id_cols: tuple[str, ...] = ()
    feature_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_dir: Path | None = None
    quality: QualityConfig = field(default_factory=QualityConfig)
    asymmetric: AsymmetricSignalConfig = field(default_factory=AsymmetricSignalConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    catboost: CatBoostProbeConfig = field(default_factory=CatBoostProbeConfig)
    confounding: ConfoundingConfig = field(default_factory=ConfoundingConfig)
    redundancy: RedundancyConfig = field(default_factory=RedundancyConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
