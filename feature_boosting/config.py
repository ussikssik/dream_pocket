from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathConfig:
    base_dataset: str
    candidate_features: str
    base_feature_cols: str
    output_dir: str = "outputs"


@dataclass(frozen=True)
class ColumnConfig:
    id_col: str = "sample_id"
    target_col: str = "yield"
    split_col: str = "split"


@dataclass(frozen=True)
class DefectConfig:
    defect_id: str
    bad_group_path: str
    good_group_path: str


@dataclass(frozen=True)
class FeatureFilterConfig:
    max_missing_rate: float = 0.5
    min_unique_values: int = 2
    min_bad_coverage: float = 0.7
    min_good_coverage: float = 0.7


@dataclass(frozen=True)
class BoostingConfig:
    n_rounds: int = 5
    select_per_round: int = 1
    main_metric: str = "valid_bad_rmse_reduction"
    min_improvement: float = 0.0
    use_test_for_selection: bool = False
    min_valid_bad_samples: int = 1
    show_progress: bool = True
    progress_every: int = 100


@dataclass(frozen=True)
class ShapConfig:
    enabled: bool = True
    max_samples: int = 5000


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    paths: PathConfig
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    defects: tuple[DefectConfig, ...] = ()
    baseline_model: dict[str, Any] = field(default_factory=dict)
    residual_model: dict[str, Any] = field(default_factory=dict)
    feature_filter: FeatureFilterConfig = field(default_factory=FeatureFilterConfig)
    boosting: BoostingConfig = field(default_factory=BoostingConfig)
    final_model: dict[str, Any] = field(default_factory=dict)
    shap: ShapConfig = field(default_factory=ShapConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = _read_mapping(path)
    return config_from_mapping(data)


def config_from_mapping(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        run_id=str(data.get("run_id", "yield_residual_boosting_poc")),
        paths=PathConfig(**data.get("paths", {})),
        columns=ColumnConfig(**data.get("columns", {})),
        defects=tuple(DefectConfig(**item) for item in data.get("defects", [])),
        baseline_model=dict(data.get("baseline_model", {})),
        residual_model=dict(data.get("residual_model", {})),
        feature_filter=FeatureFilterConfig(**data.get("feature_filter", {})),
        boosting=BoostingConfig(**data.get("boosting", {})),
        final_model=dict(data.get("final_model", {})),
        shap=ShapConfig(**data.get("shap", {})),
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for YAML configs. Install dependencies from requirements.txt "
            "or use a .json config."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded
