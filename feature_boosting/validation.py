from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FeatureFilterConfig


LEAKAGE_TOKENS = ("target", "yield", "label")


def validate_input_columns(
    base_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    base_feature_cols: list[str],
    *,
    id_col: str,
    target_col: str,
    split_col: str,
) -> None:
    missing_base = [col for col in (id_col, target_col, split_col) if col not in base_df.columns]
    if missing_base:
        raise ValueError(f"base dataset is missing required columns: {missing_base}")
    missing_features = [col for col in base_feature_cols if col not in base_df.columns]
    if missing_features:
        raise ValueError(f"base_feature_cols contains columns absent from base dataset: {missing_features[:10]}")
    if id_col not in candidate_df.columns:
        raise ValueError(f"candidate features are missing id column {id_col!r}")
    if not set(base_df[id_col].astype(str)).intersection(set(candidate_df[id_col].astype(str))):
        raise ValueError(f"candidate features cannot be matched to base dataset on {id_col!r}")


def validate_defect_groups(
    defect_id: str,
    bad_ids: set[str],
    good_ids: set[str],
    df: pd.DataFrame,
    *,
    id_col: str,
    split_col: str,
    min_valid_bad_samples: int = 1,
) -> list[str]:
    warnings: list[str] = []
    overlap = bad_ids & good_ids
    if overlap:
        raise ValueError(f"[{defect_id}] bad/good groups overlap: {len(overlap)} sample ids")
    ids = df[id_col].astype(str)
    split = df[split_col].astype(str)
    valid_bad = int(((split == "valid") & ids.isin(bad_ids)).sum())
    test_bad = int(((split == "test") & ids.isin(bad_ids)).sum())
    valid_good = int(((split == "valid") & ids.isin(good_ids)).sum())
    test_good = int(((split == "test") & ids.isin(good_ids)).sum())
    if valid_bad < min_valid_bad_samples:
        warnings.append(f"low_valid_bad_samples:{valid_bad}")
    if test_bad == 0:
        warnings.append("no_test_bad_samples")
    if valid_good == 0:
        warnings.append("no_valid_good_samples")
    if test_good == 0:
        warnings.append("no_test_good_samples")
    return warnings


def profile_candidate_features(
    df: pd.DataFrame,
    candidate_cols: list[str],
    *,
    id_col: str,
    split_col: str,
    bad_ids: set[str],
    good_ids: set[str],
    config: FeatureFilterConfig,
    protected_cols: set[str] | None = None,
) -> pd.DataFrame:
    protected_cols = protected_cols or set()
    ids = df[id_col].astype(str)
    bad_mask = ids.isin(bad_ids)
    good_mask = ids.isin(good_ids)
    rows = []
    for feature in candidate_cols:
        if feature not in df.columns:
            rows.append(_quality_row(feature, fail_reason="missing_column", config=config))
            continue
        series = df[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = series.notna()
        missing_rate = float(1.0 - non_null.mean()) if len(series) else 1.0
        unique_count = int(numeric.dropna().nunique())
        train_coverage = _coverage(numeric[df[split_col].astype(str) == "train"])
        valid_coverage = _coverage(numeric[df[split_col].astype(str) == "valid"])
        test_coverage = _coverage(numeric[df[split_col].astype(str) == "test"])
        bad_coverage = _coverage(numeric[bad_mask])
        good_coverage = _coverage(numeric[good_mask])

        reasons = []
        if feature in protected_cols or _looks_like_leakage(feature):
            reasons.append("leakage_like_feature")
        if non_null.sum() == 0:
            reasons.append("all_missing")
        elif numeric.notna().sum() < non_null.sum():
            reasons.append("non_numeric")
        if unique_count < config.min_unique_values:
            reasons.append("constant_feature")
        if missing_rate > config.max_missing_rate:
            reasons.append("high_missing_rate")
        if bad_coverage < config.min_bad_coverage:
            reasons.append("low_bad_coverage")
        if good_coverage < config.min_good_coverage:
            reasons.append("low_good_coverage")
        if train_coverage <= 0 or valid_coverage <= 0 or test_coverage <= 0:
            reasons.append("missing_split_values")

        rows.append(
            {
                "feature_name": feature,
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "train_coverage": train_coverage,
                "valid_coverage": valid_coverage,
                "test_coverage": test_coverage,
                "bad_coverage": bad_coverage,
                "good_coverage": good_coverage,
                "is_pass": len(reasons) == 0,
                "fail_reason": ";".join(dict.fromkeys(reasons)),
                **{f"filter_{k}": v for k, v in asdict(config).items()},
            }
        )
    return pd.DataFrame(rows)


def write_quality_summary(summary: pd.DataFrame, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")


def _quality_row(feature: str, fail_reason: str, config: FeatureFilterConfig) -> dict[str, object]:
    return {
        "feature_name": feature,
        "missing_rate": np.nan,
        "unique_count": 0,
        "train_coverage": 0.0,
        "valid_coverage": 0.0,
        "test_coverage": 0.0,
        "bad_coverage": 0.0,
        "good_coverage": 0.0,
        "is_pass": False,
        "fail_reason": fail_reason,
        **{f"filter_{k}": v for k, v in asdict(config).items()},
    }


def _coverage(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").notna().mean())


def _looks_like_leakage(feature: str) -> bool:
    lower = feature.lower()
    return any(token in lower for token in LEAKAGE_TOKENS)
