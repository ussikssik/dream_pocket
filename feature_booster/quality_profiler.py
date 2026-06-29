from __future__ import annotations

import numpy as np
import pandas as pd

from .config import QualityConfig
from .utils import compact_warning


def profile_quality(
    X: pd.DataFrame,
    y: pd.Series,
    config: QualityConfig,
    asymmetric_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute quality metadata without discarding one-sided Good/Bad signals."""

    records: list[dict[str, object]] = []
    row_count = len(X)

    for feature in X.columns:
        series = X[feature]
        non_null_count = int(series.notna().sum())
        missing_rate = 1.0 - (non_null_count / row_count if row_count else 0.0)
        unique_non_null = int(series.dropna().nunique())
        top_ratio = _top_value_ratio(series)

        asymmetric = asymmetric_df.loc[feature]
        bad_non_null = int(asymmetric["bad_non_null_count"])
        good_non_null = int(asymmetric["good_non_null_count"])

        keep = True
        warnings: list[str] = []
        data_quality_penalty = 0.0

        if config.drop_all_null and non_null_count == 0:
            keep = False
            warnings.append("all_null")
            data_quality_penalty = 1.0
        elif non_null_count < config.min_total_non_null_count:
            one_group_enough = max(bad_non_null, good_non_null) >= config.min_group_non_null_count
            if one_group_enough:
                warnings.append("low_total_count_but_one_group_signal")
                data_quality_penalty += 0.25
            else:
                keep = False
                warnings.append("too_few_non_null")
                data_quality_penalty += 0.75

        if config.drop_constant and unique_non_null < config.min_unique_non_null:
            if asymmetric["presence_type"] in {"bad_only", "good_only"}:
                warnings.append("constant_value_presence_signal")
                data_quality_penalty += 0.15
            else:
                keep = False
                warnings.append("constant")
                data_quality_penalty += 0.70

        if top_ratio >= config.near_constant_top_ratio and unique_non_null >= config.min_unique_non_null:
            warnings.append("near_constant")
            data_quality_penalty += 0.20

        if missing_rate >= config.warn_missing_rate and keep:
            warnings.append("high_missing_rate")
            data_quality_penalty += min((missing_rate - config.warn_missing_rate) / 0.05, 1.0) * 0.20

        records.append(
            {
                "feature_name": feature,
                "dtype": str(series.dtype),
                "non_null_count": non_null_count,
                "missing_rate": missing_rate,
                "unique_non_null": unique_non_null,
                "top_value_ratio": top_ratio,
                "quality_keep": bool(keep),
                "data_quality_penalty_score": float(np.clip(data_quality_penalty, 0.0, 1.0)),
                "quality_warning": compact_warning(warnings),
            }
        )

    return pd.DataFrame.from_records(records).set_index("feature_name")


def _top_value_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    counts = non_null.value_counts(dropna=True)
    if counts.empty:
        return 0.0
    return float(counts.iloc[0] / len(non_null))
