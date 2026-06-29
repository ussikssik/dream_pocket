from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AsymmetricSignalConfig


def detect_asymmetric_signals(
    X: pd.DataFrame,
    y: pd.Series,
    config: AsymmetricSignalConfig,
) -> pd.DataFrame:
    """Profile whether a feature exists symmetrically across Good and Bad groups."""

    y = y.astype(int)
    bad_mask = y == 1
    good_mask = y == 0
    bad_total = int(bad_mask.sum())
    good_total = int(good_mask.sum())

    records: list[dict[str, object]] = []
    for feature in X.columns:
        non_null = X[feature].notna()
        bad_non_null = int((non_null & bad_mask).sum())
        good_non_null = int((non_null & good_mask).sum())
        bad_coverage = bad_non_null / bad_total if bad_total else 0.0
        good_coverage = good_non_null / good_total if good_total else 0.0
        delta = bad_coverage - good_coverage

        if bad_non_null > 0 and good_non_null == 0:
            presence_type = "bad_only"
        elif good_non_null > 0 and bad_non_null == 0:
            presence_type = "good_only"
        elif _is_enriched(bad_coverage, good_coverage, delta, config):
            presence_type = "bad_enriched_sparse"
        elif _is_enriched(good_coverage, bad_coverage, -delta, config):
            presence_type = "good_enriched_sparse"
        else:
            presence_type = "both_groups"

        asymmetric_presence_score = min(abs(delta) / max(config.min_presence_delta, 1e-12), 1.0)
        if presence_type in {"bad_only", "good_only"}:
            asymmetric_presence_score = max(asymmetric_presence_score, 0.85)

        records.append(
            {
                "feature_name": feature,
                "bad_non_null_count": bad_non_null,
                "good_non_null_count": good_non_null,
                "bad_coverage": bad_coverage,
                "good_coverage": good_coverage,
                "presence_delta_bad_minus_good": delta,
                "presence_type": presence_type,
                "asymmetric_presence_score": float(np.clip(asymmetric_presence_score, 0.0, 1.0)),
            }
        )

    return pd.DataFrame.from_records(records).set_index("feature_name")


def _is_enriched(
    numerator_rate: float,
    denominator_rate: float,
    delta: float,
    config: AsymmetricSignalConfig,
) -> bool:
    if delta < config.min_presence_delta:
        return False
    if denominator_rate <= config.sparse_presence_rate:
        return numerator_rate > denominator_rate
    ratio = numerator_rate / max(denominator_rate, 1e-12)
    return ratio >= config.min_presence_ratio
