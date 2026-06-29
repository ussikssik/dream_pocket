from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RedundancyConfig
from .utils import numeric_or_none


def assign_redundancy_groups(
    X: pd.DataFrame,
    ranked_df: pd.DataFrame,
    config: RedundancyConfig,
) -> pd.DataFrame:
    """Assign correlation clusters so near-duplicate features do not crowd top K."""

    features = ranked_df.index.tolist()
    result = pd.DataFrame(index=ranked_df.index)
    result["redundancy_group"] = ""
    result["redundancy_penalty_score"] = 0.0

    if not config.enabled or not features:
        return result

    candidates = features[: config.max_features]
    numeric_data = {}
    for feature in candidates:
        numeric = numeric_or_none(X[feature])
        if numeric is not None and numeric.notna().sum() >= 3:
            numeric_data[feature] = numeric

    if len(numeric_data) < 2:
        return result

    matrix = pd.DataFrame(numeric_data)
    corr = matrix.corr(method="spearman").abs().fillna(0.0)
    group_id = 0
    assigned: dict[str, str] = {}

    for feature in candidates:
        if feature in assigned:
            continue
        group_id += 1
        cluster_name = f"corr_cluster_{group_id:04d}"
        assigned[feature] = cluster_name
        result.loc[feature, "redundancy_group"] = cluster_name

        if feature not in corr.columns:
            continue
        neighbors = corr.index[(corr[feature] >= config.correlation_threshold) & (corr.index != feature)]
        for neighbor in neighbors:
            if neighbor in assigned:
                continue
            assigned[neighbor] = cluster_name
            result.loc[neighbor, "redundancy_group"] = cluster_name
            result.loc[neighbor, "redundancy_penalty_score"] = float(
                np.clip(corr.loc[neighbor, feature], 0.0, 1.0)
            )

    return result
