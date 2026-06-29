from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_booster import BoosterConfig, CatBoostProbeConfig, DefectAFeatureEvidenceBooster


def make_synthetic_order(order_id: int) -> pd.DataFrame:
    rng = np.random.default_rng(2026 + order_id)
    n = 600
    lot = rng.choice([f"L{i:02d}" for i in range(12)], size=n)
    target = rng.binomial(1, 0.25, size=n)

    df = pd.DataFrame(
        {
            "sample_id": [f"{order_id}_{i}" for i in range(n)],
            "lot_id": lot,
            "target_bad_a": target,
            "noise_001": rng.normal(0, 1, size=n),
            "noise_002": rng.normal(0, 1, size=n),
            "bad_shift_sensor": rng.normal(target * 1.2, 1, size=n),
            "good_shift_sensor": rng.normal((1 - target) * 1.1, 1, size=n),
            "route_like_sparse_bad": np.where(target == 1, rng.normal(5, 0.2, size=n), np.nan),
            "near_duplicate_of_bad_shift": np.nan,
        }
    )
    df["near_duplicate_of_bad_shift"] = df["bad_shift_sensor"] + rng.normal(0, 0.01, size=n)
    return df


def load_order(order_id: int) -> pd.DataFrame:
    return make_synthetic_order(order_id)


if __name__ == "__main__":
    config = BoosterConfig(
        label_col="target_bad_a",
        group_cols=("lot_id",),
        sample_id_cols=("sample_id",),
        catboost=CatBoostProbeConfig(enabled=False),
    )
    booster = DefectAFeatureEvidenceBooster(load_order, config)
    result = booster.run(order_list=[1, 2], top_k_per_order=5)
    print(result[["order_id", "final_rank", "feature_name", "final_score", "evidence_reason"]])
