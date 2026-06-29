from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_booster import BoosterConfig, CatBoostProbeConfig, DefectAFeatureEvidenceBooster


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOY_DIR = PROJECT_ROOT / "data" / "toy_semiconductor"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "toy_semiconductor_booster"


def load_order(order_id: int) -> pd.DataFrame:
    return pd.read_csv(TOY_DIR / f"order_{order_id:03d}.csv")


def load_feature_metadata() -> dict[str, dict[str, object]]:
    metadata_path = TOY_DIR / "feature_metadata.csv"
    if not metadata_path.exists():
        return {}
    metadata = pd.read_csv(metadata_path)
    return metadata.set_index("feature_name").to_dict(orient="index")


if __name__ == "__main__":
    config = BoosterConfig(
        label_col="target_bad_a",
        positive_label=1,
        group_cols=("lot_id", "product_id"),
        sample_id_cols=("sample_id", "wafer_id"),
        exclude_cols=(
            "order_id",
            "eds_bin_no_wf_mean",
            "eds_bin_a_wf_mean",
            "eds_bin_b_wf_mean",
            "eds_bin_c_wf_mean",
            "eds_yield_wf_mean",
            "sim_true_defect_a_flag",
            "sim_true_defect_b_flag",
            "sim_true_defect_c_flag",
            "sim_true_defect_d_flag",
            "sim_dominant_defect",
        ),
        feature_metadata=load_feature_metadata(),
        output_dir=OUTPUT_DIR,
        catboost=CatBoostProbeConfig(enabled=False),
    )

    booster = DefectAFeatureEvidenceBooster(load_order, config)
    result = booster.run(order_list=[1, 2, 3, 4, 5], top_k_per_order=10)

    display_cols = [
        "order_id",
        "final_rank",
        "feature_name",
        "final_score",
        "presence_type",
        "direction",
        "evidence_reason",
    ]
    print(result[display_cols].to_string(index=False))
    print(f"\nSaved reports to {OUTPUT_DIR}")
