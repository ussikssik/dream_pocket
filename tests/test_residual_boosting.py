from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from feature_boosting.residual_boosting import ResidualFeatureBooster, ResidualFeatureBoosterConfig


class ResidualBoostingTests(unittest.TestCase):
    def test_known_residual_feature_is_selected(self) -> None:
        df = _synthetic_frame()
        train_df = df[df["split"] == "train"].copy()
        valid_df = df[df["split"] == "valid"].copy()
        test_df = df[df["split"] == "test"].copy()
        bad_ids = set(df.loc[df["hidden"] > df["hidden"].median(), "sample_id"])
        good_ids = set(df.loc[df["hidden"] <= df["hidden"].median(), "sample_id"])

        booster = ResidualFeatureBooster(
            ResidualFeatureBoosterConfig(
                residual_model_params={"backend": "numpy"},
                n_rounds=1,
                min_improvement=0.0,
            )
        )
        result = booster.run_for_defect(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            candidate_cols=["hidden", "noise_feature"],
            target_col="yield",
            id_col="sample_id",
            baseline_pred_col="baseline_pred",
            defect_id="defect_1",
            bad_sample_ids=bad_ids,
            good_sample_ids=good_ids,
        )
        self.assertFalse(result.selected_features.empty)
        self.assertEqual(result.selected_features["feature_name"].iloc[0], "hidden")
        self.assertGreater(float(result.selected_features["valid_bad_rmse_reduction"].iloc[0]), 0)

    def test_zero_improvement_feature_is_not_selected(self) -> None:
        df = _synthetic_frame()
        df["yield"] = df["baseline_pred"]
        train_df = df[df["split"] == "train"].copy()
        valid_df = df[df["split"] == "valid"].copy()
        test_df = df[df["split"] == "test"].copy()
        bad_ids = set(valid_df["sample_id"]) | set(test_df["sample_id"])

        booster = ResidualFeatureBooster(
            ResidualFeatureBoosterConfig(
                residual_model_params={"backend": "numpy"},
                n_rounds=1,
                min_improvement=0.0,
            )
        )
        result = booster.run_for_defect(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            candidate_cols=["hidden"],
            target_col="yield",
            id_col="sample_id",
            baseline_pred_col="baseline_pred",
            defect_id="defect_1",
            bad_sample_ids=bad_ids,
            good_sample_ids=set(),
        )
        self.assertTrue(result.selected_features.empty)


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 90
    sample_id = [f"WF_{idx:04d}" for idx in range(n)]
    split = np.array(["train"] * 50 + ["valid"] * 20 + ["test"] * 20)
    base = np.linspace(50, 60, n)
    hidden = rng.normal(0, 1, n)
    noise_feature = rng.normal(0, 1, n)
    y = base + 4.0 * hidden + rng.normal(0, 0.02, n)
    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "split": split,
            "yield": y,
            "baseline_pred": base,
            "hidden": hidden,
            "noise_feature": noise_feature,
        }
    )


if __name__ == "__main__":
    unittest.main()
