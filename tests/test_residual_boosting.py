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
                show_progress=False,
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
        ranking = result.rankings[0].set_index("feature_name")
        self.assertIn("valid_bad_rmse_after_over_baseline", ranking.columns)
        self.assertLess(float(ranking.loc["hidden", "valid_bad_rmse_after_over_baseline"]), 1.0)

    def test_threshold_mode_selects_by_residual_ratio(self) -> None:
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
                selection_mode="threshold",
                selection_metric="bad_rmse_after_over_baseline",
                selection_threshold=0.25,
                max_select_per_round=None,
                show_progress=False,
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
        selected = set(result.selected_features["feature_name"].astype(str))
        self.assertEqual(selected, {"hidden"})

    def test_always_rank_cols_drop_feature_after_selection(self) -> None:
        df = _synthetic_frame()
        train_df = df[df["split"] == "train"].copy()
        valid_df = df[df["split"] == "valid"].copy()
        test_df = df[df["split"] == "test"].copy()
        bad_ids = set(df.loc[df["hidden"] > df["hidden"].median(), "sample_id"])
        good_ids = set(df.loc[df["hidden"] <= df["hidden"].median(), "sample_id"])

        booster = ResidualFeatureBooster(
            ResidualFeatureBoosterConfig(
                residual_model_params={"backend": "numpy"},
                n_rounds=2,
                min_improvement=0.0,
                show_progress=False,
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
            always_rank_cols=["hidden"],
        )

        self.assertGreaterEqual(len(result.rankings), 2)
        round_1 = result.rankings[0].set_index("feature_name")
        round_2 = result.rankings[1].set_index("feature_name")
        self.assertTrue(bool(round_1.loc["hidden", "selected"]))
        self.assertNotIn("hidden", round_2.index)

    def test_always_rank_cols_do_not_force_selection(self) -> None:
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
                select_per_round=1,
                min_improvement=0.0,
                show_progress=False,
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
            always_rank_cols=["noise_feature"],
        )

        ranking = result.rankings[0].set_index("feature_name")
        self.assertIn("noise_feature", ranking.index)
        self.assertFalse(bool(ranking.loc["noise_feature", "selected"]))
        self.assertEqual(result.selected_features["feature_name"].tolist(), ["hidden"])

    def test_overfit_guard_rejects_train_only_signal(self) -> None:
        df = _overfit_frame()
        train_df = df[df["split"] == "train"].copy()
        valid_df = df[df["split"] == "valid"].copy()
        test_df = df[df["split"] == "test"].copy()
        bad_ids = set(df["sample_id"])

        booster = ResidualFeatureBooster(
            ResidualFeatureBoosterConfig(
                residual_model_params={"backend": "numpy", "l2": 1e-6},
                n_rounds=1,
                overfit_guard_enabled=True,
                overfit_guard_max_valid_after_over_baseline=1.0,
                overfit_guard_max_valid_train_gap=0.25,
                show_progress=False,
            )
        )
        result = booster.run_for_defect(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            candidate_cols=["train_only_signal"],
            target_col="yield",
            id_col="sample_id",
            baseline_pred_col="baseline_pred",
            defect_id="defect_1",
            bad_sample_ids=bad_ids,
            good_sample_ids=set(),
        )

        self.assertTrue(result.selected_features.empty)
        ranking = result.rankings[0].set_index("feature_name")
        self.assertIn("train_bad_rmse_after_over_baseline", ranking.columns)
        self.assertFalse(bool(ranking.loc["train_only_signal", "overfit_guard_pass"]))
        self.assertIn("valid_bad_rmse_after_over_baseline", str(ranking.loc["train_only_signal", "overfit_guard_reason"]))

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
                show_progress=False,
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


def _overfit_frame() -> pd.DataFrame:
    n_train = 40
    n_valid = 20
    n_test = 20
    n = n_train + n_valid + n_test
    sample_id = [f"WF_OVERFIT_{idx:04d}" for idx in range(n)]
    split = np.array(["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test)
    signal = np.linspace(-2.0, 2.0, n)
    y = signal.copy()
    y[n_train:] = -signal[n_train:]
    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "split": split,
            "yield": y,
            "baseline_pred": np.zeros(n),
            "train_only_signal": signal,
        }
    )


if __name__ == "__main__":
    unittest.main()
