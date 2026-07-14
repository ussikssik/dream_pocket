from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from feature_boosting.modeling import fit_regressor, predict_regressor
from feature_boosting.overfit import recommend_overfit_safe_settings
from feature_boosting.residual_boosting import ResidualFeatureBooster, ResidualFeatureBoosterConfig


class ResidualBoostingTests(unittest.TestCase):
    def test_auto_parameter_adjustment_remains_without_candidate_rejection(self) -> None:
        recommendation = recommend_overfit_safe_settings(
            {
                "backend": "numpy",
                "iterations": 1000,
                "depth": 8,
                "learning_rate": 0.2,
                "l2": 1e-6,
            },
            n_train_rows=500,
            n_candidate_features=1000,
            metric_name="mae",
        )

        params = recommendation["residual_model_params"]
        self.assertEqual(params["iterations"], 120)
        self.assertEqual(params["depth"], 2)
        self.assertEqual(params["learning_rate"], 0.03)
        self.assertGreaterEqual(params["l2"], 20.0)
        self.assertFalse(recommendation["guard"]["overfit_guard_enabled"])
        self.assertIn(
            {"setting": "candidate_overfit_rejection", "value": "disabled"},
            recommendation["summary_rows"],
        )

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
            base_feature_cols=["base_feature"],
            base_model_params={"backend": "numpy"},
            defect_id="defect_1",
            bad_sample_ids=bad_ids,
            good_sample_ids=good_ids,
        )
        self.assertFalse(result.selected_features.empty)
        self.assertEqual(result.selected_features["feature_name"].iloc[0], "hidden")
        self.assertGreater(float(result.selected_features["valid_bad_rmse_reduction"].iloc[0]), 0)
        self.assertIn("valid_bad_mae_reduction", result.selected_features.columns)
        ranking = result.rankings[0].set_index("feature_name")
        self.assertIn("valid_bad_rmse_after_over_baseline", ranking.columns)
        self.assertLess(float(ranking.loc["hidden", "valid_bad_rmse_after_over_baseline"]), 1.0)
        before = float(ranking.loc["hidden", "valid_bad_rmse_before"])
        after = float(ranking.loc["hidden", "valid_bad_rmse_after"])
        relative_improvement = float(ranking.loc["hidden", "valid_bad_rmse_reduction_over_before"])
        self.assertAlmostEqual(relative_improvement, (before - after) / before)
        self.assertEqual(ranking.loc["hidden", "ranking_metric"], "valid_bad_rmse_reduction_over_before")
        expected_model = fit_regressor(
            train_df,
            train_df["yield"].to_numpy(dtype=float),
            valid_df,
            valid_df["yield"].to_numpy(dtype=float),
            ["base_feature", "hidden"],
            {"backend": "numpy"},
        )
        expected_valid = predict_regressor(expected_model, valid_df, ["base_feature", "hidden"])
        np.testing.assert_allclose(result.final_predictions["valid"], expected_valid)
        self.assertEqual(result.rankings[0]["iteration_model"].iloc[0], "cumulative_base_refit")

    def test_threshold_mode_selects_by_relative_improvement(self) -> None:
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
                selection_metric="bad_rmse_reduction_over_before",
                selection_threshold=0.5,
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
            base_feature_cols=["base_feature"],
            base_model_params={"backend": "numpy"},
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
            base_feature_cols=["base_feature"],
            base_model_params={"backend": "numpy"},
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
            base_feature_cols=["base_feature"],
            base_model_params={"backend": "numpy"},
            defect_id="defect_1",
            bad_sample_ids=bad_ids,
            good_sample_ids=good_ids,
            always_rank_cols=["noise_feature"],
        )

        ranking = result.rankings[0].set_index("feature_name")
        self.assertIn("noise_feature", ranking.index)
        self.assertFalse(bool(ranking.loc["noise_feature", "selected"]))
        self.assertEqual(result.selected_features["feature_name"].tolist(), ["hidden"])

    def test_overfit_guard_diagnostics_do_not_reject_candidate(self) -> None:
        df = _synthetic_frame()
        train_df = df[df["split"] == "train"].copy()
        valid_df = df[df["split"] == "valid"].copy()
        test_df = df[df["split"] == "test"].copy()
        bad_ids = set(df["sample_id"])

        booster = ResidualFeatureBooster(
            ResidualFeatureBoosterConfig(
                residual_model_params={"backend": "numpy", "l2": 1e-6},
                n_rounds=1,
                overfit_guard_enabled=True,
                overfit_guard_max_valid_after_over_baseline=0.0,
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
            base_feature_cols=["base_feature"],
            base_model_params={"backend": "numpy"},
            defect_id="defect_1",
            bad_sample_ids=bad_ids,
            good_sample_ids=set(),
        )

        self.assertEqual(result.selected_features["feature_name"].tolist(), ["hidden"])
        ranking = result.rankings[0].set_index("feature_name")
        self.assertIn("train_bad_rmse_after_over_baseline", ranking.columns)
        self.assertFalse(bool(ranking.loc["hidden", "overfit_guard_pass"]))
        self.assertIn("valid_bad_rmse_after_over_baseline", str(ranking.loc["hidden", "overfit_guard_reason"]))

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
            base_feature_cols=["base_feature"],
            base_model_params={"backend": "numpy"},
            defect_id="defect_1",
            bad_sample_ids=bad_ids,
            good_sample_ids=set(),
        )
        self.assertTrue(result.selected_features.empty)

    def test_sequential_defects_carry_predictions_and_exclude_selected_features(self) -> None:
        df = _sequential_frame()
        train_df = df[df["split"] == "train"].copy()
        valid_df = df[df["split"] == "valid"].copy()
        test_df = df[df["split"] == "test"].copy()
        all_ids = set(df["sample_id"])
        booster = ResidualFeatureBooster(
            ResidualFeatureBoosterConfig(
                residual_model_params={"backend": "numpy"},
                n_rounds=1,
                overfit_guard_enabled=False,
                show_progress=False,
            )
        )

        first = booster.run_for_defect(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            candidate_cols=["hidden_1"],
            target_col="yield",
            id_col="sample_id",
            baseline_pred_col="baseline_pred",
            base_feature_cols=["base_feature"],
            base_model_params={"backend": "numpy"},
            defect_id="defect_1",
            bad_sample_ids=all_ids,
            good_sample_ids=set(),
            global_iter_start=0,
        )
        selected_first = set(first.selected_features["feature_name"].astype(str))
        second = booster.run_for_defect(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            candidate_cols=["hidden_1", "hidden_2"],
            target_col="yield",
            id_col="sample_id",
            baseline_pred_col="baseline_pred",
            base_feature_cols=["base_feature"],
            base_model_params={"backend": "numpy"},
            defect_id="defect_2",
            bad_sample_ids=all_ids,
            good_sample_ids=set(),
            initial_predictions=first.final_predictions,
            previously_selected_features=selected_first,
            global_iter_start=len(first.rankings),
        )

        self.assertEqual(first.iteration_summary["global_iter"].tolist(), [1])
        self.assertEqual(second.iteration_summary["global_iter"].tolist(), [2])
        self.assertNotIn("hidden_1", second.rankings[0]["feature_name"].astype(str).tolist())
        self.assertEqual(second.selected_features["feature_name"].tolist(), ["hidden_2"])
        first_valid_after = float(first.iteration_summary.loc[0, "valid_global_rmse_after"])
        second_valid_before = float(second.iteration_summary.loc[0, "valid_global_rmse_before"])
        self.assertAlmostEqual(first_valid_after, second_valid_before)
        self.assertEqual(int(second.iteration_summary.loc[0, "n_cumulative_selected"]), 2)
        self.assertEqual(int(second.iteration_summary.loc[0, "n_effective_features"]), 3)
        self.assertEqual(sorted(second.test_predictions["global_iter"].unique().tolist()), [2])
        self.assertIn("valid_bad_rmse_reduction_over_before", second.rankings[0].columns)


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
            "base_feature": base,
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


def _sequential_frame() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    n = 120
    hidden_1 = rng.normal(0, 1, n)
    hidden_2 = rng.normal(0, 1, n)
    baseline = np.linspace(40, 50, n)
    return pd.DataFrame(
        {
            "sample_id": [f"WF_SEQ_{idx:04d}" for idx in range(n)],
            "split": np.array(["train"] * 70 + ["valid"] * 25 + ["test"] * 25),
            "yield": baseline + 4.0 * hidden_1 + 2.0 * hidden_2,
            "baseline_pred": baseline,
            "base_feature": baseline,
            "hidden_1": hidden_1,
            "hidden_2": hidden_2,
        }
    )


if __name__ == "__main__":
    unittest.main()
