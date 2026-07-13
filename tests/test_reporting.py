from __future__ import annotations

import unittest

import pandas as pd

from feature_boosting.reporting import final_metric_summary, iteration_residual_summary, round_residual_summary


class ReportingTests(unittest.TestCase):
    def test_round_residual_summary_keeps_one_final_point_per_round(self) -> None:
        baseline = pd.DataFrame(
            [
                {
                    "defect_id": "defect_1",
                    "group": "bad",
                    "split": "valid",
                    "mean_abs_residual": 10.0,
                }
            ]
        )
        curve = pd.DataFrame(
            [
                {
                    "defect_id": "defect_1",
                    "round": 1,
                    "selected_feature": "feature_a",
                    "valid_bad_mae": 8.0,
                    "is_answer_feature": False,
                },
                {
                    "defect_id": "defect_1",
                    "round": 1,
                    "selected_feature": "feature_b",
                    "valid_bad_mae": 6.0,
                    "is_answer_feature": True,
                },
                {
                    "defect_id": "defect_1",
                    "round": 2,
                    "selected_feature": "feature_c",
                    "valid_bad_mae": 5.0,
                    "is_answer_feature": False,
                },
            ]
        )

        summary = round_residual_summary(curve, baseline, group="bad")
        valid = summary[summary["split"].astype(str) == "valid"].reset_index(drop=True)

        self.assertEqual(valid["round"].tolist(), [0, 1, 2])
        self.assertEqual(valid["mean_abs_residual"].tolist(), [10.0, 6.0, 5.0])
        self.assertEqual(valid.loc[1, "round_selected_features"], "feature_a, feature_b")
        self.assertEqual(int(valid.loc[1, "n_selected_features_in_round"]), 2)
        self.assertTrue(bool(valid.loc[1, "round_contains_answer_feature"]))

    def test_final_metric_summary_calculates_reductions(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "model_name": "baseline",
                    "defect_id": "defect_1",
                    "group": "bad",
                    "split": "valid",
                    "n_samples": 10,
                    "rmse": 5.0,
                    "mae": 4.0,
                    "r2": 0.10,
                },
                {
                    "model_name": "final",
                    "defect_id": "defect_1",
                    "group": "bad",
                    "split": "valid",
                    "n_samples": 10,
                    "rmse": 4.0,
                    "mae": 3.0,
                    "r2": 0.25,
                },
            ]
        )

        summary = final_metric_summary(metrics)

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["label"], "defect_1 / bad")
        self.assertAlmostEqual(float(row["rmse_reduction"]), 1.0)
        self.assertAlmostEqual(float(row["rmse_reduction_pct"]), 20.0)
        self.assertAlmostEqual(float(row["mae_reduction"]), 1.0)
        self.assertAlmostEqual(float(row["mae_reduction_pct"]), 25.0)
        self.assertAlmostEqual(float(row["r2_delta"]), 0.15)

    def test_iteration_residual_summary_uses_inherited_defect_start(self) -> None:
        iterations = pd.DataFrame(
            [
                {
                    "global_iter": 1,
                    "defect_id": "defect_1",
                    "round": 1,
                    "selected_features": "feature_a",
                    "valid_bad_mae_before": 10.0,
                    "valid_bad_mae_after": 8.0,
                },
                {
                    "global_iter": 2,
                    "defect_id": "defect_2",
                    "round": 1,
                    "selected_features": "feature_b",
                    "valid_bad_mae_before": 7.0,
                    "valid_bad_mae_after": 5.0,
                },
            ]
        )

        summary = iteration_residual_summary(iterations, group="bad")
        defect_2 = summary[
            (summary["defect_id"].astype(str) == "defect_2")
            & (summary["split"].astype(str) == "valid")
        ].reset_index(drop=True)

        self.assertEqual(defect_2["round"].tolist(), [0, 1])
        self.assertEqual(defect_2["global_iter"].tolist(), [1, 2])
        self.assertEqual(defect_2["mean_abs_residual"].tolist(), [7.0, 5.0])


if __name__ == "__main__":
    unittest.main()
