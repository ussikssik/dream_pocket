from __future__ import annotations

import unittest

import pandas as pd

from feature_boosting.metrics import mae, residual_reduction_metrics, rmse
from feature_boosting.reporting import round_residual_summary


class MetricTests(unittest.TestCase):
    def test_rmse_and_mae(self) -> None:
        self.assertAlmostEqual(rmse([1, 2, 3], [1, 2, 5]), (4 / 3) ** 0.5)
        self.assertAlmostEqual(mae([1, 2, 3], [1, 2, 5]), 2 / 3)

    def test_residual_reduction_sign(self) -> None:
        metrics = residual_reduction_metrics([1, 2, 3], [1, 1, 1], [1, 2, 3])
        self.assertGreater(metrics["rmse_reduction"], 0)
        self.assertGreater(metrics["mae_reduction"], 0)

    def test_round_residual_summary_includes_baseline_round(self) -> None:
        baseline = pd.DataFrame(
            [
                {"defect_id": "defect_1", "group": "bad", "split": "valid", "mean_abs_residual": 3.0},
                {"defect_id": "defect_1", "group": "bad", "split": "test", "mean_abs_residual": 4.0},
            ]
        )
        curve = pd.DataFrame(
            [
                {
                    "defect_id": "defect_1",
                    "round": 1,
                    "selected_feature": "hidden",
                    "valid_bad_mae": 2.0,
                    "test_bad_mae": 3.5,
                }
            ]
        )
        summary = round_residual_summary(curve, baseline, group="bad")
        self.assertEqual(summary["round"].tolist(), [0, 1, 0, 1])
        self.assertEqual(summary["mean_abs_residual"].tolist(), [4.0, 3.5, 3.0, 2.0])


if __name__ == "__main__":
    unittest.main()
