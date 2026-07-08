from __future__ import annotations

import unittest

import pandas as pd

from feature_boosting.reporting import final_metric_summary


class ReportingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
