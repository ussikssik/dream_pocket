from __future__ import annotations

import unittest

from feature_boosting.metrics import mae, residual_reduction_metrics, rmse


class MetricTests(unittest.TestCase):
    def test_rmse_and_mae(self) -> None:
        self.assertAlmostEqual(rmse([1, 2, 3], [1, 2, 5]), (4 / 3) ** 0.5)
        self.assertAlmostEqual(mae([1, 2, 3], [1, 2, 5]), 2 / 3)

    def test_residual_reduction_sign(self) -> None:
        metrics = residual_reduction_metrics([1, 2, 3], [1, 1, 1], [1, 2, 3])
        self.assertGreater(metrics["rmse_reduction"], 0)
        self.assertGreater(metrics["mae_reduction"], 0)


if __name__ == "__main__":
    unittest.main()
