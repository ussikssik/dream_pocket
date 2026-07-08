from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_boosting.data_loader import standardize_six_file_inputs


class DataLoaderTests(unittest.TestCase):
    def test_standardize_six_file_inputs_with_custom_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            y_path = root / "yield_file.csv"
            candidate_path = root / "candidate_file.csv"
            base_path = root / "base_file.csv"
            group_path = root / "defect_a_group.csv"
            output_dir = root / "standardized"

            pd.DataFrame(
                {
                    "lot_id": ["L1", "L1", "L2", "L2", "L3", "L3"],
                    "wafer_no": [1, 2, 1, 2, 1, 2],
                    "target_y": [80.1, 80.3, 78.4, 79.5, 82.0, 81.1],
                }
            ).to_csv(y_path, index=False)
            pd.DataFrame(
                {
                    "lot_key": ["L1", "L1", "L2", "L2", "L3", "L3"],
                    "wf_key": [1, 2, 1, 2, 1, 2],
                    "candidate_signal": [1, 2, 3, 4, 5, 6],
                    "base_pressure": [10, 11, 12, 13, 14, 15],
                }
            ).to_csv(candidate_path, index=False)
            pd.DataFrame(
                {
                    "lot_name": ["L1", "L1", "L2", "L2", "L3", "L3"],
                    "wf_name": [1, 2, 1, 2, 1, 2],
                    "base_temp": [30, 31, 32, 33, 34, 35],
                    "base_pressure": [100, 101, 102, 103, 104, 105],
                }
            ).to_csv(base_path, index=False)
            pd.DataFrame(
                {
                    "lot_group": ["L1", "L1", "L2", "L2"],
                    "wf_group": [1, 2, 1, 2],
                    "judgement": ["bad", "good", "bad", "good"],
                }
            ).to_csv(group_path, index=False)

            result = standardize_six_file_inputs(
                y_path=y_path,
                candidate_path=candidate_path,
                base_feature_path=base_path,
                defect_groups=[
                    {
                        "defect_id": "defect_a",
                        "group_path": group_path,
                        "lot_col": "lot_group",
                        "wf_col": "wf_group",
                        "label_col": "judgement",
                    }
                ],
                output_dir=output_dir,
                y_lot_col="lot_id",
                y_wf_col="wafer_no",
                y_target_col="target_y",
                candidate_lot_col="lot_key",
                candidate_wf_col="wf_key",
                base_lot_col="lot_name",
                base_wf_col="wf_name",
            )

            base_df = pd.read_csv(result["base_dataset"])
            candidate_df = pd.read_csv(result["candidate_features"])
            base_cols = Path(result["base_feature_cols"]).read_text(encoding="utf-8").splitlines()
            bad_df = pd.read_csv(result["defects"][0]["bad_group_path"])
            good_df = pd.read_csv(result["defects"][0]["good_group_path"])

            self.assertEqual({"sample_id", "yield", "split", "base_temp", "base_pressure"}, set(base_df.columns))
            self.assertIn("candidate_signal", candidate_df.columns)
            self.assertIn("candidate__base_pressure", candidate_df.columns)
            self.assertEqual(["base_temp", "base_pressure"], base_cols)
            self.assertEqual({"L1_1", "L2_1"}, set(bad_df["sample_id"].astype(str)))
            self.assertEqual({"L1_2", "L2_2"}, set(good_df["sample_id"].astype(str)))
            self.assertEqual({"train", "valid", "test"}, set(base_df["split"].astype(str)))


if __name__ == "__main__":
    unittest.main()
