from __future__ import annotations

import unittest

import pandas as pd

from feature_boosting.config import FeatureFilterConfig
from feature_boosting.validation import profile_candidate_features, validate_defect_groups


class ValidationTests(unittest.TestCase):
    def test_group_overlap_raises(self) -> None:
        df = pd.DataFrame({"sample_id": ["a", "b", "c"], "split": ["train", "valid", "test"]})
        with self.assertRaises(ValueError):
            validate_defect_groups("d1", {"a", "b"}, {"b", "c"}, df, id_col="sample_id", split_col="split")

    def test_missing_and_constant_filter(self) -> None:
        df = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d", "e", "f"],
                "split": ["train", "train", "valid", "valid", "test", "test"],
                "signal_feature": [1, 2, 3, 4, 5, 6],
                "constant": [1, 1, 1, 1, 1, 1],
                "mostly_missing": [1, None, None, None, None, None],
            }
        )
        summary = profile_candidate_features(
            df,
            ["signal_feature", "constant", "mostly_missing"],
            id_col="sample_id",
            split_col="split",
            bad_ids={"c", "e"},
            good_ids={"d", "f"},
            config=FeatureFilterConfig(max_missing_rate=0.5, min_unique_values=2, min_bad_coverage=0.5, min_good_coverage=0.5),
        ).set_index("feature_name")
        self.assertTrue(bool(summary.loc["signal_feature", "is_pass"]))
        self.assertIn("constant_feature", summary.loc["constant", "fail_reason"])
        self.assertIn("high_missing_rate", summary.loc["mostly_missing", "fail_reason"])

    def test_defect_named_candidate_is_not_automatically_leakage(self) -> None:
        df = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d", "e", "f"],
                "split": ["train", "train", "valid", "valid", "test", "test"],
                "hidden_defect_1": [1, 2, 3, 4, 5, 6],
                "yield_target_like": [1, 2, 3, 4, 5, 6],
            }
        )
        summary = profile_candidate_features(
            df,
            ["hidden_defect_1", "yield_target_like"],
            id_col="sample_id",
            split_col="split",
            bad_ids={"c", "e"},
            good_ids={"d", "f"},
            config=FeatureFilterConfig(max_missing_rate=0.5, min_unique_values=2, min_bad_coverage=0.5, min_good_coverage=0.5),
        ).set_index("feature_name")

        self.assertNotIn("leakage_like_feature", str(summary.loc["hidden_defect_1", "fail_reason"]))
        self.assertTrue(bool(summary.loc["hidden_defect_1", "is_pass"]))
        self.assertIn("leakage_like_feature", str(summary.loc["yield_target_like", "fail_reason"]))


if __name__ == "__main__":
    unittest.main()
