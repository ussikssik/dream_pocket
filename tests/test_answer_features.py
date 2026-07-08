from __future__ import annotations

import unittest

import pandas as pd

from feature_boosting.answer_features import add_answer_feature_flags, answer_feature_mask


class AnswerFeatureTests(unittest.TestCase):
    def test_exact_and_contains_all_rules(self) -> None:
        names = pd.Series(["exact_feature", "abc_sensor_step2_mean", "abc_sensor_step1_mean", "other"])
        rules = [
            "exact_feature",
            {"contains_all": ["abc", "step2"]},
        ]

        mask = answer_feature_mask(names, rules)

        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_string_rule_is_case_insensitive(self) -> None:
        names = pd.Series(["Hidden_Defect_2", "hidden_defect_1", "noise"])

        mask = answer_feature_mask(names, ["hidden_defect_2"])

        self.assertEqual(mask.tolist(), [True, False, False])

    def test_add_answer_feature_flags_adds_reason(self) -> None:
        frame = pd.DataFrame({"feature_name": ["tool_root_signal", "noise"]})
        result = add_answer_feature_flags(
            frame,
            feature_col="feature_name",
            rules=[{"contains_any": ["root", "defect"]}],
        )

        self.assertEqual(result["is_answer_feature"].tolist(), [True, False])
        self.assertTrue(str(result.loc[0, "answer_match_rule"]).startswith("contains_any:"))
        self.assertEqual(result.loc[1, "answer_match_rule"], "")


if __name__ == "__main__":
    unittest.main()
