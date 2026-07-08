from __future__ import annotations

import re
from typing import Any

import pandas as pd


AnswerRule = str | dict[str, Any]


def answer_feature_mask(feature_names: pd.Series, rules: list[AnswerRule] | set[str] | tuple[AnswerRule, ...] | None) -> pd.Series:
    """Return True for feature names matching answer-feature rules.

    Supported rules:
    - "exact_feature_name"
    - {"exact": "feature_name"} or {"exact": ["a", "b"]}
    - {"contains": "abc"} or {"contains": ["abc", "step2"]}  # all tokens
    - {"contains_all": ["abc", "step2"]}
    - {"contains_any": ["abc", "step2"]}
    - {"regex": r"abc.*step2"} or {"regex": [r"a", r"b"]}

    Matching is case-insensitive by default. Add {"case_sensitive": True}
    inside a dict rule when exact case matters.
    """
    if rules is None:
        return pd.Series(False, index=feature_names.index)
    normalized_rules = list(rules)
    if not normalized_rules:
        return pd.Series(False, index=feature_names.index)
    return feature_names.astype(str).apply(lambda name: any(_matches_rule(name, rule) for rule in normalized_rules))


def answer_feature_match_reason(feature_name: str, rules: list[AnswerRule] | set[str] | tuple[AnswerRule, ...] | None) -> str:
    if rules is None:
        return ""
    for rule in list(rules):
        if _matches_rule(str(feature_name), rule):
            return _rule_label(rule)
    return ""


def add_answer_feature_flags(
    frame: pd.DataFrame,
    *,
    feature_col: str,
    rules: list[AnswerRule] | set[str] | tuple[AnswerRule, ...] | None,
    flag_col: str = "is_answer_feature",
    reason_col: str = "answer_match_rule",
) -> pd.DataFrame:
    result = frame.copy()
    if feature_col not in result.columns:
        result[flag_col] = False
        result[reason_col] = ""
        return result
    result[flag_col] = answer_feature_mask(result[feature_col], rules)
    result[reason_col] = [
        answer_feature_match_reason(feature_name, rules) if is_answer else ""
        for feature_name, is_answer in zip(result[feature_col].astype(str), result[flag_col])
    ]
    return result


def _matches_rule(feature_name: str, rule: AnswerRule) -> bool:
    if isinstance(rule, str):
        return feature_name == rule
    if not isinstance(rule, dict):
        return False

    case_sensitive = bool(rule.get("case_sensitive", False))
    name = feature_name if case_sensitive else feature_name.lower()

    exact_values = _as_list(rule.get("exact"))
    if exact_values and any(name == _normalize_token(item, case_sensitive) for item in exact_values):
        return True

    contains_values = _as_list(rule.get("contains"))
    if contains_values and all(_normalize_token(item, case_sensitive) in name for item in contains_values):
        return True

    contains_all_values = _as_list(rule.get("contains_all"))
    if contains_all_values and all(_normalize_token(item, case_sensitive) in name for item in contains_all_values):
        return True

    contains_any_values = _as_list(rule.get("contains_any"))
    if contains_any_values and any(_normalize_token(item, case_sensitive) in name for item in contains_any_values):
        return True

    regex_values = _as_list(rule.get("regex"))
    flags = 0 if case_sensitive else re.IGNORECASE
    return bool(regex_values and any(re.search(str(pattern), feature_name, flags=flags) for pattern in regex_values))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_token(value: Any, case_sensitive: bool) -> str:
    text = str(value)
    return text if case_sensitive else text.lower()


def _rule_label(rule: AnswerRule) -> str:
    if isinstance(rule, str):
        return f"exact:{rule}"
    if not isinstance(rule, dict):
        return str(rule)
    for key in ("exact", "contains", "contains_all", "contains_any", "regex"):
        if key in rule:
            return f"{key}:{rule[key]}"
    return str(rule)
