from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


def as_binary_target(y: pd.Series, positive_label: Any) -> pd.Series:
    target = (y == positive_label).astype(int)
    target.name = y.name
    return target


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator is None or abs(denominator) < EPS:
        return default
    return float(numerator / denominator)


def robust_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lower = values.quantile(0.02)
    upper = values.quantile(0.98)
    clipped = values.clip(lower=lower, upper=upper)
    span = upper - lower
    if not np.isfinite(span) or abs(span) < EPS:
        return pd.Series(0.0, index=series.index)
    return ((clipped - lower) / span).fillna(0.0).clip(0.0, 1.0)


def signed_log10_pvalue(p_value: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_value, errors="coerce").clip(lower=1e-300)
    return -np.log10(values)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return result
    ordered = valid.sort_values()
    m = float(len(ordered))
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    adjusted = ordered.to_numpy() * m / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result.loc[ordered.index] = adjusted
    return result


def infer_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    group_cols: Iterable[str],
    sample_id_cols: Iterable[str],
    exclude_cols: Iterable[str] = (),
) -> list[str]:
    excluded = {label_col, *group_cols, *sample_id_cols, *exclude_cols}
    return [col for col in df.columns if col not in excluded]


def compact_warning(parts: Iterable[str]) -> str:
    return "; ".join(part for part in parts if part)


def numeric_or_none(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    converted = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    if non_null == 0:
        return None
    if converted.notna().sum() / non_null >= 0.95:
        return converted.astype(float)
    return None


def effect_direction(effect: float | None, presence_type: str | None = None) -> str:
    if presence_type in {"bad_only", "bad_enriched_sparse"}:
        return "bad_presence"
    if presence_type in {"good_only", "good_enriched_sparse"}:
        return "good_presence"
    if effect is None or not np.isfinite(effect) or abs(effect) < EPS:
        return "flat"
    return "bad_higher" if effect > 0 else "good_higher"


def stable_random(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def normalize_group_frame(groups: pd.DataFrame | pd.Series | None) -> pd.Series | None:
    if groups is None:
        return None
    if isinstance(groups, pd.Series):
        return groups.astype(str)
    if groups.shape[1] == 0:
        return None
    return groups.astype(str).agg("|".join, axis=1)


def cramers_v_from_table(table: pd.DataFrame) -> float:
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total <= 0:
        return 0.0
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    r, k = observed.shape
    denom = total * max(min(k - 1, r - 1), 1)
    return math.sqrt(max(chi2 / denom, 0.0))
