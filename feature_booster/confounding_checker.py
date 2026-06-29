from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ConfoundingConfig
from .utils import cramers_v_from_table, normalize_group_frame, numeric_or_none


def check_group_confounding(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.DataFrame | pd.Series | None,
    candidate_features: list[str],
    config: ConfoundingConfig,
) -> pd.DataFrame:
    """Estimate whether a candidate feature mostly follows lot/tool/product groups."""

    result = pd.DataFrame(index=pd.Index(candidate_features, name="feature_name"))
    result["group_eta_or_cramers_v"] = np.nan
    result["label_group_concentration"] = np.nan
    result["confounding_risk_score"] = 0.0
    result["confounding_warning"] = ""

    group_key = normalize_group_frame(groups)
    if not config.enabled or group_key is None or group_key.nunique() <= 1:
        result["confounding_warning"] = "group_info_unavailable"
        return result

    y = y.astype(int)
    group_key = _cap_groups(group_key, config.max_groups)
    label_concentration = _label_group_concentration(y, group_key)

    for feature in candidate_features:
        association = _feature_group_association(X[feature], group_key)
        risk = max(association, label_concentration * association)
        if risk >= config.high_risk_threshold:
            warning = "high_group_confounding_risk"
        elif risk >= config.medium_risk_threshold:
            warning = "medium_group_confounding_risk"
        else:
            warning = ""

        result.loc[feature, "group_eta_or_cramers_v"] = association
        result.loc[feature, "label_group_concentration"] = label_concentration
        result.loc[feature, "confounding_risk_score"] = float(np.clip(risk, 0.0, 1.0))
        result.loc[feature, "confounding_warning"] = warning

    return result


def compute_groupwise_stability(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.DataFrame | pd.Series | None,
    candidate_features: list[str],
) -> pd.Series:
    group_key = normalize_group_frame(groups)
    if group_key is None or group_key.nunique() <= 1:
        return pd.Series(0.0, index=candidate_features, name="groupwise_stability_score")

    y = y.astype(int)
    scores = {}
    for feature in candidate_features:
        directions: list[float] = []
        for _, idx in group_key.groupby(group_key).groups.items():
            idx_list = list(idx)
            y_group = y.loc[idx_list]
            if y_group.nunique() < 2:
                continue
            direction = _within_group_direction(X.loc[idx_list, feature], y_group)
            if np.isfinite(direction) and abs(direction) > 1e-12:
                directions.append(float(np.sign(direction)))
        if not directions:
            scores[feature] = 0.0
        else:
            scores[feature] = abs(float(np.mean(directions)))
    return pd.Series(scores, name="groupwise_stability_score")


def _within_group_direction(series: pd.Series, y: pd.Series) -> float:
    numeric = numeric_or_none(series)
    if numeric is not None and numeric.dropna().nunique() > 5:
        bad = numeric[y == 1].dropna()
        good = numeric[y == 0].dropna()
        if len(bad) < 1 or len(good) < 1:
            return np.nan
        return float(bad.median() - good.median())
    presence = series.notna().astype(float)
    return float(presence[y == 1].mean() - presence[y == 0].mean())


def _feature_group_association(series: pd.Series, group_key: pd.Series) -> float:
    numeric = numeric_or_none(series)
    if numeric is not None and numeric.dropna().nunique() > 5:
        frame = pd.DataFrame({"value": numeric, "group": group_key}).dropna()
        if frame.empty or frame["value"].var() <= 0:
            return 0.0
        overall_mean = frame["value"].mean()
        grouped = frame.groupby("group", observed=True)["value"].agg(["mean", "count"])
        between = float((grouped["count"] * (grouped["mean"] - overall_mean) ** 2).sum())
        total = float(((frame["value"] - overall_mean) ** 2).sum())
        return float(np.clip(between / total if total > 0 else 0.0, 0.0, 1.0))

    clean = series.astype("object").where(series.notna(), "__MISSING__")
    table = pd.crosstab(clean, group_key)
    return float(np.clip(cramers_v_from_table(table), 0.0, 1.0))


def _label_group_concentration(y: pd.Series, group_key: pd.Series) -> float:
    bad = group_key[y == 1]
    if bad.empty:
        return 0.0
    rates = bad.value_counts(normalize=True)
    return float(rates.iloc[0])


def _cap_groups(group_key: pd.Series, max_groups: int) -> pd.Series:
    if group_key.nunique() <= max_groups:
        return group_key
    top = set(group_key.value_counts().head(max_groups).index)
    return group_key.where(group_key.isin(top), "__OTHER_GROUP__")
