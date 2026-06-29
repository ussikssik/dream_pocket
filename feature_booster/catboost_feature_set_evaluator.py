from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .utils import as_binary_target


DataLoader = Callable[[object], pd.DataFrame]


@dataclass(frozen=True)
class CatBoostFeatureSetEvalConfig:
    """Train CatBoost on selected feature sets and rank features by SHAP delta."""

    label_col: str = "target_bad_a"
    positive_label: object = 1
    y_col: str = "eds_bin_a_wf_mean"
    role_col: str = "booster_sample_role"
    selected_features_per_order: int = 20
    shap_top_n: int = 15
    output_dir: Path | None = None
    plot_enabled: bool = True
    iterations: int = 300
    learning_rate: float = 0.05
    depth: int = 4
    l2_leaf_reg: float = 6.0
    random_seed: int = 42
    thread_count: int = -1
    verbose: bool = False


def evaluate_catboost_feature_sets(
    selection_result: pd.DataFrame,
    train_loader: DataLoader,
    full_pool_loader: DataLoader | None = None,
    config: CatBoostFeatureSetEvalConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each method's selected features with a fresh CatBoost model.

    For each order/method, the function:
    1. takes the first N selected features,
    2. trains CatBoost on only those features,
    3. computes abs(mean SHAP in Bad - mean SHAP in Good),
    4. saves top-N feature-vs-y plots when matplotlib is available.
    """

    config = config or CatBoostFeatureSetEvalConfig()
    if selection_result.empty:
        return pd.DataFrame(), pd.DataFrame()

    output_dir = config.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []

    for (order_id, method), group in selection_result.groupby(["order_id", "method"], sort=False):
        features = _selected_features(group, config.selected_features_per_order)
        if not features:
            metrics.append(_metric_record(order_id, method, "no_selected_features", 0))
            continue

        train_df = train_loader(order_id)
        available_features = [feature for feature in features if feature in train_df.columns]
        missing_features = [feature for feature in features if feature not in train_df.columns]
        if not available_features:
            metrics.append(_metric_record(order_id, method, "selected_features_missing_in_train_data", 0))
            continue

        report, metric = _fit_catboost_and_shap(
            train_df=train_df,
            order_id=order_id,
            method=method,
            features=available_features,
            missing_features=missing_features,
            config=config,
        )
        reports.append(report)
        metrics.append(metric)

        if config.plot_enabled and metric.get("model_status") == "ok" and full_pool_loader is not None:
            full_pool_df = full_pool_loader(order_id)
            plot_path, plot_warning = plot_shap_delta_top_features(
                full_pool_df,
                report[report["shap_rank"] <= config.shap_top_n],
                order_id=order_id,
                method=str(method),
                y_col=config.y_col,
                label_col=config.label_col,
                role_col=config.role_col,
                output_dir=output_dir / "plots" if output_dir is not None else None,
            )
            reports[-1]["plot_path"] = str(plot_path) if plot_path else ""
            reports[-1]["plot_warning"] = plot_warning

    combined = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    metric_df = pd.DataFrame(metrics)

    if output_dir is not None:
        combined.to_csv(output_dir / "catboost_shap_delta_by_method.csv", index=False, encoding="utf-8-sig")
        top = combined[combined.get("shap_rank", pd.Series(dtype=float)) <= config.shap_top_n] if not combined.empty else combined
        top.to_csv(output_dir / "catboost_shap_delta_top_features.csv", index=False, encoding="utf-8-sig")
        metric_df.to_csv(output_dir / "catboost_model_metrics_by_method.csv", index=False, encoding="utf-8-sig")

    return combined, metric_df


def plot_shap_delta_top_features(
    full_pool_df: pd.DataFrame,
    shap_top: pd.DataFrame,
    order_id: object,
    method: str,
    y_col: str = "eds_bin_a_wf_mean",
    label_col: str = "target_bad_a",
    role_col: str = "booster_sample_role",
    output_dir: Path | None = None,
) -> tuple[Path | None, str]:
    """Save one grid plot of SHAP-delta top features versus y."""

    if shap_top.empty:
        return None, "no_shap_top_features"
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None, "matplotlib_not_installed"

    if y_col not in full_pool_df.columns:
        return None, f"missing_y_col:{y_col}"

    features = [feature for feature in shap_top["feature_name"].tolist() if feature in full_pool_df.columns]
    if not features:
        return None, "top_features_missing_in_full_pool"

    y = pd.to_numeric(full_pool_df[y_col], errors="coerce")
    roles = _roles(full_pool_df, label_col=label_col, role_col=role_col)
    rows = math.ceil(len(features) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, max(4, rows * 3.4)), squeeze=False)
    fig.suptitle(f"order={order_id} | method={method} | CatBoost SHAP delta top {len(features)}", fontweight="bold")

    shap_lookup = shap_top.set_index("feature_name").to_dict(orient="index")
    for idx, feature in enumerate(features):
        ax = axes[idx // 3][idx % 3]
        x = full_pool_df[feature]
        numeric = pd.to_numeric(x, errors="coerce")
        if numeric.notna().sum() >= max(3, int(x.notna().sum() * 0.8)):
            _scatter_by_role(ax, numeric, y, roles)
            ax.set_xlabel(feature)
        else:
            codes, uniques = pd.factorize(x.astype("object").where(x.notna(), "__MISSING__"))
            jitter = np.random.default_rng(42).normal(0, 0.04, size=len(codes))
            _scatter_by_role(ax, pd.Series(codes + jitter, index=x.index), y, roles)
            tick_count = min(len(uniques), 6)
            ax.set_xticks(range(tick_count))
            ax.set_xticklabels([str(v)[:12] for v in uniques[:tick_count]], rotation=25, ha="right")
            ax.set_xlabel(feature)

        row = shap_lookup.get(feature, {})
        ax.set_title(
            f"rank={int(row.get('shap_rank', idx + 1))} | delta={float(row.get('shap_delta_abs', np.nan)):.4f}",
            fontsize=9,
        )
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.25)

    for idx in range(len(features), rows * 3):
        axes[idx // 3][idx % 3].set_axis_off()

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 4))
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    if output_dir is None:
        return None, ""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"order_{order_id}_{_slug(method)}_shap_delta_top_features.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, ""


def _fit_catboost_and_shap(
    train_df: pd.DataFrame,
    order_id: object,
    method: object,
    features: list[str],
    missing_features: list[str],
    config: CatBoostFeatureSetEvalConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception:
        report = _unavailable_report(order_id, method, features, "catboost_not_installed", missing_features)
        metric = _metric_record(order_id, method, "catboost_not_installed", len(features), missing_features=missing_features)
        return report, metric

    if config.label_col not in train_df.columns:
        report = _unavailable_report(order_id, method, features, f"missing_label_col:{config.label_col}", missing_features)
        metric = _metric_record(order_id, method, f"missing_label_col:{config.label_col}", len(features), missing_features=missing_features)
        return report, metric

    y = as_binary_target(train_df[config.label_col], config.positive_label).astype(int)
    if y.nunique() < 2:
        report = _unavailable_report(order_id, method, features, "single_class_target", missing_features)
        metric = _metric_record(order_id, method, "single_class_target", len(features), missing_features=missing_features)
        return report, metric

    X = train_df[features].copy()
    cat_feature_indices = [
        idx
        for idx, col in enumerate(X.columns)
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col])
    ]
    X_prepared = _prepare_for_catboost(X, cat_feature_indices)

    try:
        pool = Pool(X_prepared, y, cat_features=cat_feature_indices)
        model = CatBoostClassifier(
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            loss_function="Logloss",
            random_seed=config.random_seed,
            thread_count=config.thread_count,
            verbose=config.verbose,
            allow_writing_files=False,
        )
        model.fit(pool)
        probability = model.predict_proba(pool)[:, 1]
        shap_values = model.get_feature_importance(pool, type="ShapValues")
    except Exception as exc:
        warning = f"catboost_failed:{type(exc).__name__}"
        report = _unavailable_report(order_id, method, features, warning, missing_features)
        metric = _metric_record(order_id, method, warning, len(features), missing_features=missing_features)
        return report, metric

    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape[1] == len(features) + 1:
        values = values[:, :-1]

    bad_mask = y.to_numpy(dtype=int) == 1
    good_mask = ~bad_mask
    rows = []
    for idx, feature in enumerate(features):
        feature_shap = values[:, idx]
        bad_mean = float(np.nanmean(feature_shap[bad_mask])) if bad_mask.any() else np.nan
        good_mean = float(np.nanmean(feature_shap[good_mask])) if good_mask.any() else np.nan
        delta = bad_mean - good_mean if np.isfinite(bad_mean) and np.isfinite(good_mean) else np.nan
        rows.append(
            {
                "order_id": order_id,
                "method": method,
                "feature_name": feature,
                "selected_feature_count": len(features),
                "shap_bad_mean": bad_mean,
                "shap_good_mean": good_mean,
                "shap_delta_bad_minus_good": delta,
                "shap_delta_abs": abs(delta) if np.isfinite(delta) else np.nan,
                "shap_abs_mean": float(np.nanmean(np.abs(feature_shap))),
                "model_auc_train": _binary_auc(y.to_numpy(dtype=int), probability),
                "model_logloss_train": _binary_logloss(y.to_numpy(dtype=int), probability),
                "model_status": "ok",
                "evaluation_warning": "; ".join([f"missing_features:{len(missing_features)}"] if missing_features else []),
            }
        )

    report = pd.DataFrame(rows).sort_values("shap_delta_abs", ascending=False).reset_index(drop=True)
    report.insert(3, "shap_rank", range(1, len(report) + 1))
    metric = _metric_record(
        order_id,
        method,
        "ok",
        len(features),
        missing_features=missing_features,
        model_auc_train=float(report["model_auc_train"].iloc[0]) if not report.empty else np.nan,
        model_logloss_train=float(report["model_logloss_train"].iloc[0]) if not report.empty else np.nan,
        mean_top_shap_delta_abs=float(report["shap_delta_abs"].head(config.shap_top_n).mean()) if not report.empty else np.nan,
    )
    return report, metric


def _selected_features(group: pd.DataFrame, top_n: int) -> list[str]:
    work = group.copy()
    if "rank" in work.columns:
        work = work.sort_values("rank")
    elif "final_rank" in work.columns:
        work = work.sort_values("final_rank")
    features = []
    for feature in work["feature_name"].dropna().astype(str):
        if feature not in features:
            features.append(feature)
        if len(features) >= top_n:
            break
    return features


def _unavailable_report(
    order_id: object,
    method: object,
    features: list[str],
    warning: str,
    missing_features: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": order_id,
            "method": method,
            "feature_name": features,
            "shap_rank": np.nan,
            "selected_feature_count": len(features),
            "shap_bad_mean": np.nan,
            "shap_good_mean": np.nan,
            "shap_delta_bad_minus_good": np.nan,
            "shap_delta_abs": np.nan,
            "shap_abs_mean": np.nan,
            "model_auc_train": np.nan,
            "model_logloss_train": np.nan,
            "model_status": "skipped",
            "evaluation_warning": warning
            + (f"; missing_features:{len(missing_features)}" if missing_features else ""),
            "plot_path": "",
            "plot_warning": "",
        }
    )


def _metric_record(
    order_id: object,
    method: object,
    model_status: str,
    selected_feature_count: int,
    missing_features: list[str] | None = None,
    model_auc_train: float = np.nan,
    model_logloss_train: float = np.nan,
    mean_top_shap_delta_abs: float = np.nan,
) -> dict[str, object]:
    missing_features = missing_features or []
    return {
        "order_id": order_id,
        "method": method,
        "model_status": model_status,
        "selected_feature_count": selected_feature_count,
        "missing_feature_count": len(missing_features),
        "model_auc_train": model_auc_train,
        "model_logloss_train": model_logloss_train,
        "mean_top_shap_delta_abs": mean_top_shap_delta_abs,
    }


def _prepare_for_catboost(X: pd.DataFrame, cat_feature_indices: list[int]) -> pd.DataFrame:
    prepared = X.copy()
    cat_cols = {prepared.columns[idx] for idx in cat_feature_indices}
    for col in prepared.columns:
        if col in cat_cols:
            prepared[col] = prepared[col].astype("object").where(prepared[col].notna(), "__MISSING__")
        else:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    return prepared


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    rank_sum_pos = float(ranks[y_true == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc)


def _binary_logloss(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    score = np.clip(score, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y_true * np.log(score) + (1.0 - y_true) * np.log(1.0 - score)))


def _roles(df: pd.DataFrame, label_col: str, role_col: str) -> pd.Series:
    if role_col in df.columns:
        return df[role_col].astype("object").where(df[role_col].notna(), "Ignored").astype(str)
    if label_col in df.columns:
        label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        return pd.Series(np.where(label == 1, "Bad", "Good"), index=df.index)
    return pd.Series("All", index=df.index)


def _scatter_by_role(ax, x: pd.Series, y: pd.Series, roles: pd.Series) -> None:
    for role in _ordered_roles(roles):
        color, size, alpha = _role_style(role)
        mask = roles == role
        if mask.any():
            ax.scatter(x[mask], y[mask], s=size, alpha=alpha, c=color, label=role)


def _ordered_roles(roles: pd.Series) -> list[str]:
    present = {str(role) for role in roles.dropna().unique()}
    preferred = ["Ignored", "Good", "Bad", "All"]
    ordered = [role for role in preferred if role in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _role_style(role: object) -> tuple[str, int, float]:
    styles = {
        "Ignored": ("#9e9e9e", 12, 0.22),
        "Good": ("#1f77b4", 18, 0.65),
        "Bad": ("#d62728", 22, 0.78),
        "All": ("#4c4c4c", 16, 0.55),
    }
    return styles.get(str(role), ("#9467bd", 16, 0.50))


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9가-힣_-]+", "_", text)
    return text.strip("_") or "method"
