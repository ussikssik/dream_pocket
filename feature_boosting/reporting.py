from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .answer_features import answer_feature_mask
from .metrics import mae, rmse


def prepare_output_dir(base_output_dir: str | Path, run_id: str) -> Path:
    output_dir = Path(base_output_dir) / run_id
    for child in ("rankings", "plots", "models"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"feature_boosting.{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def copy_config(config_path: str | Path, output_dir: Path) -> None:
    shutil.copy2(config_path, output_dir / "config_used.yaml")


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def baseline_residual_summary(
    df: pd.DataFrame,
    *,
    target_col: str,
    pred_col: str,
    residual_col: str,
    split_col: str,
    id_col: str,
    defects: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    rows = []
    for split, split_df in df.groupby(split_col, sort=False):
        rows.append(_residual_row("global", "global", split, split_df, target_col, pred_col, residual_col))
        ids = split_df[id_col].astype(str)
        for defect_id, groups in defects.items():
            rows.append(
                _residual_row(
                    defect_id,
                    "bad",
                    split,
                    split_df[ids.isin(groups.get("bad", set()))],
                    target_col,
                    pred_col,
                    residual_col,
                )
            )
            rows.append(
                _residual_row(
                    defect_id,
                    "good",
                    split,
                    split_df[ids.isin(groups.get("good", set()))],
                    target_col,
                    pred_col,
                    residual_col,
                )
            )
    return pd.DataFrame(rows)


def plot_residual_curve(curve: pd.DataFrame, output_dir: Path, answer_features_by_defect: dict[str, Any] | None = None) -> None:
    if curve.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for defect_id, group in curve.groupby("defect_id", sort=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        for split, color in (("train", "#59a14f"), ("valid", "#4e79a7"), ("test", "#e15759")):
            col = f"{split}_bad_rmse"
            if col in group.columns:
                ax.plot(group["round"], group[col], marker="o", color=color, label=f"{split} bad RMSE")
        answer_rules = answer_features_by_defect.get(str(defect_id), []) if answer_features_by_defect else []
        if answer_rules and "selected_feature" in group.columns:
            answers = group[answer_feature_mask(group["selected_feature"], answer_rules)]
            if not answers.empty:
                ax.scatter(answers["round"], answers["valid_bad_rmse"], marker="X", s=130, color="#2a9d8f", label="answer feature", zorder=6)
                for _, row in answers.iterrows():
                    ax.annotate(
                        str(row.get("selected_feature", ""))[:24],
                        (row["round"], row["valid_bad_rmse"]),
                        textcoords="offset points",
                        xytext=(5, 6),
                        fontsize=8,
                        color="#2a9d8f",
                    )
        ax.set_xlabel("round")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{defect_id} residual curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"{defect_id}_residual_curve.png", dpi=150)
        plt.close(fig)


def round_residual_summary(
    residual_curve: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    *,
    group: str = "bad",
) -> pd.DataFrame:
    """Build round-level mean absolute residual points.

    Round 0 comes from the baseline residual summary. Later rounds come from
    the selected-feature residual curve. The value is MAE, i.e.
    mean(abs(y - prediction)), which is a stable "average residual" measure.
    """
    rows: list[dict[str, object]] = []
    if baseline_summary is not None and not baseline_summary.empty:
        base = baseline_summary[baseline_summary["group"].astype(str) == group].copy()
        base = base[base["defect_id"].astype(str) != "global"]
        for _, row in base.iterrows():
            rows.append(
                {
                    "defect_id": row["defect_id"],
                    "round": 0,
                    "split": row["split"],
                    "group": group,
                    "selected_feature": "baseline",
                    "mean_abs_residual": row.get("mean_abs_residual", np.nan),
                }
            )

    if residual_curve is not None and not residual_curve.empty:
        for _, row in residual_curve.iterrows():
            for split in ("train", "valid", "test"):
                col = f"{split}_{group}_mae"
                if col not in residual_curve.columns:
                    continue
                rows.append(
                    {
                        "defect_id": row["defect_id"],
                        "round": int(row["round"]),
                        "split": split,
                        "group": group,
                        "selected_feature": row.get("selected_feature", ""),
                        "mean_abs_residual": row.get(col, np.nan),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["mean_abs_residual"] = pd.to_numeric(result["mean_abs_residual"], errors="coerce")
    return result.sort_values(["defect_id", "split", "round"]).reset_index(drop=True)


def final_metric_summary(
    final_metrics: pd.DataFrame,
    *,
    baseline_name: str = "baseline",
    final_name: str = "final",
) -> pd.DataFrame:
    """Compare baseline and final metrics on the same split/defect/group rows."""
    if final_metrics.empty:
        return pd.DataFrame()
    required = {"model_name", "defect_id", "group", "split"}
    missing = sorted(required - set(final_metrics.columns))
    if missing:
        raise ValueError(f"final_metrics is missing required columns: {missing}")

    id_cols = ["defect_id", "group", "split"]
    metric_cols = [col for col in ("n_samples", "mae", "rmse", "r2") if col in final_metrics.columns]
    baseline = (
        final_metrics[final_metrics["model_name"].astype(str) == baseline_name][id_cols + metric_cols]
        .copy()
        .rename(columns={col: f"baseline_{col}" for col in metric_cols})
    )
    final = (
        final_metrics[final_metrics["model_name"].astype(str) == final_name][id_cols + metric_cols]
        .copy()
        .rename(columns={col: f"final_{col}" for col in metric_cols})
    )
    summary = baseline.merge(final, on=id_cols, how="outer")
    if summary.empty:
        return summary

    if "baseline_n_samples" in summary.columns or "final_n_samples" in summary.columns:
        summary["n_samples"] = summary.get("final_n_samples", pd.Series(index=summary.index, dtype=float)).combine_first(
            summary.get("baseline_n_samples", pd.Series(index=summary.index, dtype=float))
        )

    for metric in ("rmse", "mae"):
        before_col = f"baseline_{metric}"
        after_col = f"final_{metric}"
        if before_col not in summary.columns or after_col not in summary.columns:
            continue
        before = pd.to_numeric(summary[before_col], errors="coerce")
        after = pd.to_numeric(summary[after_col], errors="coerce")
        reduction = before - after
        summary[f"{metric}_reduction"] = reduction
        summary[f"{metric}_reduction_pct"] = np.where(before.abs() > 0, 100.0 * reduction / before, np.nan)

    if "baseline_r2" in summary.columns and "final_r2" in summary.columns:
        summary["r2_delta"] = pd.to_numeric(summary["final_r2"], errors="coerce") - pd.to_numeric(summary["baseline_r2"], errors="coerce")

    summary["label"] = [_metric_label(defect_id, group) for defect_id, group in zip(summary["defect_id"], summary["group"])]
    return _sort_metric_summary(summary).reset_index(drop=True)


def plot_final_metric_comparison(
    summary_or_metrics: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    splits: tuple[str, ...] = ("valid", "test"),
    metrics: tuple[str, ...] = ("rmse", "mae"),
    title: str = "baseline vs final model metrics",
):
    """Plot baseline/final RMSE and MAE side by side by split and defect group."""
    if summary_or_metrics.empty:
        return None
    summary = summary_or_metrics.copy()
    if not any(col.startswith("baseline_") for col in summary.columns):
        summary = final_metric_summary(summary)
    if summary.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    available_splits = [split for split in splits if (summary["split"].astype(str) == split).any()]
    if not available_splits:
        available_splits = sorted(summary["split"].dropna().astype(str).unique())
    available_metrics = [metric for metric in metrics if f"baseline_{metric}" in summary.columns and f"final_{metric}" in summary.columns]
    if not available_splits or not available_metrics:
        return None

    max_rows = max(int((summary["split"].astype(str) == split).sum()) for split in available_splits)
    height = max(4.0, 0.45 * max_rows) * len(available_splits)
    fig, axes = plt.subplots(
        len(available_splits),
        len(available_metrics),
        figsize=(7.5 * len(available_metrics), height),
        squeeze=False,
    )

    for row_idx, split in enumerate(available_splits):
        split_df = _sort_metric_summary(summary[summary["split"].astype(str) == split].copy())
        for col_idx, metric in enumerate(available_metrics):
            _plot_final_metric_axis(axes[row_idx][col_idx], split_df, metric, split)

    fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_round_residual_points(
    summary: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    title: str = "round mean absolute residual",
    answer_features_by_defect: dict[str, Any] | None = None,
):
    """Plot round 0/1/2/... average residual points by defect."""
    if summary.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    splits = [split for split in ("train", "valid", "test") if (summary["split"].astype(str) == split).any()]
    if not splits:
        splits = sorted(summary["split"].dropna().astype(str).unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(8 * len(splits), 4.5), squeeze=False)
    axes = axes[0]
    for ax, split in zip(axes, splits):
        work = summary[summary["split"].astype(str) == split].copy()
        for defect_id, group_df in work.groupby("defect_id", sort=False):
            group_df = group_df.sort_values("round")
            ax.plot(group_df["round"], group_df["mean_abs_residual"], marker="o", linewidth=1.7, label=str(defect_id))
            answer_rules = answer_features_by_defect.get(str(defect_id), []) if answer_features_by_defect else []
            if answer_rules and "selected_feature" in group_df.columns:
                answers = group_df[answer_feature_mask(group_df["selected_feature"], answer_rules)]
                if not answers.empty:
                    ax.scatter(
                        answers["round"],
                        answers["mean_abs_residual"],
                        marker="X",
                        s=95,
                        color="#2a9d8f",
                        label=f"{defect_id} answer",
                        zorder=6,
                    )
            for _, row in group_df.iterrows():
                if int(row["round"]) == 0:
                    continue
                ax.annotate(
                    str(row.get("selected_feature", ""))[:24],
                    (row["round"], row["mean_abs_residual"]),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=8,
                    alpha=0.8,
                )
        ax.set_title(f"{split} {title}")
        ax.set_xlabel("round")
        ax.set_ylabel("mean abs residual (MAE)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_final_feature_set_summary(
    feature_summary: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    title: str = "final model feature set",
):
    if feature_summary.empty or not {"feature_type", "count"}.issubset(feature_summary.columns):
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    work = feature_summary.copy()
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#4e79a7" if item == "base" else "#f28e2b" for item in work["feature_type"].astype(str)]
    bars = ax.bar(work["feature_type"].astype(str), work["count"], color=colors, width=0.55)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}", (bar.get_x() + bar.get_width() / 2, height), ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("feature count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_candidate_loss_ranking(
    ranking_df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    global_metric_col: str = "valid_global_rmse_after_over_baseline",
    bad_metric_col: str = "valid_bad_rmse_after_over_baseline",
    selected_col: str = "selected",
    answer_features: Any = None,
    answer_col: str = "is_answer_feature",
    title_prefix: str = "candidate loss after residual boost",
):
    """Plot candidate rank vs after-boosting loss for global and bad groups.

    The x-axis is rank after sorting each metric in ascending order, so lower
    points on the left are better. This mirrors the manual fb_metric plots:
    each dot is one candidate feature, and selected features are highlighted.
    """
    if ranking_df.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), squeeze=False)
    axes = axes[0]
    _plot_loss_axis(ranking_df, global_metric_col, axes[0], "all wafers", selected_col, answer_features, answer_col)
    _plot_loss_axis(ranking_df, bad_metric_col, axes[1], "bad group", selected_col, answer_features, answer_col)
    fig.suptitle(title_prefix)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def _plot_loss_axis(
    ranking_df: pd.DataFrame,
    metric_col: str,
    ax,
    title: str,
    selected_col: str,
    answer_features: Any,
    answer_col: str,
) -> None:
    if metric_col not in ranking_df.columns:
        ax.set_title(title)
        ax.text(0.5, 0.5, f"missing column: {metric_col}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    source = ranking_df.copy()
    source[metric_col] = pd.to_numeric(source[metric_col], errors="coerce")
    source_answer_mask = pd.Series(False, index=source.index)
    if answer_features and "feature_name" in source.columns:
        source_answer_mask = source_answer_mask | answer_feature_mask(source["feature_name"], answer_features)
    if answer_col in source.columns:
        source_answer_mask = source_answer_mask | source[answer_col].fillna(False).astype(bool)
    answer_rows_all = source[source_answer_mask].copy()

    work = source.dropna(subset=[metric_col]).copy()
    work = work.dropna(subset=[metric_col]).sort_values([metric_col, "feature_name"]).reset_index(drop=True)
    if work.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no finite metric values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(1, len(work) + 1)
    ax.scatter(x, work[metric_col], s=13, alpha=0.75, label=metric_col)
    if "over_baseline" in metric_col:
        ax.axhline(1.0, color="#6c757d", linestyle="--", linewidth=1.0, alpha=0.8, label="baseline residual ratio = 1.0")

    if selected_col in work.columns:
        selected = work[work[selected_col].astype(bool)]
        if not selected.empty:
            selected_x = selected.index.to_numpy() + 1
            ax.scatter(selected_x, selected[metric_col], marker="*", s=140, color="#e76f51", label="selected feature", zorder=5)
            for xpos, (_, row) in zip(selected_x, selected.iterrows()):
                ax.annotate(f"rank {int(xpos)}", (xpos, row[metric_col]), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#e76f51")

    answer_mask = pd.Series(False, index=work.index)
    if answer_features and "feature_name" in work.columns:
        answer_mask = answer_mask | answer_feature_mask(work["feature_name"], answer_features)
    if answer_col in work.columns:
        answer_mask = answer_mask | work[answer_col].fillna(False).astype(bool)
    answers = work[answer_mask]
    answer_requested = bool(answer_features) or not answer_rows_all.empty
    if not answers.empty:
        answer_x = answers.index.to_numpy() + 1
        ax.scatter(
            answer_x,
            answers[metric_col],
            marker="X",
            s=220,
            color="#2a9d8f",
            edgecolors="#0b3d35",
            linewidths=1.4,
            label="answer feature",
            zorder=7,
        )
        for xpos, (_, row) in zip(answer_x, answers.iterrows()):
            ax.annotate(
                f"ANSWER rank {int(xpos)}\n{str(row.get('feature_name', ''))[:24]}",
                (xpos, row[metric_col]),
                textcoords="offset points",
                xytext=(7, -18),
                fontsize=8,
                color="#2a9d8f",
                fontweight="bold",
            )
    missing_metric_answers = answer_rows_all[pd.to_numeric(answer_rows_all[metric_col], errors="coerce").isna()]
    if not missing_metric_answers.empty:
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if y_max > y_min else 1.0
        marker_y = y_max - 0.06 * y_span
        marker_x_values = []
        for _, row in missing_metric_answers.iterrows():
            xpos = _answer_rank_x(row, fallback=len(work) + 1)
            marker_x_values.append(xpos)
            ax.axvline(xpos, color="#2a9d8f", linestyle=":", linewidth=1.0, alpha=0.8)
            ax.scatter(
                [xpos],
                [marker_y],
                marker="X",
                s=220,
                color="#2a9d8f",
                edgecolors="#0b3d35",
                linewidths=1.4,
                label="answer feature (metric NaN)",
                zorder=7,
            )
            reason = str(row.get("fail_reason", "") or row.get("overfit_guard_reason", "") or "metric NaN")
            ax.annotate(
                f"ANSWER rank {int(xpos)}\n{str(row.get('feature_name', ''))[:24]}\n{reason[:34]}",
                (xpos, marker_y),
                textcoords="offset points",
                xytext=(7, -18),
                fontsize=8,
                color="#2a9d8f",
                fontweight="bold",
            )
        if marker_x_values:
            left, right = ax.get_xlim()
            ax.set_xlim(left=min(left, 0.5), right=max(right, max(marker_x_values) + 1.0))
    if answers.empty and missing_metric_answers.empty and answer_requested:
        ax.text(
            0.02,
            0.96,
            "answer feature not found in this plotted ranking\ncheck ANSWER_FEATURES, matched columns, or kernel restart",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#b00020",
            bbox={"facecolor": "white", "edgecolor": "#b00020", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
        )

    ax.set_title(title)
    ax.set_xlabel("feature rank (ascending loss)")
    ylabel = "after residual / baseline residual" if "over_baseline" in metric_col else metric_col
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _answer_rank_x(row: pd.Series, *, fallback: int) -> int:
    try:
        value = int(row.get("rank", fallback))
    except (TypeError, ValueError):
        return int(fallback)
    return value if value > 0 else int(fallback)


def _plot_final_metric_axis(ax, split_df: pd.DataFrame, metric: str, split: str) -> None:
    if split_df.empty:
        ax.set_title(f"{split} {metric.upper()}")
        ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    labels = split_df["label"].astype(str).tolist()
    y = np.arange(len(split_df))
    baseline = pd.to_numeric(split_df[f"baseline_{metric}"], errors="coerce")
    final = pd.to_numeric(split_df[f"final_{metric}"], errors="coerce")
    ax.barh(y - 0.18, baseline, height=0.36, color="#4e79a7", label="baseline")
    ax.barh(y + 0.18, final, height=0.36, color="#f28e2b", label="final")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"{split} {metric.upper()} (lower is better)")
    ax.set_xlabel(metric.upper())
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    finite_values = pd.concat([baseline, final]).dropna()
    max_value = float(finite_values.max()) if not finite_values.empty else 0.0
    x_pad = max(max_value * 0.015, 1e-9)
    for ypos, (_, row) in enumerate(split_df.iterrows()):
        pct = row.get(f"{metric}_reduction_pct", np.nan)
        if pd.isna(pct) or not np.isfinite(float(pct)):
            continue
        row_values = pd.to_numeric(pd.Series([row.get(f"baseline_{metric}", np.nan), row.get(f"final_{metric}", np.nan)]), errors="coerce").dropna()
        if row_values.empty:
            continue
        x_value = float(row_values.max())
        color = "#2a9d8f" if float(pct) >= 0 else "#d62828"
        ax.text(x_value + x_pad, ypos, f"{float(pct):+.1f}%", va="center", fontsize=8, color=color)
    ax.margins(x=0.15)


def _sort_metric_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    work = summary.copy()
    split_order = {"train": 0, "valid": 1, "test": 2}
    group_order = {"global": 0, "bad": 1, "good": 2}
    work["_split_order"] = work["split"].astype(str).map(split_order).fillna(99)
    work["_defect_order"] = np.where(work["defect_id"].astype(str) == "global", "", work["defect_id"].astype(str))
    work["_group_order"] = work["group"].astype(str).map(group_order).fillna(99)
    work = work.sort_values(["_split_order", "_defect_order", "_group_order", "label"])
    return work.drop(columns=["_split_order", "_defect_order", "_group_order"], errors="ignore")


def _metric_label(defect_id: object, group: object) -> str:
    if str(defect_id) == "global" and str(group) == "global":
        return "all wafers"
    return f"{defect_id} / {group}"


def _residual_row(
    defect_id: str,
    group: str,
    split: object,
    frame: pd.DataFrame,
    target_col: str,
    pred_col: str,
    residual_col: str,
) -> dict[str, object]:
    residual = pd.to_numeric(frame[residual_col], errors="coerce") if len(frame) else pd.Series(dtype=float)
    return {
        "defect_id": defect_id,
        "group": group,
        "split": split,
        "n_samples": len(frame),
        "mae": mae(frame[target_col], frame[pred_col]) if len(frame) else np.nan,
        "rmse": rmse(frame[target_col], frame[pred_col]) if len(frame) else np.nan,
        "mean_residual": float(residual.mean()) if len(residual) else np.nan,
        "mean_abs_residual": float(residual.abs().mean()) if len(residual) else np.nan,
    }
