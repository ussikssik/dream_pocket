from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def plot_residual_curve(curve: pd.DataFrame, output_dir: Path) -> None:
    if curve.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for defect_id, group in curve.groupby("defect_id", sort=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(group["round"], group["valid_bad_rmse"], marker="o", label="valid bad RMSE")
        ax.plot(group["round"], group["test_bad_rmse"], marker="o", label="test bad RMSE")
        ax.set_xlabel("round")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{defect_id} residual curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"{defect_id}_residual_curve.png", dpi=150)
        plt.close(fig)


def plot_candidate_loss_ranking(
    ranking_df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    global_metric_col: str = "valid_global_rmse_after",
    bad_metric_col: str = "valid_bad_rmse_after",
    selected_col: str = "selected",
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
    _plot_loss_axis(ranking_df, global_metric_col, axes[0], "all wafers", selected_col)
    _plot_loss_axis(ranking_df, bad_metric_col, axes[1], "bad group", selected_col)
    fig.suptitle(title_prefix)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def _plot_loss_axis(ranking_df: pd.DataFrame, metric_col: str, ax, title: str, selected_col: str) -> None:
    if metric_col not in ranking_df.columns:
        ax.set_title(title)
        ax.text(0.5, 0.5, f"missing column: {metric_col}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    work = ranking_df.dropna(subset=[metric_col]).copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col]).sort_values([metric_col, "feature_name"]).reset_index(drop=True)
    if work.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no finite metric values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(1, len(work) + 1)
    ax.scatter(x, work[metric_col], s=13, alpha=0.75, label=metric_col)

    if selected_col in work.columns:
        selected = work[work[selected_col].astype(bool)]
        if not selected.empty:
            selected_x = selected.index.to_numpy() + 1
            ax.scatter(selected_x, selected[metric_col], marker="*", s=140, color="#e76f51", label="selected feature", zorder=5)
            for xpos, (_, row) in zip(selected_x, selected.iterrows()):
                ax.annotate(f"rank {int(xpos)}", (xpos, row[metric_col]), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#e76f51")

    ax.set_title(title)
    ax.set_xlabel("feature rank (ascending loss)")
    ax.set_ylabel(metric_col)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


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
