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
