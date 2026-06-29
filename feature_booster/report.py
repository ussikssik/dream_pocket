from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_order_report(df: pd.DataFrame, output_dir: Path, order_id: object) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"order_{order_id}_feature_evidence.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_combined_report(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "combined_feature_evidence.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
