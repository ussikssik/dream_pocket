from __future__ import annotations

import pandas as pd


REQUIRED_SPLITS = ("train", "valid", "test")


def validate_splits(df: pd.DataFrame, split_col: str) -> None:
    if split_col not in df.columns:
        raise ValueError(f"dataset is missing split column {split_col!r}")
    present = set(df[split_col].dropna().astype(str))
    missing = [split for split in REQUIRED_SPLITS if split not in present]
    if missing:
        raise ValueError(f"dataset must contain train/valid/test splits; missing={missing}")


def split_frame(df: pd.DataFrame, split_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_splits(df, split_col)
    split = df[split_col].astype(str)
    train_df = df[split == "train"].copy()
    valid_df = df[split == "valid"].copy()
    test_df = df[split == "test"].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("train/valid/test split must all be non-empty")
    return train_df, valid_df, test_df
