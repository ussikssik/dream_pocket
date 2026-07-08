from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported table format: {path}")


def load_base_dataset(path: str | Path) -> pd.DataFrame:
    return read_table(path)


def load_candidate_features(path: str | Path) -> pd.DataFrame:
    return read_table(path)


def load_base_feature_cols(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"base feature column file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_group_ids(path: str | Path, id_col: str) -> set[str]:
    frame = read_table(path)
    if id_col not in frame.columns:
        raise ValueError(f"group file {path} must contain id column {id_col!r}")
    return set(frame[id_col].dropna().astype(str))


def align_base_and_candidates(base_df: pd.DataFrame, candidate_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in base_df.columns:
        raise ValueError(f"base dataset is missing id column {id_col!r}")
    if id_col not in candidate_df.columns:
        raise ValueError(f"candidate features are missing id column {id_col!r}")
    base = base_df.copy()
    candidate = candidate_df.copy()
    base[id_col] = base[id_col].astype(str)
    candidate[id_col] = candidate[id_col].astype(str)
    merged = base.merge(candidate, on=id_col, how="inner", suffixes=("", "__candidate_dup"))
    if merged.empty:
        raise ValueError(f"base dataset and candidate features do not match on {id_col!r}")
    duplicate_cols = [col for col in merged.columns if col.endswith("__candidate_dup")]
    if duplicate_cols:
        merged = merged.drop(columns=duplicate_cols)
    return merged


def candidate_feature_cols(candidate_df: pd.DataFrame, id_col: str) -> list[str]:
    return [col for col in candidate_df.columns if col != id_col]
