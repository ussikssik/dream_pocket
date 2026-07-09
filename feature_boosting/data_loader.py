from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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


def standardize_six_file_inputs(
    *,
    y_path: str | Path,
    candidate_path: str | Path,
    base_feature_path: str | Path,
    defect_groups: list[dict[str, Any]],
    output_dir: str | Path,
    id_col: str = "sample_id",
    target_col: str = "yield",
    split_col: str = "split",
    y_lot_col: str = "lot",
    y_wf_col: str = "wf",
    y_id_col: str | None = None,
    y_combined_id_col: str | None = None,
    y_target_col: str = "y",
    candidate_lot_col: str = "lot",
    candidate_wf_col: str = "wf",
    candidate_id_col: str | None = None,
    candidate_combined_id_col: str | None = None,
    base_lot_col: str = "lot",
    base_wf_col: str = "wf",
    base_id_col: str | None = None,
    base_combined_id_col: str | None = None,
    combined_id_sep: str = "_",
    split_source_col: str | None = None,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Convert raw lot/wf/y + base/candidate + good_bad files to PoC inputs.

    Expected raw inputs:
    1. y file: lot/wf/y columns, one combined lot_wf column, or one id column
    2. candidate file: lot/wf/candidate feature columns, one combined lot_wf column, or one id column
    3. base feature file: lot/wf/base feature columns, one combined lot_wf column, or one id column
    4. one group file per defect: lot/wf/good_bad columns, one combined lot_wf column, or one id column

    Column names are configurable. The standardized output uses the existing
    pipeline contract: sample_id, yield, split, base feature columns,
    candidate feature table, base_feature_cols.txt, and bad/good CSVs.
    """
    output_dir = Path(output_dir)
    group_out_dir = output_dir / "groups"
    group_out_dir.mkdir(parents=True, exist_ok=True)

    y_df = _with_sample_id(
        read_table(y_path),
        lot_col=y_lot_col,
        wf_col=y_wf_col,
        source_id_col=y_id_col,
        combined_id_col=y_combined_id_col,
        combined_id_sep=combined_id_sep,
        id_col=id_col,
    )
    candidate_df = _with_sample_id(
        read_table(candidate_path),
        lot_col=candidate_lot_col,
        wf_col=candidate_wf_col,
        source_id_col=candidate_id_col,
        combined_id_col=candidate_combined_id_col,
        combined_id_sep=combined_id_sep,
        id_col=id_col,
    )
    base_feature_df = _with_sample_id(
        read_table(base_feature_path),
        lot_col=base_lot_col,
        wf_col=base_wf_col,
        source_id_col=base_id_col,
        combined_id_col=base_combined_id_col,
        combined_id_sep=combined_id_sep,
        id_col=id_col,
    )

    if y_target_col not in y_df.columns:
        raise ValueError(f"y file is missing target column {y_target_col!r}")
    y_keep = [id_col, y_target_col]
    if split_source_col:
        if split_source_col not in y_df.columns:
            raise ValueError(f"y file is missing split source column {split_source_col!r}")
        y_keep.append(split_source_col)
    y_std = y_df[y_keep].rename(columns={y_target_col: target_col})

    base_cols = _feature_columns(
        base_feature_df,
        id_col=id_col,
        key_cols=_key_cols(base_lot_col, base_wf_col, base_id_col, base_combined_id_col),
    )
    base_std = y_std.merge(base_feature_df[[id_col] + base_cols], on=id_col, how="inner")
    if base_std.empty:
        raise ValueError("y file and base feature file do not match on lot/wf sample_id")

    if split_source_col and split_source_col in base_std.columns:
        base_std[split_col] = base_std[split_source_col].astype(str).str.lower()
        base_std = base_std.drop(columns=[split_source_col])
    else:
        split_key_col = y_lot_col if y_lot_col in y_df.columns else id_col
        if split_key_col != id_col:
            split_lookup = y_df[[id_col, split_key_col]].drop_duplicates()
            base_std = base_std.merge(split_lookup, on=id_col, how="left")
        base_std[split_col] = _make_split(base_std[split_key_col], train_ratio=train_ratio, valid_ratio=valid_ratio, seed=split_seed)
        if split_key_col != id_col:
            base_std = base_std.drop(columns=[split_key_col], errors="ignore")

    candidate_cols = _feature_columns(
        candidate_df,
        id_col=id_col,
        key_cols=_key_cols(candidate_lot_col, candidate_wf_col, candidate_id_col, candidate_combined_id_col),
    )
    candidate_std = candidate_df[[id_col] + candidate_cols].copy()
    candidate_std, candidate_cols = _rename_overlapping_candidate_cols(
        candidate_std,
        candidate_cols=candidate_cols,
        protected_cols={id_col, target_col, split_col, *base_cols},
    )

    base_dataset_path = output_dir / "base_dataset.csv"
    candidate_features_path = output_dir / "candidate_features.csv"
    base_feature_cols_path = output_dir / "base_feature_cols.txt"
    base_std.to_csv(base_dataset_path, index=False, encoding="utf-8-sig")
    candidate_std.to_csv(candidate_features_path, index=False, encoding="utf-8-sig")
    base_feature_cols_path.write_text("\n".join(base_cols) + "\n", encoding="utf-8")

    standardized_defects = []
    for spec in defect_groups:
        defect_id = str(spec["defect_id"])
        group_path = spec["group_path"]
        lot_col = spec.get("lot_col", "lot")
        wf_col = spec.get("wf_col", "wf")
        source_id_col = spec.get("id_col")
        combined_id_col = spec.get("combined_id_col")
        group_combined_id_sep = spec.get("combined_id_sep", combined_id_sep)
        label_col = spec.get("label_col", "good_bad")
        bad_value = str(spec.get("bad_value", "bad")).strip().lower()
        good_value = str(spec.get("good_value", "good")).strip().lower()

        group_df = _with_sample_id(
            read_table(group_path),
            lot_col=lot_col,
            wf_col=wf_col,
            source_id_col=source_id_col,
            combined_id_col=combined_id_col,
            combined_id_sep=group_combined_id_sep,
            id_col=id_col,
        )
        if label_col not in group_df.columns:
            raise ValueError(f"group file {group_path} is missing label column {label_col!r}")
        labels = group_df[label_col].astype(str).str.strip().str.lower()
        bad = group_df.loc[labels == bad_value, [id_col]].drop_duplicates()
        good = group_df.loc[labels == good_value, [id_col]].drop_duplicates()
        bad_path = group_out_dir / f"{defect_id}_bad.csv"
        good_path = group_out_dir / f"{defect_id}_good.csv"
        bad.to_csv(bad_path, index=False, encoding="utf-8-sig")
        good.to_csv(good_path, index=False, encoding="utf-8-sig")
        standardized_defects.append(
            {
                "defect_id": defect_id,
                "bad_group_path": bad_path,
                "good_group_path": good_path,
            }
        )

    return {
        "base_dataset": base_dataset_path,
        "candidate_features": candidate_features_path,
        "base_feature_cols": base_feature_cols_path,
        "defects": standardized_defects,
    }


def _with_sample_id(
    df: pd.DataFrame,
    *,
    lot_col: str | None,
    wf_col: str | None,
    source_id_col: str | None = None,
    combined_id_col: str | None = None,
    combined_id_sep: str = "_",
    id_col: str,
) -> pd.DataFrame:
    if source_id_col and combined_id_col:
        raise ValueError("source_id_col and combined_id_col cannot both be set")
    if combined_id_col:
        return _with_split_combined_id(
            df,
            combined_id_col=combined_id_col,
            combined_id_sep=combined_id_sep,
            lot_col=lot_col,
            wf_col=wf_col,
            id_col=id_col,
        )
    if source_id_col:
        if source_id_col not in df.columns:
            raise ValueError(f"input file is missing id column {source_id_col!r}")
        result = df.copy()
        result[id_col] = result[source_id_col].astype(str).str.strip()
        return result
    if not lot_col or not wf_col:
        raise ValueError("input file must provide either source_id_col or both lot_col and wf_col")
    missing = [col for col in (lot_col, wf_col) if col not in df.columns]
    if missing:
        raise ValueError(f"input file is missing lot/wf columns: {missing}")
    result = df.copy()
    result[wf_col] = _coerce_wf_id(result[wf_col], wf_col=wf_col)
    result[id_col] = result[lot_col].astype(str).str.strip() + "_" + result[wf_col].astype(str).str.strip()
    return result


def _with_split_combined_id(
    df: pd.DataFrame,
    *,
    combined_id_col: str,
    combined_id_sep: str,
    lot_col: str | None,
    wf_col: str | None,
    id_col: str,
) -> pd.DataFrame:
    if combined_id_col not in df.columns:
        raise ValueError(f"input file is missing combined id column {combined_id_col!r}")
    if not lot_col or not wf_col:
        raise ValueError("combined id parsing requires destination lot_col and wf_col names")
    if not combined_id_sep:
        raise ValueError("combined_id_sep must be a non-empty string")

    result = df.copy()
    values = result[combined_id_col].astype(str).str.strip()
    split_values = values.str.rsplit(combined_id_sep, n=1, expand=True)
    if split_values.shape[1] != 2:
        raise ValueError(f"combined id column {combined_id_col!r} must contain separator {combined_id_sep!r}")
    if split_values[0].eq("").any() or split_values[1].eq("").any():
        raise ValueError(f"combined id column {combined_id_col!r} contains empty lot or wf values")

    result[lot_col] = split_values[0].astype(str).str.strip()
    result[wf_col] = _coerce_wf_id(split_values[1], wf_col=wf_col)
    result[id_col] = result[lot_col].astype(str).str.strip() + "_" + result[wf_col].astype(str).str.strip()
    return result


def _coerce_wf_id(values: pd.Series, *, wf_col: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        bad_values = values[numeric.isna()].astype(str).head(5).tolist()
        raise ValueError(f"wf column {wf_col!r} contains non-integer values: {bad_values}")
    if not np.all(np.isclose(numeric.astype(float), np.round(numeric.astype(float)))):
        bad_values = values[~np.isclose(numeric.astype(float), np.round(numeric.astype(float)))].astype(str).head(5).tolist()
        raise ValueError(f"wf column {wf_col!r} contains non-integer values: {bad_values}")
    return numeric.astype("int64")


def _feature_columns(df: pd.DataFrame, *, id_col: str, key_cols: set[str]) -> list[str]:
    excluded = {id_col, *key_cols}
    return [col for col in df.columns if col not in excluded]


def _key_cols(
    lot_col: str | None,
    wf_col: str | None,
    source_id_col: str | None = None,
    combined_id_col: str | None = None,
) -> set[str]:
    return {col for col in (lot_col, wf_col, source_id_col, combined_id_col) if col}


def _rename_overlapping_candidate_cols(
    candidate_df: pd.DataFrame,
    *,
    candidate_cols: list[str],
    protected_cols: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    rename_map: dict[str, str] = {}
    used_cols = set(candidate_df.columns)
    for col in candidate_cols:
        if col not in protected_cols:
            continue
        new_col = f"candidate__{col}"
        counter = 2
        while new_col in used_cols or new_col in protected_cols:
            new_col = f"candidate__{counter}__{col}"
            counter += 1
        rename_map[col] = new_col
        used_cols.add(new_col)

    if not rename_map:
        return candidate_df, candidate_cols

    renamed = candidate_df.rename(columns=rename_map)
    renamed_cols = [rename_map.get(col, col) for col in candidate_cols]
    return renamed, renamed_cols


def _make_split(lots: pd.Series, *, train_ratio: float, valid_ratio: float, seed: int) -> pd.Series:
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio and valid_ratio must be positive and leave a non-empty test split")
    unique_lots = np.array(sorted(lots.astype(str).dropna().unique()))
    if len(unique_lots) < 3:
        raise ValueError("automatic train/valid/test split requires at least 3 unique lots")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_lots)
    n_train = max(1, int(round(len(unique_lots) * train_ratio)))
    n_valid = max(1, int(round(len(unique_lots) * valid_ratio)))
    n_train = min(n_train, len(unique_lots) - 2)
    n_valid = min(n_valid, len(unique_lots) - n_train - 1)
    train_lots = set(unique_lots[:n_train])
    valid_lots = set(unique_lots[n_train : n_train + n_valid])
    values = lots.astype(str)
    return pd.Series(
        np.where(values.isin(train_lots), "train", np.where(values.isin(valid_lots), "valid", "test")),
        index=lots.index,
    )
