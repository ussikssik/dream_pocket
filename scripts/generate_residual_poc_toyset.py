from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a toy dataset for the residual feature boosting PoC.")
    parser.add_argument("--output-dir", default="data/residual_poc_toyset", help="Directory to write toy CSV files.")
    parser.add_argument("--rows", type=int, default=14000, help="Number of wafer/sample rows.")
    parser.add_argument(
        "--candidate-features",
        type=int,
        default=5000,
        help="Total number of candidate features, including the two planted hidden defect features.",
    )
    parser.add_argument(
        "--noise-features",
        type=int,
        default=None,
        help="Backward-compatible override for the number of irrelevant candidate features.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    group_dir = output_dir / "groups"
    group_dir.mkdir(parents=True, exist_ok=True)

    noise_features = args.noise_features
    if noise_features is None:
        noise_features = max(0, args.candidate_features - 2)
    base_df, candidate_df, groups = make_toyset(args.rows, noise_features, args.seed)
    base_df.to_csv(output_dir / "base_dataset.csv", index=False, encoding="utf-8-sig")
    candidate_df.to_csv(output_dir / "candidate_features.csv", index=False, encoding="utf-8-sig")
    (output_dir / "base_feature_cols.txt").write_text("base_temp\nbase_pressure\n", encoding="utf-8")

    for name, frame in groups.items():
        frame.to_csv(group_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    print(f"toyset written to {output_dir.resolve()}")
    print(f"base rows={len(base_df)}, candidate features={candidate_df.shape[1] - 1}")
    return 0


def make_toyset(n_rows: int = 14000, n_noise_features: int = 4998, seed: int = 42):
    rng = np.random.default_rng(seed)
    sample_id = np.array([f"WF_{idx:04d}" for idx in range(n_rows)])

    n_train = int(round(n_rows * 0.6))
    n_valid = int(round(n_rows * 0.2))
    n_test = n_rows - n_train - n_valid
    split = np.array(["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test)
    rng.shuffle(split)

    base_temp = rng.normal(0, 1, n_rows)
    base_pressure = rng.normal(0, 1, n_rows)
    hidden_defect_1 = rng.normal(0, 1, n_rows)
    hidden_defect_2 = rng.normal(0, 1, n_rows)
    noise_candidates = {f"cand_noise_{i:04d}": rng.normal(0, 1, n_rows) for i in range(n_noise_features)}

    y = (
        80
        + 5.0 * base_temp
        - 3.0 * base_pressure
        + 6.0 * hidden_defect_1
        - 4.0 * hidden_defect_2
        + rng.normal(0, 0.3, n_rows)
    )

    base_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "yield": y,
            "split": split,
            "base_temp": base_temp,
            "base_pressure": base_pressure,
        }
    )
    candidate_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "hidden_defect_1": hidden_defect_1,
            "hidden_defect_2": hidden_defect_2,
            **noise_candidates,
        }
    )

    bad1 = base_df.loc[hidden_defect_1 >= np.quantile(hidden_defect_1, 0.75), ["sample_id"]]
    good1 = base_df.loc[hidden_defect_1 < np.quantile(hidden_defect_1, 0.50), ["sample_id"]]
    bad2 = base_df.loc[hidden_defect_2 <= np.quantile(hidden_defect_2, 0.25), ["sample_id"]]
    good2 = base_df.loc[hidden_defect_2 > np.quantile(hidden_defect_2, 0.50), ["sample_id"]]
    bad3 = base_df.sample(min(120, max(1, n_rows // 5)), random_state=seed)[["sample_id"]]
    good3_pool = base_df.drop(bad3.index)
    good3 = good3_pool.sample(min(180, max(1, len(good3_pool) // 3)), random_state=seed + 1)[["sample_id"]]

    groups = {
        "defect_1_bad": bad1,
        "defect_1_good": good1,
        "defect_2_bad": bad2,
        "defect_2_good": good2,
        "defect_3_bad": bad3,
        "defect_3_good": good3,
    }
    return base_df, candidate_df, groups


if __name__ == "__main__":
    raise SystemExit(main())
