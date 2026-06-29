from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_booster import (  # noqa: E402
    BoosterConfig,
    CatBoostProbeConfig,
    DefectAFeatureEvidenceBooster,
    RankingConfig,
    RedundancyConfig,
    StatisticalConfig,
)
from generate_wide_toy_semiconductor_dataset import (  # noqa: E402
    DEFAULT_BOOSTER_BAD_ROWS,
    DEFAULT_BOOSTER_GOOD_ROWS,
    DEFAULT_FULL_ROWS_PER_ORDER,
    DEFAULT_OUTPUT_DIR,
    DATASET_VERSION,
    generate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "wide_toy_semiconductor_booster"


def load_feature_metadata(data_dir: Path) -> dict[str, dict[str, object]]:
    metadata_path = data_dir / "wide_feature_metadata.csv"
    if not metadata_path.exists():
        return {}
    metadata = pd.read_csv(metadata_path)
    return metadata.set_index("feature_name").to_dict(orient="index")


def run_booster(
    data_dir: Path = DEFAULT_OUTPUT_DIR,
    output_dir: Path = DEFAULT_REPORT_DIR,
    order_list: list[int] | None = None,
    top_k: int = 10,
    bootstrap_rounds: int = 5,
    catboost_enabled: bool = False,
) -> pd.DataFrame:
    order_list = order_list or [1, 2, 3]

    def load_order(order_id: int) -> pd.DataFrame:
        return pd.read_csv(data_dir / f"wide_order_{order_id:03d}.csv")

    config = BoosterConfig(
        label_col="target_bad_a",
        positive_label=1,
        group_cols=("lot_id", "product_id"),
        sample_id_cols=("sample_id", "wafer_id"),
        exclude_cols=(
            "order_id",
            "process_run_seq",
            "process_elapsed_ratio",
            "run_block_id",
            "booster_sample_role",
            "is_booster_sample",
            "eds_bin_no_wf_mean",
            "eds_bin_a_wf_mean",
            "eds_bin_b_wf_mean",
            "eds_bin_c_wf_mean",
            "eds_yield_wf_mean",
            "sim_true_defect_a_flag",
            "sim_true_defect_b_flag",
            "sim_true_defect_c_flag",
            "sim_true_defect_d_flag",
        ),
        feature_metadata=load_feature_metadata(data_dir),
        output_dir=output_dir,
        statistical=StatisticalConfig(bootstrap_rounds=bootstrap_rounds, bootstrap_sample_frac=0.70),
        ranking=RankingConfig(preliminary_top_n=500, final_top_k_per_order=top_k),
        redundancy=RedundancyConfig(max_features=250),
        catboost=CatBoostProbeConfig(enabled=catboost_enabled, max_features_per_order=300),
    )

    booster = DefectAFeatureEvidenceBooster(load_order, config)
    return booster.run(order_list=order_list, top_k_per_order=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the evidence booster on the wide toy dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_FULL_ROWS_PER_ORDER,
        help="Full wafer rows per order before Good/Bad sampling.",
    )
    parser.add_argument("--features-per-order", type=int, default=10000)
    parser.add_argument("--booster-good-rows", type=int, default=DEFAULT_BOOSTER_GOOD_ROWS)
    parser.add_argument("--booster-bad-rows", type=int, default=DEFAULT_BOOSTER_BAD_ROWS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-rounds", type=int, default=5)
    parser.add_argument("--catboost", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def ensure_dataset(args: argparse.Namespace) -> None:
    missing = [order_id for order_id in args.orders if not (args.data_dir / f"wide_order_{order_id:03d}.csv").exists()]
    needs_regenerate = args.regenerate or bool(missing) or not _existing_dataset_matches(args)
    if needs_regenerate:
        max_order = max(args.orders)
        summary = generate_dataset(
            output_dir=args.data_dir,
            orders=max_order,
            rows=args.rows,
            features_per_order=args.features_per_order,
            booster_good_rows=args.booster_good_rows,
            booster_bad_rows=args.booster_bad_rows,
            overwrite=True,
        )
        print("Generated wide toy dataset:")
        print(summary.to_string(index=False))


def _existing_dataset_matches(args: argparse.Namespace) -> bool:
    summary_path = args.data_dir / "wide_target_summary_by_order.csv"
    if not summary_path.exists():
        return False

    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return False

    required_cols = {
        "order_id",
        "full_rows",
        "candidate_feature_count",
        "booster_good_count",
        "booster_bad_count",
        "dataset_version",
    }
    if not required_cols.issubset(summary.columns):
        return False

    for order_id in args.orders:
        row = summary[summary["order_id"] == order_id]
        if row.empty:
            return False
        values = row.iloc[0]
        if str(values["dataset_version"]) != DATASET_VERSION:
            return False
        if int(values["full_rows"]) != int(args.rows):
            return False
        if int(values["candidate_feature_count"]) != int(args.features_per_order):
            return False
        if int(values["booster_good_count"]) != int(args.booster_good_rows):
            return False
        if int(values["booster_bad_count"]) != int(args.booster_bad_rows):
            return False
    return True


def main() -> int:
    args = parse_args()
    ensure_dataset(args)

    result = run_booster(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        order_list=args.orders,
        top_k=args.top_k,
        bootstrap_rounds=args.bootstrap_rounds,
        catboost_enabled=args.catboost,
    )

    display_cols = [
        "order_id",
        "final_rank",
        "feature_name",
        "final_score",
        "presence_type",
        "direction",
        "evidence_reason",
    ]
    print(result[display_cols].to_string(index=False))
    print(f"\nSaved reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
