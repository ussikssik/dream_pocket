from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_booster import (  # noqa: E402
    AlternativeFeatureSelector,
    AlternativeSelectorConfig,
    CatBoostFeatureSetEvalConfig,
    CatBoostProbeConfig,
    evaluate_catboost_global_feature_sets,
    save_toy_truth_evaluation,
    summarize_method_overlap,
)
from generate_wide_toy_semiconductor_dataset import (  # noqa: E402
    DEFAULT_BOOSTER_BAD_ROWS,
    DEFAULT_BOOSTER_GOOD_ROWS,
    DEFAULT_FULL_ROWS_PER_ORDER,
    DEFAULT_OUTPUT_DIR as DEFAULT_DATA_DIR,
)
from run_feature_booster_on_wide_toyset import ensure_dataset, run_booster  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_DIR = PROJECT_ROOT / "outputs" / "wide_toy_selector_comparison"

DEFAULT_EXCLUDE_COLS = (
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
)


def run_alternative_selectors(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_COMPARISON_DIR,
    order_list: list[int] | None = None,
    top_k: int = 10,
    methods: tuple[str, ...] = (
        "nonparametric_random",
        "distance",
        "catboost_shap_gap",
        "stability_consensus",
    ),
    random_state: int = 42,
    stability_rounds: int = 8,
) -> pd.DataFrame:
    order_list = order_list or [1, 2, 3]

    def load_order(order_id: int) -> pd.DataFrame:
        return pd.read_csv(data_dir / f"wide_order_{order_id:03d}.csv")

    config = AlternativeSelectorConfig(
        label_col="target_bad_a",
        positive_label=1,
        group_cols=("lot_id", "product_id"),
        sample_id_cols=("sample_id", "wafer_id"),
        exclude_cols=DEFAULT_EXCLUDE_COLS,
        output_dir=output_dir,
        methods=methods,
        random_state=random_state,
        stability_rounds=stability_rounds,
        catboost=CatBoostProbeConfig(
            enabled=True,
            max_features_per_order=500,
            iterations=220,
            depth=4,
            learning_rate=0.05,
            verbose=False,
        ),
    )
    selector = AlternativeFeatureSelector(load_order, config)
    return selector.run(order_list=order_list, top_k_per_order=top_k, methods=methods)


def run_comparison(
    data_dir: Path,
    output_dir: Path,
    order_list: list[int],
    top_k: int,
    shap_top_n: int,
    methods: tuple[str, ...],
    include_evidence_booster: bool,
    run_catboost_eval: bool,
    stability_rounds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pieces = []

    if include_evidence_booster:
        evidence = run_booster(
            data_dir=data_dir,
            output_dir=output_dir / "evidence_booster",
            order_list=order_list,
            top_k=top_k,
            bootstrap_rounds=3,
            catboost_enabled=False,
        )
        pieces.append(_normalize_evidence_booster(evidence))

    alternatives = run_alternative_selectors(
        data_dir=data_dir,
        output_dir=output_dir / "alternative_selectors",
        order_list=order_list,
        top_k=top_k,
        methods=methods,
        stability_rounds=stability_rounds,
    )
    pieces.append(alternatives)

    comparison = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    comparison.to_csv(output_dir / "combined_selector_comparison.csv", index=False, encoding="utf-8-sig")

    overlap = summarize_method_overlap(comparison)
    overlap.to_csv(output_dir / "method_overlap_jaccard.csv", index=False, encoding="utf-8-sig")
    metadata_path = data_dir / "wide_feature_metadata.csv"
    if metadata_path.exists() and not comparison.empty:
        save_toy_truth_evaluation(
            comparison,
            metadata_path=metadata_path,
            output_dir=output_dir,
            prefix="selector_comparison",
            top_k=top_k,
            rank_col="rank",
        )

    shap_delta = pd.DataFrame()
    model_metrics = pd.DataFrame()
    if run_catboost_eval and not comparison.empty:
        shap_delta, model_metrics = run_post_selection_catboost_eval(
            data_dir=data_dir,
            output_dir=output_dir / "catboost_post_eval",
            selection_result=comparison,
            order_list=order_list,
            selected_features_per_order=top_k,
            shap_top_n=shap_top_n,
        )
        if metadata_path.exists() and not shap_delta.empty:
            _, truth_summary = save_toy_truth_evaluation(
                shap_delta,
                metadata_path=metadata_path,
                output_dir=output_dir / "catboost_post_eval",
                prefix="catboost_global_shap_delta",
                top_k=shap_top_n,
                rank_col="shap_rank",
            )
            scoreboard = build_global_method_scoreboard(truth_summary)
            scoreboard.to_csv(
                output_dir / "catboost_post_eval" / "catboost_global_method_scoreboard.csv",
                index=False,
                encoding="utf-8-sig",
            )
    return comparison, overlap, shap_delta, model_metrics


def build_global_method_scoreboard(truth_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank methods by synthetic-truth quality of the final global SHAP top-N set."""

    if truth_summary.empty:
        return truth_summary.copy()
    scoreboard = truth_summary.copy()
    sort_cols = [
        col
        for col in [
            "toy_truth_score",
            "target_defect_a_precision",
            "target_defect_a_hit_count",
            "nonlinear_a_hit_count",
            "sparse_a_hit_count",
        ]
        if col in scoreboard.columns
    ]
    if sort_cols:
        scoreboard = scoreboard.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
        scoreboard.insert(0, "score_rank", range(1, len(scoreboard) + 1))
    return scoreboard


def run_post_selection_catboost_eval(
    data_dir: Path,
    output_dir: Path,
    selection_result: pd.DataFrame,
    order_list: list[int],
    selected_features_per_order: int = 20,
    shap_top_n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def load_train_order(order_id: int) -> pd.DataFrame:
        return pd.read_csv(data_dir / f"wide_order_{int(order_id):03d}.csv")

    def load_full_pool_order(order_id: int) -> pd.DataFrame:
        return pd.read_csv(data_dir / f"wide_order_{int(order_id):03d}_full_pool.csv")

    config = CatBoostFeatureSetEvalConfig(
        label_col="target_bad_a",
        positive_label=1,
        y_col="eds_bin_a_wf_mean",
        role_col="booster_sample_role",
        selected_features_per_order=selected_features_per_order,
        shap_top_n=shap_top_n,
        output_dir=output_dir,
        plot_enabled=True,
        iterations=300,
        depth=4,
        learning_rate=0.05,
        verbose=False,
    )
    return evaluate_catboost_global_feature_sets(
        selection_result=selection_result,
        train_loader=load_train_order,
        order_list=order_list,
        full_pool_loader=load_full_pool_order,
        config=config,
    )


def _normalize_evidence_booster(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return evidence
    result = evidence.copy()
    result.insert(1, "method", "evidence_booster")
    result = result.rename(columns={"final_rank": "rank", "final_score": "score"})
    result["selection_reason"] = result.get("evidence_reason", "")
    result["selection_warning"] = [
        "; ".join(part for part in parts if part)
        for parts in zip(
            _text_col(result, "quality_warning"),
            _text_col(result, "confounding_warning"),
            _text_col(result, "xai_warning"),
        )
    ]
    return result


def _text_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index)
    return df[col].fillna("").astype(str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare feature selector modules on the wide toy dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--rows", type=int, default=DEFAULT_FULL_ROWS_PER_ORDER)
    parser.add_argument("--features-per-order", type=int, default=10000)
    parser.add_argument("--booster-good-rows", type=int, default=DEFAULT_BOOSTER_GOOD_ROWS)
    parser.add_argument("--booster-bad-rows", type=int, default=DEFAULT_BOOSTER_BAD_ROWS)
    parser.add_argument("--top-k", type=int, default=15, help="Features selected per order/method before global CatBoost post-evaluation.")
    parser.add_argument("--shap-top-n", type=int, default=15, help="Top global SHAP-delta features to report and plot per method.")
    parser.add_argument("--stability-rounds", type=int, default=8)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["nonparametric_random", "distance", "catboost_shap_gap", "stability_consensus"],
        choices=["nonparametric_random", "distance", "catboost_shap_gap", "stability_consensus"],
    )
    parser.add_argument("--skip-evidence-booster", action="store_true")
    parser.add_argument("--skip-catboost-eval", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dataset(args)
    comparison, overlap, shap_delta, model_metrics = run_comparison(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        order_list=args.orders,
        top_k=args.top_k,
        shap_top_n=args.shap_top_n,
        methods=tuple(args.methods),
        include_evidence_booster=not args.skip_evidence_booster,
        run_catboost_eval=not args.skip_catboost_eval,
        stability_rounds=args.stability_rounds,
    )

    display_cols = ["order_id", "method", "rank", "feature_name", "score", "selection_reason", "selection_warning"]
    existing = [col for col in display_cols if col in comparison.columns]
    print(comparison[existing].to_string(index=False))
    print(f"\nSaved comparison to {args.output_dir / 'combined_selector_comparison.csv'}")
    print(f"Saved overlap matrix to {args.output_dir / 'method_overlap_jaccard.csv'}")
    if not shap_delta.empty:
        top = shap_delta[shap_delta["shap_rank"] <= args.shap_top_n]
        display_shap_cols = [
            "order_id",
            "method",
            "shap_rank",
            "feature_name",
            "shap_delta_abs",
            "model_auc_train",
            "evaluation_warning",
            "plot_path",
        ]
        existing_shap = [col for col in display_shap_cols if col in top.columns]
        print("\nCatBoost SHAP delta top features:")
        print(top[existing_shap].to_string(index=False))
        print(f"\nSaved CatBoost SHAP delta to {args.output_dir / 'catboost_post_eval' / 'catboost_global_shap_delta_top_features.csv'}")
    elif not args.skip_catboost_eval:
        print("\nCatBoost post-evaluation produced no SHAP rows. Check catboost_post_eval CSV warnings.")

    if not model_metrics.empty:
        print("\nCatBoost model metrics:")
        print(model_metrics.to_string(index=False))
    scoreboard_path = args.output_dir / "catboost_post_eval" / "catboost_global_method_scoreboard.csv"
    if scoreboard_path.exists():
        scoreboard = pd.read_csv(scoreboard_path)
        if not scoreboard.empty:
            display_score_cols = [
                "score_rank",
                "method",
                "evaluated_top_k",
                "target_defect_a_precision",
                "target_defect_a_hit_count",
                "nonlinear_a_hit_count",
                "sparse_a_hit_count",
                "other_defect_hit_count",
                "tool_confounded_hit_count",
                "noise_hit_count",
                "toy_truth_score",
            ]
            existing_score_cols = [col for col in display_score_cols if col in scoreboard.columns]
            print("\nGlobal method scoreboard from final SHAP top-N:")
            print(scoreboard[existing_score_cols].to_string(index=False))
    if not overlap.empty:
        print("\nMethod overlap:")
        print(overlap.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
