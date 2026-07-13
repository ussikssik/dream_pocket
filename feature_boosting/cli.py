from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .baseline_model import add_baseline_predictions, metrics_by_split, train_baseline_model
from .config import load_config
from .data_loader import (
    align_base_and_candidates,
    candidate_feature_cols,
    load_base_dataset,
    load_base_feature_cols,
    load_candidate_features,
    load_group_ids,
)
from .final_model import evaluate_model_by_groups, predict_final, train_final_model
from .reporting import (
    baseline_residual_summary,
    copy_config,
    final_metric_summary,
    iteration_residual_summary,
    plot_candidate_loss_ranking,
    plot_final_feature_set_summary,
    plot_final_metric_comparison,
    plot_residual_curve,
    plot_round_residual_points,
    prepare_output_dir,
    setup_logger,
    write_csv,
)
from .residual_boosting import ResidualFeatureBooster, ResidualFeatureBoosterConfig
from .shap_analysis import compute_shap_summary
from .splitter import split_frame
from .validation import profile_candidate_features, validate_defect_groups, validate_input_columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run residual-based feature boosting PoC.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML/JSON config.")
    args = parser.parse_args(argv)
    return run_experiment(Path(args.config))


def run_experiment(config_path: Path) -> int:
    config = load_config(config_path)
    output_dir = prepare_output_dir(config.paths.output_dir, config.run_id)
    logger = setup_logger(output_dir)
    copy_config(config_path, output_dir)

    paths = config.paths
    cols = config.columns

    logger.info("[LOAD] loading base dataset")
    base_df = load_base_dataset(_resolve_path(paths.base_dataset))
    logger.info("[LOAD] base dataset loaded: n_rows=%s, n_cols=%s", len(base_df), len(base_df.columns))

    logger.info("[LOAD] loading candidate features")
    candidate_df = load_candidate_features(_resolve_path(paths.candidate_features))
    candidate_cols = candidate_feature_cols(candidate_df, cols.id_col)
    logger.info("[LOAD] candidate features loaded: n_features=%s", len(candidate_cols))

    base_feature_cols = load_base_feature_cols(_resolve_path(paths.base_feature_cols))
    validate_input_columns(
        base_df,
        candidate_df,
        base_feature_cols,
        id_col=cols.id_col,
        target_col=cols.target_col,
        split_col=cols.split_col,
    )
    all_df = align_base_and_candidates(base_df, candidate_df, cols.id_col)
    train_df, valid_df, test_df = split_frame(all_df, cols.split_col)

    defect_groups: dict[str, dict[str, set[str]]] = {}
    for defect in config.defects:
        bad_ids = load_group_ids(_resolve_path(defect.bad_group_path), cols.id_col)
        good_ids = load_group_ids(_resolve_path(defect.good_group_path), cols.id_col)
        warnings = validate_defect_groups(
            defect.defect_id,
            bad_ids,
            good_ids,
            all_df,
            id_col=cols.id_col,
            split_col=cols.split_col,
            min_valid_bad_samples=config.boosting.min_valid_bad_samples,
        )
        logger.info("[DEFECT %s] bad n=%s, good n=%s%s", defect.defect_id, len(bad_ids), len(good_ids), _warn_suffix(warnings))
        defect_groups[defect.defect_id] = {"bad": bad_ids, "good": good_ids, "warnings": set(warnings)}

    logger.info("[BASELINE] training started")
    baseline_model = train_baseline_model(train_df, valid_df, base_feature_cols, cols.target_col, config.baseline_model)
    all_df = add_baseline_predictions(
        baseline_model,
        all_df,
        feature_cols=base_feature_cols,
        target_col=cols.target_col,
    )
    train_df, valid_df, test_df = split_frame(all_df, cols.split_col)
    baseline_metrics = metrics_by_split(all_df, target_col=cols.target_col, pred_col="baseline_pred", split_col=cols.split_col)
    write_csv(baseline_metrics, output_dir / "baseline_metrics.csv")
    baseline_summary = baseline_residual_summary(
        all_df,
        target_col=cols.target_col,
        pred_col="baseline_pred",
        residual_col="baseline_residual",
        split_col=cols.split_col,
        id_col=cols.id_col,
        defects=defect_groups,
    )
    write_csv(
        baseline_summary,
        output_dir / "baseline_residual_summary.csv",
    )
    valid_rmse = baseline_metrics.loc[baseline_metrics["split"].astype(str) == "valid", "rmse"]
    logger.info("[BASELINE] valid RMSE: %s", round(float(valid_rmse.iloc[0]), 6) if not valid_rmse.empty else "nan")
    _save_model(baseline_model, output_dir / "models" / "baseline_model.cbm")

    quality_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    iteration_frames: list[pd.DataFrame] = []
    test_prediction_frames: list[pd.DataFrame] = []
    booster = ResidualFeatureBooster(
        ResidualFeatureBoosterConfig(
            residual_model_params=config.residual_model,
            n_rounds=config.boosting.n_rounds,
            select_per_round=config.boosting.select_per_round,
            main_metric=config.boosting.main_metric,
            min_improvement=config.boosting.min_improvement,
            selection_mode=config.boosting.selection_mode,
            selection_metric=config.boosting.selection_metric,
            selection_threshold=config.boosting.selection_threshold,
            selection_direction=config.boosting.selection_direction,
            max_select_per_round=config.boosting.max_select_per_round,
            use_test_for_selection=config.boosting.use_test_for_selection,
            min_valid_bad_samples=config.boosting.min_valid_bad_samples,
            overfit_guard_enabled=config.boosting.overfit_guard_enabled,
            overfit_guard_metric_scope=config.boosting.overfit_guard_metric_scope,
            overfit_guard_min_valid_rmse_reduction=config.boosting.overfit_guard_min_valid_rmse_reduction,
            overfit_guard_max_valid_after_over_baseline=config.boosting.overfit_guard_max_valid_after_over_baseline,
            overfit_guard_max_valid_train_gap=config.boosting.overfit_guard_max_valid_train_gap,
            overfit_guard_use_test=config.boosting.overfit_guard_use_test,
            overfit_guard_max_test_after_over_baseline=config.boosting.overfit_guard_max_test_after_over_baseline,
            show_progress=config.boosting.show_progress,
            progress_every=config.boosting.progress_every,
        )
    )

    sequential_predictions = {
        "train": train_df["baseline_pred"].to_numpy(dtype=float).copy(),
        "valid": valid_df["baseline_pred"].to_numpy(dtype=float).copy(),
        "test": test_df["baseline_pred"].to_numpy(dtype=float).copy(),
    }
    global_selected_features: set[str] = set()
    global_iter_offset = 0

    for defect in config.defects:
        groups = defect_groups[defect.defect_id]
        quality = profile_candidate_features(
            all_df,
            candidate_cols,
            id_col=cols.id_col,
            split_col=cols.split_col,
            bad_ids=groups["bad"],
            good_ids=groups["good"],
            config=config.feature_filter,
            protected_cols={cols.id_col, cols.target_col, cols.split_col, *base_feature_cols},
        )
        quality.insert(0, "defect_id", defect.defect_id)
        quality_frames.append(quality)
        if any(str(item).startswith("low_valid_bad_samples") for item in groups.get("warnings", set())):
            logger.info("[BOOST %s] skipped: low valid bad sample count", defect.defect_id)
            continue
        logger.info("[BOOST %s round 1] scoring candidates: %s", defect.defect_id, len(candidate_cols))
        result = booster.run_for_defect(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            candidate_cols=candidate_cols,
            target_col=cols.target_col,
            id_col=cols.id_col,
            baseline_pred_col="baseline_pred",
            defect_id=defect.defect_id,
            bad_sample_ids=groups["bad"],
            good_sample_ids=groups["good"],
            quality_summary=quality,
            initial_predictions=sequential_predictions,
            previously_selected_features=global_selected_features,
            global_iter_start=global_iter_offset,
            base_feature_count=len(base_feature_cols),
            output_dir=output_dir / "rankings",
        )
        if result.final_predictions:
            sequential_predictions = {
                split: values.copy() for split, values in result.final_predictions.items()
            }
        if not result.selected_features.empty:
            global_selected_features.update(result.selected_features["feature_name"].dropna().astype(str))
        global_iter_offset += len(result.rankings)
        if not result.iteration_summary.empty:
            iteration_frames.append(result.iteration_summary)
        if not result.test_predictions.empty:
            test_prediction_frames.append(result.test_predictions)
        for ranking_df in result.rankings:
            if ranking_df.empty or "round" not in ranking_df.columns:
                continue
            round_no = int(ranking_df["round"].max())
            plot_candidate_loss_ranking(
                ranking_df,
                output_path=output_dir / "plots" / f"{defect.defect_id}_round_{round_no}_candidate_loss.png",
                global_metric_col="valid_global_rmse_reduction_over_before",
                bad_metric_col="valid_bad_rmse_reduction_over_before",
                title_prefix=f"{defect.defect_id} round {round_no}",
            )
        if not result.selected_features.empty:
            selected_frames.append(result.selected_features)
            first = result.selected_features.iloc[0]
            logger.info(
                "[BOOST %s round %s] selected feature: %s, valid_bad_rmse_reduction=%s",
                defect.defect_id,
                first["round"],
                first["feature_name"],
                round(float(first["valid_bad_rmse_reduction"]), 6),
            )
        if not result.residual_curve.empty:
            curve_frames.append(result.residual_curve)

    quality_summary = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()
    selected_features = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    residual_curve = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    iteration_summary = pd.concat(iteration_frames, ignore_index=True) if iteration_frames else pd.DataFrame()
    test_predictions = pd.concat(test_prediction_frames, ignore_index=True) if test_prediction_frames else pd.DataFrame()
    write_csv(quality_summary, output_dir / "candidate_quality_summary.csv")
    write_csv(selected_features, output_dir / "selected_features.csv")
    write_csv(residual_curve, output_dir / "residual_reduction_curve.csv")
    write_csv(iteration_summary, output_dir / "boosting_iteration_audit.csv")
    write_csv(test_predictions, output_dir / "boosting_test_predictions_by_iteration.csv")
    plot_residual_curve(residual_curve, output_dir)
    round_mean_residual = iteration_residual_summary(iteration_summary, group="bad")
    write_csv(round_mean_residual, output_dir / "round_mean_residual_summary.csv")
    plot_round_residual_points(round_mean_residual, output_path=output_dir / "plots" / "round_mean_abs_residual_points.png")

    selected_cols = _unique_selected_features(selected_features, candidate_cols)
    final_feature_cols = base_feature_cols + selected_cols
    final_feature_summary = pd.DataFrame(
        [
            {"feature_type": "base", "count": len(base_feature_cols)},
            {"feature_type": "selected", "count": len(selected_cols)},
        ]
    )
    write_csv(final_feature_summary, output_dir / "final_feature_set_summary.csv")
    plot_final_feature_set_summary(final_feature_summary, output_path=output_dir / "plots" / "final_feature_set_summary.png")
    logger.info("[FINAL] training final model with n_selected=%s", len(selected_cols))
    final_params = config.final_model or config.baseline_model
    final_model = train_final_model(train_df, valid_df, feature_cols=final_feature_cols, target_col=cols.target_col, catboost_params=final_params)
    all_df["final_pred"] = predict_final(final_model, all_df, final_feature_cols)
    _save_model(final_model, output_dir / "models" / "final_model.cbm")
    final_metrics = pd.concat(
        [
            evaluate_model_by_groups(
                all_df,
                model_name="baseline",
                pred_col="baseline_pred",
                target_col=cols.target_col,
                split_col=cols.split_col,
                id_col=cols.id_col,
                defects=defect_groups,
            ),
            evaluate_model_by_groups(
                all_df,
                model_name="final",
                pred_col="final_pred",
                target_col=cols.target_col,
                split_col=cols.split_col,
                id_col=cols.id_col,
                defects=defect_groups,
            ),
        ],
        ignore_index=True,
    )
    write_csv(final_metrics, output_dir / "final_model_metrics.csv")
    final_metric_summary_df = final_metric_summary(final_metrics)
    write_csv(final_metric_summary_df, output_dir / "final_model_metric_summary.csv")
    plot_final_metric_comparison(final_metric_summary_df, output_path=output_dir / "plots" / "final_model_metric_comparison.png")

    if config.shap.enabled:
        logger.info("[SHAP] calculating SHAP summary")
        shap_summary = compute_shap_summary(
            final_model,
            all_df[all_df[cols.split_col].astype(str) == "test"],
            feature_cols=final_feature_cols,
            id_col=cols.id_col,
            defects=defect_groups,
            selected_features=selected_features,
            max_samples=config.shap.max_samples,
        )
    else:
        shap_summary = pd.DataFrame([{"status": "disabled"}])
    write_csv(shap_summary, output_dir / "shap_summary.csv")
    logger.info("[DONE] outputs saved to %s", output_dir)
    return 0


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_model"):
        try:
            model.save_model(path)
            return
        except TypeError:
            model.save_model(str(path))


def _unique_selected_features(selected_features: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    if selected_features.empty or "feature_name" not in selected_features.columns:
        return []
    available = set(candidate_cols)
    result = []
    for feature in selected_features["feature_name"].dropna().astype(str):
        if feature in available and feature not in result:
            result.append(feature)
    return result


def _warn_suffix(warnings: list[str]) -> str:
    return f", warnings={warnings}" if warnings else ""


if __name__ == "__main__":
    raise SystemExit(main())
