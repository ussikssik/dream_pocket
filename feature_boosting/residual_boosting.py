from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import mae, residual_reduction_metrics, rmse
from .modeling import fit_regressor, predict_regressor


@dataclass(frozen=True)
class ResidualFeatureBoosterConfig:
    residual_model_params: dict[str, Any]
    n_rounds: int = 5
    select_per_round: int = 1
    main_metric: str = "valid_bad_rmse_reduction"
    min_improvement: float = 0.0
    use_test_for_selection: bool = False
    min_valid_bad_samples: int = 1


@dataclass
class BoostingResult:
    selected_features: pd.DataFrame
    residual_curve: pd.DataFrame
    rankings: list[pd.DataFrame]
    final_predictions: dict[str, np.ndarray]


@dataclass
class _ScorePayload:
    row: dict[str, Any]
    model: Any | None = None
    pred_train: np.ndarray | None = None
    pred_valid: np.ndarray | None = None
    pred_test: np.ndarray | None = None


class ResidualFeatureBooster:
    def __init__(self, config: ResidualFeatureBoosterConfig):
        self.config = config

    def run_for_defect(
        self,
        *,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        candidate_cols: list[str],
        target_col: str,
        id_col: str,
        baseline_pred_col: str,
        defect_id: str,
        bad_sample_ids: set[str],
        good_sample_ids: set[str],
        quality_summary: pd.DataFrame | None = None,
        output_dir: str | Path | None = None,
    ) -> BoostingResult:
        valid_bad_mask = valid_df[id_col].astype(str).isin(bad_sample_ids).to_numpy()
        if int(valid_bad_mask.sum()) < self.config.min_valid_bad_samples:
            return BoostingResult(pd.DataFrame(), pd.DataFrame(), [], {})

        y_train = train_df[target_col].to_numpy(dtype=float)
        y_valid = valid_df[target_col].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)
        current_pred_train = train_df[baseline_pred_col].to_numpy(dtype=float).copy()
        current_pred_valid = valid_df[baseline_pred_col].to_numpy(dtype=float).copy()
        current_pred_test = test_df[baseline_pred_col].to_numpy(dtype=float).copy()

        remaining = list(dict.fromkeys(candidate_cols))
        selected_records: list[dict[str, Any]] = []
        curve_records: list[dict[str, Any]] = []
        rankings: list[pd.DataFrame] = []
        output_dir = Path(output_dir) if output_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        quality_lookup = _quality_lookup(quality_summary)
        main_metric = _main_metric_name(self.config.main_metric, self.config.use_test_for_selection)

        for round_idx in range(1, self.config.n_rounds + 1):
            if not remaining:
                break
            if not _finite(current_pred_train, current_pred_valid, current_pred_test):
                break

            payloads = [
                self._score_candidate(
                    feature,
                    train_df=train_df,
                    valid_df=valid_df,
                    test_df=test_df,
                    target_col=target_col,
                    id_col=id_col,
                    y_train=y_train,
                    y_valid=y_valid,
                    y_test=y_test,
                    current_pred_train=current_pred_train,
                    current_pred_valid=current_pred_valid,
                    current_pred_test=current_pred_test,
                    bad_sample_ids=bad_sample_ids,
                    good_sample_ids=good_sample_ids,
                    defect_id=defect_id,
                    round_idx=round_idx,
                    quality=quality_lookup.get(feature),
                )
                for feature in remaining
            ]
            ranking = pd.DataFrame([payload.row for payload in payloads])
            if ranking.empty:
                break
            ranking = ranking.sort_values([main_metric, "feature_name"], ascending=[False, True], na_position="last")
            ranking = ranking.reset_index(drop=True)
            ranking.insert(2, "rank", np.arange(1, len(ranking) + 1))

            valid_payloads = [
                payload
                for payload in payloads
                if not payload.row.get("fail_reason")
                and np.isfinite(float(payload.row.get(main_metric, np.nan)))
                and payload.model is not None
            ]
            valid_payloads.sort(key=lambda item: float(item.row.get(main_metric, np.nan)), reverse=True)
            selected_payloads = [
                payload
                for payload in valid_payloads
                if float(payload.row.get(main_metric, np.nan)) > self.config.min_improvement
            ][: max(1, self.config.select_per_round)]

            selected_names = {payload.row["feature_name"] for payload in selected_payloads}
            ranking["selected"] = ranking["feature_name"].isin(selected_names)
            rankings.append(ranking)
            if output_dir is not None:
                ranking.to_csv(output_dir / f"{defect_id}_round_{round_idx}.csv", index=False, encoding="utf-8-sig")

            if not selected_payloads:
                break

            for payload in selected_payloads:
                feature = str(payload.row["feature_name"])
                current_pred_train = current_pred_train + payload.pred_train
                current_pred_valid = current_pred_valid + payload.pred_valid
                current_pred_test = current_pred_test + payload.pred_test
                selected_records.append(_selected_record(payload.row))
                curve_records.append(
                    self._curve_record(
                        defect_id=defect_id,
                        round_idx=round_idx,
                        selected_feature=feature,
                        valid_df=valid_df,
                        test_df=test_df,
                        target_col=target_col,
                        id_col=id_col,
                        y_valid=y_valid,
                        y_test=y_test,
                        current_pred_valid=current_pred_valid,
                        current_pred_test=current_pred_test,
                        bad_sample_ids=bad_sample_ids,
                        good_sample_ids=good_sample_ids,
                    )
                )
                remaining = [candidate for candidate in remaining if candidate != feature]

        return BoostingResult(
            selected_features=pd.DataFrame(selected_records),
            residual_curve=pd.DataFrame(curve_records),
            rankings=rankings,
            final_predictions={"train": current_pred_train, "valid": current_pred_valid, "test": current_pred_test},
        )

    def _score_candidate(
        self,
        feature: str,
        *,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        id_col: str,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        current_pred_train: np.ndarray,
        current_pred_valid: np.ndarray,
        current_pred_test: np.ndarray,
        bad_sample_ids: set[str],
        good_sample_ids: set[str],
        defect_id: str,
        round_idx: int,
        quality: dict[str, Any] | None,
    ) -> _ScorePayload:
        base_row = {
            "defect_id": defect_id,
            "round": round_idx,
            "feature_name": feature,
            "valid_bad_rmse_reduction": np.nan,
            "valid_bad_mae_reduction": np.nan,
            "valid_bad_rmse_before": np.nan,
            "valid_bad_rmse_after": np.nan,
            "valid_bad_mae_before": np.nan,
            "valid_bad_mae_after": np.nan,
            "valid_good_rmse_reduction": np.nan,
            "valid_good_mae_reduction": np.nan,
            "valid_good_rmse_before": np.nan,
            "valid_good_rmse_after": np.nan,
            "valid_good_mae_before": np.nan,
            "valid_good_mae_after": np.nan,
            "valid_global_rmse_reduction": np.nan,
            "valid_global_mae_reduction": np.nan,
            "valid_global_rmse_before": np.nan,
            "valid_global_rmse_after": np.nan,
            "valid_global_mae_before": np.nan,
            "valid_global_mae_after": np.nan,
            "test_bad_rmse_reduction": np.nan,
            "test_bad_mae_reduction": np.nan,
            "test_bad_rmse_before": np.nan,
            "test_bad_rmse_after": np.nan,
            "test_bad_mae_before": np.nan,
            "test_bad_mae_after": np.nan,
            "test_good_rmse_reduction": np.nan,
            "test_good_mae_reduction": np.nan,
            "test_good_rmse_before": np.nan,
            "test_good_rmse_after": np.nan,
            "test_good_mae_before": np.nan,
            "test_good_mae_after": np.nan,
            "test_global_rmse_reduction": np.nan,
            "test_global_mae_reduction": np.nan,
            "test_global_rmse_before": np.nan,
            "test_global_rmse_after": np.nan,
            "test_global_mae_before": np.nan,
            "test_global_mae_after": np.nan,
            "missing_rate": np.nan,
            "bad_coverage": np.nan,
            "good_coverage": np.nan,
            "selected": False,
            "fail_reason": "",
        }
        if quality:
            base_row.update(
                {
                    "missing_rate": quality.get("missing_rate", np.nan),
                    "bad_coverage": quality.get("bad_coverage", np.nan),
                    "good_coverage": quality.get("good_coverage", np.nan),
                }
            )
            if not bool(quality.get("is_pass", True)):
                base_row["fail_reason"] = quality.get("fail_reason", "quality_filter_failed")
                return _ScorePayload(base_row)

        try:
            residual_train = y_train - current_pred_train
            residual_valid = y_valid - current_pred_valid
            if not np.isfinite(residual_train).all() or not np.isfinite(residual_valid).all():
                raise ValueError("residual contains NaN or inf")
            model = fit_regressor(
                train_df,
                residual_train,
                valid_df,
                residual_valid,
                [feature],
                self.config.residual_model_params,
            )
            pred_train = predict_regressor(model, train_df, [feature])
            pred_valid = predict_regressor(model, valid_df, [feature])
            pred_test = predict_regressor(model, test_df, [feature])
            if not _finite(pred_train, pred_valid, pred_test):
                raise ValueError("invalid prediction")
        except Exception as exc:
            base_row["fail_reason"] = f"model_fit_failed:{type(exc).__name__}"
            return _ScorePayload(base_row)

        pred_valid_after = current_pred_valid + pred_valid
        pred_test_after = current_pred_test + pred_test
        base_row.update(
            _reduction_columns(
                prefix="valid",
                df=valid_df,
                id_col=id_col,
                y=y_valid,
                before=current_pred_valid,
                after=pred_valid_after,
                bad_sample_ids=bad_sample_ids,
                good_sample_ids=good_sample_ids,
            )
        )
        base_row.update(
            _reduction_columns(
                prefix="test",
                df=test_df,
                id_col=id_col,
                y=y_test,
                before=current_pred_test,
                after=pred_test_after,
                bad_sample_ids=bad_sample_ids,
                good_sample_ids=good_sample_ids,
            )
        )
        return _ScorePayload(base_row, model=model, pred_train=pred_train, pred_valid=pred_valid, pred_test=pred_test)

    def _curve_record(
        self,
        *,
        defect_id: str,
        round_idx: int,
        selected_feature: str,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        id_col: str,
        y_valid: np.ndarray,
        y_test: np.ndarray,
        current_pred_valid: np.ndarray,
        current_pred_test: np.ndarray,
        bad_sample_ids: set[str],
        good_sample_ids: set[str],
    ) -> dict[str, Any]:
        valid_bad = _mask(valid_df, id_col, bad_sample_ids)
        valid_good = _mask(valid_df, id_col, good_sample_ids)
        test_bad = _mask(test_df, id_col, bad_sample_ids)
        test_good = _mask(test_df, id_col, good_sample_ids)
        return {
            "defect_id": defect_id,
            "round": round_idx,
            "selected_feature": selected_feature,
            "valid_bad_rmse": rmse(y_valid[valid_bad], current_pred_valid[valid_bad]),
            "valid_bad_mae": mae(y_valid[valid_bad], current_pred_valid[valid_bad]),
            "valid_good_rmse": rmse(y_valid[valid_good], current_pred_valid[valid_good]),
            "valid_good_mae": mae(y_valid[valid_good], current_pred_valid[valid_good]),
            "test_bad_rmse": rmse(y_test[test_bad], current_pred_test[test_bad]),
            "test_bad_mae": mae(y_test[test_bad], current_pred_test[test_bad]),
            "test_good_rmse": rmse(y_test[test_good], current_pred_test[test_good]),
            "test_good_mae": mae(y_test[test_good], current_pred_test[test_good]),
            "valid_global_rmse": rmse(y_valid, current_pred_valid),
            "test_global_rmse": rmse(y_test, current_pred_test),
        }


def _reduction_columns(
    *,
    prefix: str,
    df: pd.DataFrame,
    id_col: str,
    y: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    bad_sample_ids: set[str],
    good_sample_ids: set[str],
) -> dict[str, float]:
    bad = _mask(df, id_col, bad_sample_ids)
    good = _mask(df, id_col, good_sample_ids)
    groups = {"bad": bad, "good": good, "global": np.ones(len(df), dtype=bool)}
    result: dict[str, float] = {}
    for name, mask in groups.items():
        values = residual_reduction_metrics(y[mask], before[mask], after[mask])
        result[f"{prefix}_{name}_rmse_before"] = values["rmse_before"]
        result[f"{prefix}_{name}_rmse_after"] = values["rmse_after"]
        result[f"{prefix}_{name}_rmse_reduction"] = values["rmse_reduction"]
        result[f"{prefix}_{name}_mae_before"] = values["mae_before"]
        result[f"{prefix}_{name}_mae_after"] = values["mae_after"]
        result[f"{prefix}_{name}_mae_reduction"] = values["mae_reduction"]
    return result


def _mask(df: pd.DataFrame, id_col: str, sample_ids: set[str]) -> np.ndarray:
    return df[id_col].astype(str).isin(sample_ids).to_numpy()


def _finite(*arrays: np.ndarray) -> bool:
    return all(np.isfinite(np.asarray(arr, dtype=float)).all() for arr in arrays)


def _quality_lookup(summary: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if summary is None or summary.empty or "feature_name" not in summary.columns:
        return {}
    return summary.set_index("feature_name").to_dict(orient="index")


def _main_metric_name(metric: str, use_test_for_selection: bool) -> str:
    if metric in {"bad_rmse_reduction", "rmse_reduction"}:
        return "test_bad_rmse_reduction" if use_test_for_selection else "valid_bad_rmse_reduction"
    if metric.startswith("valid_") or metric.startswith("test_"):
        return metric
    return f"valid_{metric}"


def _selected_record(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "defect_id",
        "round",
        "feature_name",
        "valid_bad_rmse_before",
        "valid_bad_rmse_after",
        "valid_bad_rmse_reduction",
        "test_bad_rmse_before",
        "test_bad_rmse_after",
        "test_bad_rmse_reduction",
        "valid_good_rmse_reduction",
        "test_good_rmse_reduction",
        "valid_global_rmse_after",
        "valid_global_rmse_reduction",
        "test_global_rmse_after",
        "test_global_rmse_reduction",
    ]
    return {key: row.get(key, np.nan) for key in keys}
