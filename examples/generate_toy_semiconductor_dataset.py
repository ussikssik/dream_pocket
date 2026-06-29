from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "toy_semiconductor"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_order(order_id: int, n_rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(202600 + order_id)

    product_id = rng.choice(["P_A", "P_B", "P_C"], size=n_rows, p=[0.45, 0.35, 0.20])
    lot_id = rng.choice([f"LOT_{order_id:02d}_{i:02d}" for i in range(1, 9)], size=n_rows)
    wafer_id = rng.integers(1, 26, size=n_rows)
    route_id = rng.choice(["R_BASE", "R_FAST", "R_REWORK"], size=n_rows, p=[0.70, 0.20, 0.10])
    process_step = rng.choice(["ETCH", "CVD", "CMP", "PHOTO"], size=n_rows, p=[0.38, 0.25, 0.22, 0.15])
    equipment_name = rng.choice(
        ["ETCH_01", "ETCH_02", "ETCH_03", "CVD_01", "CMP_01", "PHOTO_01"],
        size=n_rows,
        p=[0.22, 0.18, 0.14, 0.18, 0.16, 0.12],
    )
    chamber_id = rng.choice(["CH_A", "CH_B", "CH_C", "CH_D"], size=n_rows, p=[0.28, 0.27, 0.23, 0.22])

    etch_risky = (equipment_name == "ETCH_03").astype(float)
    chamber_risky = (chamber_id == "CH_C").astype(float)
    rework_route = (route_id == "R_REWORK").astype(float)
    product_shift = np.select([product_id == "P_A", product_id == "P_B", product_id == "P_C"], [0.0, 0.10, -0.08])

    process_time_sec = (
        1800
        + 180 * (process_step == "ETCH")
        + 110 * rework_route
        + 95 * etch_risky
        + rng.normal(0, 90, size=n_rows)
    ).round(1)

    plasma_instability_base = rng.normal(0, 1, size=n_rows)
    latent_a_prob = sigmoid(
        -2.4
        + 1.25 * etch_risky
        + 0.85 * chamber_risky
        + 0.55 * rework_route
        + 0.35 * product_shift
        + 0.004 * (process_time_sec - 1900)
        + 0.45 * plasma_instability_base
    )
    latent_b_prob = sigmoid(-2.0 + 0.70 * (process_step == "PHOTO") + 0.40 * (product_id == "P_B"))
    latent_c_prob = sigmoid(-2.2 + 0.75 * (process_step == "CMP") + 0.35 * (equipment_name == "CMP_01"))
    latent_d_prob = sigmoid(-2.3 + 0.65 * (process_step == "CVD") + 0.45 * (route_id == "R_FAST"))

    sim_defect_a_flag = rng.binomial(1, latent_a_prob)
    sim_defect_b_flag = rng.binomial(1, latent_b_prob)
    sim_defect_c_flag = rng.binomial(1, latent_c_prob)
    sim_defect_d_flag = rng.binomial(1, latent_d_prob)

    sensor_pressure_chamber_mean = rng.normal(3.0, 0.22, size=n_rows) + 0.42 * sim_defect_a_flag + 0.18 * etch_risky
    sensor_rf_power_mean = rng.normal(850, 28, size=n_rows) + 36 * sim_defect_a_flag + 18 * (process_step == "ETCH")
    sensor_plasma_instability_score = rng.normal(0.0, 0.9, size=n_rows) + 1.15 * sim_defect_a_flag + 0.35 * chamber_risky
    sensor_endpoint_time_sec = rng.normal(62, 4.2, size=n_rows) + 3.6 * sim_defect_a_flag + 1.4 * rework_route

    measure_cd_post_etch_mean_nm = rng.normal(42.0, 1.7, size=n_rows) + 2.1 * sim_defect_a_flag
    measure_cd_post_etch_std_nm = rng.normal(1.2, 0.25, size=n_rows) + 0.55 * sim_defect_a_flag
    measure_overlay_x_nm = rng.normal(0.0, 5.0, size=n_rows) + 5.8 * sim_defect_b_flag
    measure_overlay_y_nm = rng.normal(0.0, 5.2, size=n_rows) - 4.6 * sim_defect_b_flag
    measure_thickness_post_cvd_nm = rng.normal(780, 18, size=n_rows) + 22 * sim_defect_d_flag
    measure_film_resistance_ohm = rng.normal(38, 2.8, size=n_rows) + 4.5 * sim_defect_d_flag

    midproc_defect_residue_count = rng.poisson(2.0 + 5.0 * sim_defect_a_flag + 0.8 * rework_route)
    midproc_defect_microbridge_count = rng.poisson(0.7 + 3.8 * sim_defect_a_flag)
    midproc_defect_particle_count = rng.poisson(4.0 + 2.5 * sim_defect_c_flag + 1.2 * rework_route)
    midproc_defect_scratch_count = rng.poisson(0.8 + 2.9 * sim_defect_c_flag)
    midproc_defect_bridge_count = rng.poisson(1.0 + 3.4 * sim_defect_b_flag)
    midproc_defect_void_count = rng.poisson(1.2 + 2.8 * sim_defect_d_flag)
    midproc_defect_pattern_collapse_count = rng.poisson(0.5 + 1.7 * sim_defect_b_flag)

    eds_bin_a_wf_mean = np.clip(rng.normal(0.035, 0.018, size=n_rows) + 0.145 * sim_defect_a_flag, 0, 0.55)
    eds_bin_b_wf_mean = np.clip(rng.normal(0.025, 0.015, size=n_rows) + 0.115 * sim_defect_b_flag, 0, 0.45)
    eds_bin_c_wf_mean = np.clip(rng.normal(0.022, 0.014, size=n_rows) + 0.095 * sim_defect_c_flag, 0, 0.40)
    eds_bin_no_wf_mean = np.clip(
        1.0
        + 2.7 * eds_bin_a_wf_mean
        + 2.1 * eds_bin_b_wf_mean
        + 1.8 * eds_bin_c_wf_mean
        + 0.35 * sim_defect_d_flag
        + rng.normal(0, 0.08, size=n_rows),
        0,
        4,
    )
    eds_yield_wf_mean = np.clip(
        0.965
        - 0.32 * eds_bin_a_wf_mean
        - 0.22 * eds_bin_b_wf_mean
        - 0.20 * eds_bin_c_wf_mean
        - 0.05 * sim_defect_d_flag
        + rng.normal(0, 0.012, size=n_rows),
        0.55,
        0.995,
    )

    target_bad_a = ((eds_bin_a_wf_mean >= 0.11) | ((sim_defect_a_flag == 1) & (rng.random(n_rows) < 0.18))).astype(int)
    false_positive = ((sim_defect_b_flag + sim_defect_c_flag + sim_defect_d_flag) > 0) & (rng.random(n_rows) < 0.035)
    target_bad_a = np.where(false_positive, 1, target_bad_a)

    dominant = np.array(["GOOD"] * n_rows, dtype=object)
    dominant[sim_defect_a_flag == 1] = "A"
    dominant[(sim_defect_a_flag == 0) & (sim_defect_b_flag == 1)] = "B"
    dominant[(sim_defect_a_flag == 0) & (sim_defect_b_flag == 0) & (sim_defect_c_flag == 1)] = "C"
    dominant[
        (sim_defect_a_flag == 0)
        & (sim_defect_b_flag == 0)
        & (sim_defect_c_flag == 0)
        & (sim_defect_d_flag == 1)
    ] = "D"

    df = pd.DataFrame(
        {
            "sample_id": [f"O{order_id:03d}_W{i:04d}" for i in range(n_rows)],
            "order_id": order_id,
            "lot_id": lot_id,
            "wafer_id": wafer_id,
            "product_id": product_id,
            "route_id": route_id,
            "process_step": process_step,
            "equipment_name": equipment_name,
            "chamber_id": chamber_id,
            "process_time_sec": process_time_sec,
            "target_bad_a": target_bad_a,
            "eds_bin_no_wf_mean": eds_bin_no_wf_mean.round(4),
            "eds_bin_a_wf_mean": eds_bin_a_wf_mean.round(4),
            "eds_bin_b_wf_mean": eds_bin_b_wf_mean.round(4),
            "eds_bin_c_wf_mean": eds_bin_c_wf_mean.round(4),
            "eds_yield_wf_mean": eds_yield_wf_mean.round(4),
            "sensor_temp_zone1_mean": rng.normal(322, 5.5, size=n_rows).round(3),
            "sensor_temp_zone2_mean": rng.normal(318, 5.0, size=n_rows).round(3),
            "sensor_pressure_chamber_mean": sensor_pressure_chamber_mean.round(4),
            "sensor_rf_power_mean": sensor_rf_power_mean.round(3),
            "sensor_gas_flow_o2_mean": rng.normal(126, 8, size=n_rows).round(3),
            "sensor_gas_flow_ar_mean": rng.normal(208, 10, size=n_rows).round(3),
            "sensor_vacuum_leak_rate": np.clip(rng.normal(0.08, 0.025, size=n_rows) + 0.025 * sim_defect_a_flag, 0, 0.35).round(5),
            "sensor_plasma_impedance_std": np.clip(rng.normal(0.42, 0.11, size=n_rows) + 0.18 * sim_defect_a_flag, 0, 2).round(4),
            "sensor_endpoint_time_sec": sensor_endpoint_time_sec.round(3),
            "sensor_plasma_instability_score": sensor_plasma_instability_score.round(4),
            "measure_cd_post_etch_mean_nm": measure_cd_post_etch_mean_nm.round(4),
            "measure_cd_post_etch_std_nm": np.clip(measure_cd_post_etch_std_nm, 0.1, None).round(4),
            "measure_overlay_x_nm": measure_overlay_x_nm.round(4),
            "measure_overlay_y_nm": measure_overlay_y_nm.round(4),
            "measure_thickness_post_cvd_nm": measure_thickness_post_cvd_nm.round(4),
            "measure_film_resistance_ohm": measure_film_resistance_ohm.round(4),
            "measure_inline_particle_density": np.clip(rng.normal(0.18, 0.05, size=n_rows) + 0.05 * sim_defect_c_flag, 0, 1).round(5),
            "midproc_defect_particle_count": midproc_defect_particle_count,
            "midproc_defect_scratch_count": midproc_defect_scratch_count,
            "midproc_defect_bridge_count": midproc_defect_bridge_count,
            "midproc_defect_residue_count": midproc_defect_residue_count,
            "midproc_defect_void_count": midproc_defect_void_count,
            "midproc_defect_pattern_collapse_count": midproc_defect_pattern_collapse_count,
            "midproc_defect_microbridge_count": midproc_defect_microbridge_count,
            "bad_only_residue_signature": np.where(target_bad_a == 1, rng.normal(10.0, 1.5, size=n_rows), np.nan),
            "good_only_stable_signature": np.where(target_bad_a == 0, rng.normal(1.0, 0.15, size=n_rows), np.nan),
            "sim_true_defect_a_flag": sim_defect_a_flag,
            "sim_true_defect_b_flag": sim_defect_b_flag,
            "sim_true_defect_c_flag": sim_defect_c_flag,
            "sim_true_defect_d_flag": sim_defect_d_flag,
            "sim_dominant_defect": dominant,
        }
    )
    return df


def build_feature_metadata() -> pd.DataFrame:
    rows = [
        ("equipment_name", "equipment", "process metadata", "", 0.55),
        ("chamber_id", "equipment", "process metadata", "", 0.45),
        ("process_step", "process", "process metadata", "", 0.35),
        ("route_id", "process", "process metadata", "", 0.25),
        ("process_time_sec", "process", "process metadata", "sec", 0.55),
        ("sensor_pressure_chamber_mean", "sensor", "etch", "Torr", 0.75),
        ("sensor_rf_power_mean", "sensor", "etch", "W", 0.70),
        ("sensor_plasma_instability_score", "sensor", "etch", "score", 0.85),
        ("sensor_endpoint_time_sec", "sensor", "etch", "sec", 0.65),
        ("measure_cd_post_etch_mean_nm", "measure", "post etch metrology", "nm", 0.80),
        ("measure_cd_post_etch_std_nm", "measure", "post etch metrology", "nm", 0.75),
        ("midproc_defect_residue_count", "midproc defect count", "inline inspection", "count", 0.90),
        ("midproc_defect_microbridge_count", "midproc defect count", "inline inspection", "count", 0.80),
        ("bad_only_residue_signature", "asymmetric sparse", "inline inspection", "score", 0.60),
    ]
    return pd.DataFrame(rows, columns=["feature_name", "feature_family", "process_area", "unit", "domain_prior_score"])


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for order_id in range(1, 6):
        df = generate_order(order_id)
        frames.append(df)
        df.to_csv(OUTPUT_DIR / f"order_{order_id:03d}.csv", index=False, encoding="utf-8-sig")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "toy_semiconductor_all_orders.csv", index=False, encoding="utf-8-sig")
    build_feature_metadata().to_csv(OUTPUT_DIR / "feature_metadata.csv", index=False, encoding="utf-8-sig")

    summary = combined.groupby("order_id")["target_bad_a"].agg(["count", "sum", "mean"]).reset_index()
    summary.rename(columns={"sum": "bad_count", "mean": "bad_rate"}, inplace=True)
    summary.to_csv(OUTPUT_DIR / "target_summary_by_order.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote toy dataset to {OUTPUT_DIR}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
