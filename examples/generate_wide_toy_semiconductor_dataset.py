from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "toy_semiconductor_wide"


DEFAULT_FULL_ROWS_PER_ORDER = 2500
DEFAULT_BOOSTER_GOOD_ROWS = 180
DEFAULT_BOOSTER_BAD_ROWS = 70
DEFAULT_FEATURES_PER_ORDER = 10000
DATASET_VERSION = "wide_toyset_v3_nonlinear"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_order(
    order_id: int,
    n_rows: int,
    features_per_order: int,
    seed: int,
    booster_good_rows: int = DEFAULT_BOOSTER_GOOD_ROWS,
    booster_bad_rows: int = DEFAULT_BOOSTER_BAD_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one realistic full wafer pool and a Good/Bad booster slice.

    The full pool mimics 2,000-3,000 wafers passing through a process order.
    Only a small, operator-selected Good/Bad subset is returned for the booster,
    which matches the intended workflow: find contrasts among indexed samples
    without pretending every wafer in the factory is cleanly labeled.
    """

    if features_per_order < 500:
        raise ValueError("features_per_order should be at least 500 for the wide toyset")
    if n_rows <= booster_good_rows + booster_bad_rows:
        raise ValueError(
            "n_rows must be larger than booster_good_rows + booster_bad_rows. "
            "Try --rows 2500 --booster-good-rows 180 --booster-bad-rows 70."
        )

    rng = np.random.default_rng(seed + order_id * 1009)
    full_pool = _generate_full_order_pool(order_id, n_rows, features_per_order, rng)
    booster_slice = _select_booster_good_bad_slice(
        full_pool,
        booster_good_rows=booster_good_rows,
        booster_bad_rows=booster_bad_rows,
        rng=rng,
    )
    return booster_slice, full_pool


def _generate_full_order_pool(
    order_id: int,
    n_rows: int,
    features_per_order: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    process_run_seq = np.arange(1, n_rows + 1)
    process_elapsed_ratio = (process_run_seq - 0.5) / n_rows
    block_size = max(1, n_rows // 24)
    block_idx = np.minimum((process_run_seq - 1) // block_size + 1, 24)

    a_excursion = _make_excursion_curve(process_elapsed_ratio, rng, window_count=2)
    b_excursion = _make_excursion_curve(process_elapsed_ratio, rng, window_count=1)
    c_excursion = _make_excursion_curve(process_elapsed_ratio, rng, window_count=1)
    d_excursion = _make_excursion_curve(process_elapsed_ratio, rng, window_count=1)

    product_id = rng.choice(["P_A", "P_B", "P_C"], size=n_rows, p=[0.45, 0.35, 0.20])
    route_id = rng.choice(["R_BASE", "R_FAST", "R_REWORK"], size=n_rows, p=[0.68, 0.22, 0.10])
    lot_id = np.array([f"LOT_{order_id:03d}_{idx:02d}" for idx in block_idx])
    wafer_id = ((process_run_seq - 1) % 25) + 1

    equipment_name = _draw_equipment(n_rows, a_excursion, rng)
    process_step = np.array([_step_from_equipment(name) for name in equipment_name])
    chamber_id = _draw_chamber(n_rows, a_excursion, rng)

    etch_risky = (equipment_name == "ETCH_03").astype(float)
    chamber_risky = (chamber_id == "CH_C").astype(float)
    rework_route = (route_id == "R_REWORK").astype(float)
    fast_route = (route_id == "R_FAST").astype(float)
    product_b = (product_id == "P_B").astype(float)
    product_c = (product_id == "P_C").astype(float)

    lot_drift = rng.normal(0, 0.18, size=25)
    lot_effect = lot_drift[block_idx]

    process_time_sec = (
        1780
        + 90 * (process_step == "ETCH")
        + 115 * rework_route
        + 85 * etch_risky
        + 130 * a_excursion
        + rng.normal(0, 75, size=n_rows)
    ).astype("float32")

    a_latent = rng.normal(0, 1, size=n_rows)
    b_latent = rng.normal(0, 1, size=n_rows)
    c_latent = rng.normal(0, 1, size=n_rows)
    d_latent = rng.normal(0, 1, size=n_rows)

    prob_a = sigmoid(
        -2.95
        + 2.25 * a_excursion
        + 0.42 * etch_risky
        + 0.34 * chamber_risky
        + 0.35 * rework_route
        + 0.28 * lot_effect
        + 0.38 * a_latent
    )
    prob_b = sigmoid(-2.15 + 1.15 * b_excursion + 0.65 * (process_step == "PHOTO") + 0.35 * product_b + 0.35 * b_latent)
    prob_c = sigmoid(-2.30 + 1.10 * c_excursion + 0.70 * (process_step == "CMP") + 0.35 * product_c + 0.35 * c_latent)
    prob_d = sigmoid(-2.35 + 1.05 * d_excursion + 0.70 * (process_step == "CVD") + 0.35 * fast_route + 0.35 * d_latent)

    sim_a = rng.binomial(1, prob_a).astype("float32")
    sim_b = rng.binomial(1, prob_b).astype("float32")
    sim_c = rng.binomial(1, prob_c).astype("float32")
    sim_d = rng.binomial(1, prob_d).astype("float32")

    eds_bin_a_wf_mean = np.clip(
        rng.normal(0.030, 0.020, size=n_rows)
        + 0.050 * a_excursion
        + 0.105 * sim_a
        + 0.018 * etch_risky
        + 0.012 * rng.normal(0, 1, size=n_rows),
        0,
        0.60,
    )
    eds_bin_b_wf_mean = np.clip(rng.normal(0.025, 0.015, size=n_rows) + 0.045 * b_excursion + 0.105 * sim_b, 0, 0.50)
    eds_bin_c_wf_mean = np.clip(rng.normal(0.022, 0.014, size=n_rows) + 0.040 * c_excursion + 0.090 * sim_c, 0, 0.45)
    eds_bin_no_wf_mean = np.clip(
        1.0
        + 2.65 * eds_bin_a_wf_mean
        + 1.90 * eds_bin_b_wf_mean
        + 1.65 * eds_bin_c_wf_mean
        + 0.30 * sim_d
        + rng.normal(0, 0.10, size=n_rows),
        0,
        4,
    )
    eds_yield_wf_mean = np.clip(
        0.965
        - 0.31 * eds_bin_a_wf_mean
        - 0.22 * eds_bin_b_wf_mean
        - 0.20 * eds_bin_c_wf_mean
        - 0.045 * sim_d
        + rng.normal(0, 0.013, size=n_rows),
        0.55,
        0.995,
    )

    high_a_threshold = np.quantile(eds_bin_a_wf_mean, 0.885)
    target_bad_a = ((eds_bin_a_wf_mean >= high_a_threshold) | ((sim_a == 1) & (rng.random(n_rows) < 0.28))).astype(int)
    false_positive = ((sim_b + sim_c + sim_d) > 0) & (rng.random(n_rows) < 0.025)
    false_negative = (target_bad_a == 1) & (rng.random(n_rows) < 0.045)
    target_bad_a = np.where(false_positive, 1, target_bad_a)
    target_bad_a = np.where(false_negative, 0, target_bad_a).astype(int)

    base_df = pd.DataFrame(
        {
            "sample_id": [f"O{order_id:03d}_W{i:05d}" for i in process_run_seq],
            "order_id": order_id,
            "process_run_seq": process_run_seq,
            "process_elapsed_ratio": process_elapsed_ratio.round(6),
            "run_block_id": [f"RUN_{order_id:03d}_{idx:02d}" for idx in block_idx],
            "lot_id": lot_id,
            "wafer_id": wafer_id,
            "product_id": product_id,
            "route_id": route_id,
            "process_step": process_step,
            "equipment_name": equipment_name,
            "chamber_id": chamber_id,
            "process_time_sec": process_time_sec.round(2),
            "target_bad_a": target_bad_a,
            "booster_sample_role": "Ignored",
            "is_booster_sample": 0,
            "eds_bin_no_wf_mean": eds_bin_no_wf_mean.round(5),
            "eds_bin_a_wf_mean": eds_bin_a_wf_mean.round(5),
            "eds_bin_b_wf_mean": eds_bin_b_wf_mean.round(5),
            "eds_bin_c_wf_mean": eds_bin_c_wf_mean.round(5),
            "eds_yield_wf_mean": eds_yield_wf_mean.round(5),
            "sim_true_defect_a_flag": sim_a.astype(int),
            "sim_true_defect_b_flag": sim_b.astype(int),
            "sim_true_defect_c_flag": sim_c.astype(int),
            "sim_true_defect_d_flag": sim_d.astype(int),
        }
    )

    feature_names = build_feature_names(features_per_order)
    features = rng.normal(0, 1, size=(n_rows, features_per_order)).astype("float32")

    _plant_signal_blocks(
        features=features,
        names=feature_names,
        rng=rng,
        sim_a=sim_a,
        sim_b=sim_b,
        sim_c=sim_c,
        sim_d=sim_d,
        a_excursion=a_excursion,
        b_excursion=b_excursion,
        c_excursion=c_excursion,
        d_excursion=d_excursion,
        etch_risky=etch_risky,
        chamber_risky=chamber_risky,
        rework_route=rework_route,
        eds_bin_a_wf_mean=eds_bin_a_wf_mean,
        target_bad_a=target_bad_a,
    )

    feature_df = pd.DataFrame(features, columns=feature_names)
    return pd.concat([base_df, feature_df], axis=1)


def _make_excursion_curve(
    process_elapsed_ratio: np.ndarray,
    rng: np.random.Generator,
    window_count: int,
) -> np.ndarray:
    curve = np.zeros_like(process_elapsed_ratio, dtype=float)
    centers = rng.uniform(0.18, 0.86, size=window_count)
    widths = rng.uniform(0.018, 0.055, size=window_count)
    weights = rng.uniform(0.70, 1.20, size=window_count)
    for center, width, weight in zip(centers, widths, weights):
        curve += weight * np.exp(-0.5 * ((process_elapsed_ratio - center) / width) ** 2)
    if curve.max() > 0:
        curve = curve / curve.max()
    return np.clip(curve + rng.normal(0, 0.015, size=len(curve)), 0.0, 1.0)


def _draw_equipment(n_rows: int, a_excursion: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    baseline = rng.choice(
        ["ETCH_01", "ETCH_02", "CVD_01", "CMP_01", "PHOTO_01"],
        size=n_rows,
        p=[0.24, 0.20, 0.22, 0.20, 0.14],
    )
    p_etch03 = np.clip(0.08 + 0.30 * a_excursion, 0.08, 0.42)
    return np.where(rng.random(n_rows) < p_etch03, "ETCH_03", baseline)


def _draw_chamber(n_rows: int, a_excursion: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    baseline = rng.choice(["CH_A", "CH_B", "CH_D"], size=n_rows, p=[0.35, 0.35, 0.30])
    p_ch_c = np.clip(0.12 + 0.25 * a_excursion, 0.12, 0.36)
    return np.where(rng.random(n_rows) < p_ch_c, "CH_C", baseline)


def _step_from_equipment(equipment_name: str) -> str:
    if equipment_name.startswith("ETCH"):
        return "ETCH"
    if equipment_name.startswith("CVD"):
        return "CVD"
    if equipment_name.startswith("CMP"):
        return "CMP"
    return "PHOTO"


def _select_booster_good_bad_slice(
    full_pool: pd.DataFrame,
    booster_good_rows: int,
    booster_bad_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    bad_pool = full_pool[full_pool["target_bad_a"] == 1].copy()
    good_pool = full_pool[full_pool["target_bad_a"] == 0].copy()

    if len(bad_pool) < booster_bad_rows:
        raise ValueError(f"Only {len(bad_pool)} bad rows are available, but {booster_bad_rows} were requested.")
    if len(good_pool) < booster_good_rows:
        raise ValueError(f"Only {len(good_pool)} good rows are available, but {booster_good_rows} were requested.")

    bad_strength = bad_pool["eds_bin_a_wf_mean"].rank(pct=True).to_numpy()
    good_strength = 1.0 - good_pool["eds_bin_a_wf_mean"].rank(pct=True).to_numpy()
    bad_weights = 0.25 + bad_strength**2
    good_weights = 0.25 + good_strength**2

    bad_idx = rng.choice(
        bad_pool.index.to_numpy(),
        size=booster_bad_rows,
        replace=False,
        p=bad_weights / bad_weights.sum(),
    )
    good_idx = rng.choice(
        good_pool.index.to_numpy(),
        size=booster_good_rows,
        replace=False,
        p=good_weights / good_weights.sum(),
    )

    full_pool.loc[bad_idx, "booster_sample_role"] = "Bad"
    full_pool.loc[good_idx, "booster_sample_role"] = "Good"
    full_pool.loc[np.r_[bad_idx, good_idx], "is_booster_sample"] = 1

    selected = full_pool.loc[np.r_[bad_idx, good_idx]].copy()
    selected = selected.sort_values("process_run_seq").reset_index(drop=True)
    return selected


def build_feature_names(features_per_order: int) -> list[str]:
    planted = []
    planted += [f"sensor_a_pressure_like_{i:03d}" for i in range(40)]
    planted += [f"sensor_a_rf_power_like_{i:03d}" for i in range(40)]
    planted += [f"sensor_a_plasma_instability_like_{i:03d}" for i in range(40)]
    planted += [f"measure_a_cd_shift_like_{i:03d}" for i in range(40)]
    planted += [f"measure_a_cd_uniformity_like_{i:03d}" for i in range(40)]
    planted += [f"midproc_a_residue_count_like_{i:03d}" for i in range(40)]
    planted += [f"midproc_a_microbridge_count_like_{i:03d}" for i in range(40)]
    planted += [f"nonlinear_a_elbow_y_like_{i:03d}" for i in range(40)]
    planted += [f"nonlinear_a_saturation_like_{i:03d}" for i in range(40)]
    planted += [f"nonlinear_a_u_shape_like_{i:03d}" for i in range(40)]
    planted += [f"interaction_a_pressure_time_like_{i:03d}" for i in range(40)]
    planted += [f"bad_only_a_sparse_signature_{i:03d}" for i in range(20)]
    planted += [f"good_only_stable_signature_{i:03d}" for i in range(20)]
    planted += [f"defect_b_overlay_like_{i:03d}" for i in range(40)]
    planted += [f"defect_c_particle_like_{i:03d}" for i in range(40)]
    planted += [f"defect_d_thickness_like_{i:03d}" for i in range(40)]
    planted += [f"tool_confounded_etch03_like_{i:03d}" for i in range(40)]

    if len(planted) > features_per_order:
        return planted[:features_per_order]

    noise_count = features_per_order - len(planted)
    families = ["sensor_noise", "measure_noise", "midproc_noise", "derived_noise"]
    noise = [f"{families[i % len(families)]}_{i:05d}" for i in range(noise_count)]
    return planted + noise


def _plant_signal_blocks(
    features: np.ndarray,
    names: list[str],
    rng: np.random.Generator,
    sim_a: np.ndarray,
    sim_b: np.ndarray,
    sim_c: np.ndarray,
    sim_d: np.ndarray,
    a_excursion: np.ndarray,
    b_excursion: np.ndarray,
    c_excursion: np.ndarray,
    d_excursion: np.ndarray,
    etch_risky: np.ndarray,
    chamber_risky: np.ndarray,
    rework_route: np.ndarray,
    eds_bin_a_wf_mean: np.ndarray,
    target_bad_a: np.ndarray,
) -> None:
    index = {name: idx for idx, name in enumerate(names)}

    def matching_cols(prefix: str) -> list[int]:
        return [idx for name, idx in index.items() if name.startswith(prefix)]

    def fill_continuous(prefix: str, signal: np.ndarray, strength: float, noise: float, base: float = 0.0) -> None:
        for offset, col in enumerate(matching_cols(prefix)):
            jitter = rng.normal(0, noise, size=features.shape[0])
            col_gain = strength * rng.uniform(0.75, 1.20)
            features[:, col] = (base + col_gain * signal + 0.015 * (offset % 11) + jitter).astype("float32")

    def fill_sensor_spike(prefix: str, window: np.ndarray, event: np.ndarray, strength: float, base: float) -> None:
        for offset, col in enumerate(matching_cols(prefix)):
            local_baseline = base + rng.normal(0, 0.15)
            spike_prob = np.clip(0.025 + 0.48 * window + 0.14 * event, 0.0, 0.72)
            spike_mask = rng.random(features.shape[0]) < spike_prob
            spike = spike_mask * rng.gamma(shape=1.6, scale=strength * rng.uniform(0.75, 1.15), size=features.shape[0])
            small_variance = rng.normal(0, 0.10 + 0.02 * (offset % 5), size=features.shape[0])
            features[:, col] = (local_baseline + 0.18 * event + spike + small_variance).astype("float32")

    def fill_count(prefix: str, lam: np.ndarray, zero_inflation: float = 0.0) -> None:
        for offset, col in enumerate(matching_cols(prefix)):
            adjusted_lam = np.clip(lam * rng.uniform(0.70, 1.30) + 0.02 * (offset % 7), 0.01, None)
            values = rng.poisson(adjusted_lam).astype("float32")
            if zero_inflation > 0:
                values[rng.random(features.shape[0]) < zero_inflation] = 0
            features[:, col] = values

    a_sensor_signal = 0.72 * a_excursion + 0.22 * sim_a + 0.18 * etch_risky + 0.12 * chamber_risky
    a_measure_signal = 0.45 * a_excursion + 0.36 * sim_a + 1.45 * eds_bin_a_wf_mean
    a_count_lam = 0.18 + 1.65 * a_excursion + 1.15 * sim_a + 0.25 * rework_route
    elbow_driver = rng.normal(0, 0.75, size=features.shape[0]) + 2.2 * a_excursion + 0.45 * sim_a
    elbow_signal = np.maximum(elbow_driver - 1.10, 0.0) ** 1.35
    saturation_signal = np.log1p(5.5 * a_excursion + 1.4 * sim_a)
    u_shape_sign = rng.choice([-1.0, 1.0], size=features.shape[0])
    u_shape_signal = u_shape_sign * (0.35 + 1.9 * a_excursion + 0.70 * sim_a) + rng.normal(0, 0.18, size=features.shape[0])
    interaction_signal = (0.65 * a_excursion + 0.35 * sim_a) * (0.75 + 0.55 * rework_route + 0.30 * etch_risky)

    fill_sensor_spike("sensor_a_pressure_like_", a_excursion, sim_a, strength=0.65, base=49.5)
    fill_sensor_spike("sensor_a_rf_power_like_", np.clip(0.75 * a_excursion + 0.25 * rework_route, 0, 1), sim_a, strength=0.55, base=31.0)
    fill_continuous("sensor_a_plasma_instability_like_", a_sensor_signal, 0.95, 0.52, base=4.0)
    fill_continuous("measure_a_cd_shift_like_", a_measure_signal, 1.05, 0.55)
    fill_continuous("measure_a_cd_uniformity_like_", a_measure_signal + 0.08 * chamber_risky, 0.85, 0.50)
    fill_count("midproc_a_residue_count_like_", a_count_lam, zero_inflation=0.10)
    fill_count("midproc_a_microbridge_count_like_", 0.12 + 1.25 * a_excursion + 0.95 * sim_a, zero_inflation=0.18)
    fill_continuous("nonlinear_a_elbow_y_like_", elbow_signal, 1.10, 0.42)
    fill_continuous("nonlinear_a_saturation_like_", saturation_signal, 1.20, 0.48)
    fill_continuous("nonlinear_a_u_shape_like_", u_shape_signal, 1.00, 0.44)
    fill_continuous("interaction_a_pressure_time_like_", interaction_signal, 1.25, 0.50)

    fill_continuous("defect_b_overlay_like_", 0.65 * b_excursion + 0.55 * sim_b, 1.10, 0.70)
    fill_count("defect_c_particle_like_", 0.15 + 1.40 * c_excursion + 1.20 * sim_c, zero_inflation=0.15)
    fill_continuous("defect_d_thickness_like_", 0.60 * d_excursion + 0.55 * sim_d, 1.05, 0.72)
    fill_continuous("tool_confounded_etch03_like_", etch_risky + 0.35 * a_excursion, 1.30, 0.58)

    for name, col in index.items():
        if name.startswith("bad_only_a_sparse_signature_"):
            values = rng.normal(7.0, 1.4, size=features.shape[0]).astype("float32")
            present = (target_bad_a == 1) & (rng.random(features.shape[0]) < 0.48)
            values[~present] = np.nan
            features[:, col] = values
        elif name.startswith("good_only_stable_signature_"):
            values = rng.normal(1.0, 0.25, size=features.shape[0]).astype("float32")
            present = (target_bad_a == 0) & (rng.random(features.shape[0]) < 0.58)
            values[~present] = np.nan
            features[:, col] = values


def build_feature_metadata(features_per_order: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in build_feature_names(features_per_order):
        if "_a_" in feature or feature.startswith("bad_only_a_") or feature.startswith("good_only_stable_signature_"):
            prior = 0.80
            family = "planted_defect_a"
        elif feature.startswith("tool_confounded"):
            prior = 0.20
            family = "tool_confounded"
        elif feature.startswith("defect_"):
            prior = 0.30
            family = "other_defect_signal"
        else:
            prior = 0.00
            family = "noise"
        rows.append(
            {
                "feature_name": feature,
                "feature_family": family,
                "process_area": feature.split("_")[0],
                "unit": "synthetic",
                "domain_prior_score": prior,
            }
        )
    return pd.DataFrame(rows)


def generate_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    orders: int = 3,
    rows: int = DEFAULT_FULL_ROWS_PER_ORDER,
    features_per_order: int = DEFAULT_FEATURES_PER_ORDER,
    booster_good_rows: int = DEFAULT_BOOSTER_GOOD_ROWS,
    booster_bad_rows: int = DEFAULT_BOOSTER_BAD_ROWS,
    seed: int = 2026,
    overwrite: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for order_id in range(1, orders + 1):
        path = output_dir / f"wide_order_{order_id:03d}.csv"
        full_path = output_dir / f"wide_order_{order_id:03d}_full_pool.csv"
        if path.exists() and full_path.exists() and not overwrite:
            selected = pd.read_csv(path, usecols=["order_id", "target_bad_a", "is_booster_sample"])
            full = pd.read_csv(full_path, usecols=["order_id", "target_bad_a", "is_booster_sample"])
        else:
            selected, full = generate_order(
                order_id=order_id,
                n_rows=rows,
                features_per_order=features_per_order,
                seed=seed,
                booster_good_rows=booster_good_rows,
                booster_bad_rows=booster_bad_rows,
            )
            full.to_csv(full_path, index=False, encoding="utf-8-sig")
            selected.to_csv(path, index=False, encoding="utf-8-sig")
        summaries.append(
            {
                "dataset_version": DATASET_VERSION,
                "order_id": order_id,
                "full_rows": len(full),
                "booster_rows": len(selected),
                "booster_good_count": int((selected["target_bad_a"] == 0).sum()),
                "booster_bad_count": int((selected["target_bad_a"] == 1).sum()),
                "ignored_rows": int((full["is_booster_sample"] == 0).sum()),
                "full_pool_bad_count": int(full["target_bad_a"].sum()),
                "full_pool_bad_rate": float(full["target_bad_a"].mean()),
                "candidate_feature_count": features_per_order,
                "booster_file": str(path),
                "full_pool_file": str(full_path),
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "wide_target_summary_by_order.csv", index=False, encoding="utf-8-sig")
    build_feature_metadata(features_per_order).to_csv(
        output_dir / "wide_feature_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a wide semiconductor toy dataset.")
    parser.add_argument("--orders", type=int, default=3)
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_FULL_ROWS_PER_ORDER,
        help="Full wafer rows per order before selecting Good/Bad booster rows.",
    )
    parser.add_argument("--features-per-order", type=int, default=DEFAULT_FEATURES_PER_ORDER)
    parser.add_argument("--booster-good-rows", type=int, default=DEFAULT_BOOSTER_GOOD_ROWS)
    parser.add_argument("--booster-bad-rows", type=int, default=DEFAULT_BOOSTER_BAD_ROWS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = generate_dataset(
        output_dir=args.output_dir,
        orders=args.orders,
        rows=args.rows,
        features_per_order=args.features_per_order,
        booster_good_rows=args.booster_good_rows,
        booster_bad_rows=args.booster_bad_rows,
        seed=args.seed,
        overwrite=not args.no_overwrite,
    )
    print(f"Wrote wide toy dataset to {args.output_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
