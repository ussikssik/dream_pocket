from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "toy_semiconductor_wide"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_order(
    order_id: int,
    n_rows: int,
    features_per_order: int,
    seed: int,
) -> pd.DataFrame:
    """Generate one wide order with many candidate features.

    The first feature blocks contain intentionally planted signals. The remaining
    columns are mostly noise, which makes the booster behave more like a real
    large candidate-feature search.
    """

    if features_per_order < 500:
        raise ValueError("features_per_order should be at least 500 for the wide toyset")

    rng = np.random.default_rng(seed + order_id * 1009)

    product_id = rng.choice(["P_A", "P_B", "P_C"], size=n_rows, p=[0.45, 0.35, 0.20])
    lot_id = rng.choice([f"LOT_{order_id:03d}_{i:02d}" for i in range(1, 9)], size=n_rows)
    wafer_id = rng.integers(1, 26, size=n_rows)
    route_id = rng.choice(["R_BASE", "R_FAST", "R_REWORK"], size=n_rows, p=[0.68, 0.22, 0.10])
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
    fast_route = (route_id == "R_FAST").astype(float)
    product_b = (product_id == "P_B").astype(float)
    product_c = (product_id == "P_C").astype(float)

    process_time_sec = (
        1780
        + 190 * (process_step == "ETCH")
        + 105 * rework_route
        + 90 * etch_risky
        + rng.normal(0, 85, size=n_rows)
    ).astype("float32")

    a_latent = rng.normal(0, 1, size=n_rows)
    b_latent = rng.normal(0, 1, size=n_rows)
    c_latent = rng.normal(0, 1, size=n_rows)
    d_latent = rng.normal(0, 1, size=n_rows)

    prob_a = sigmoid(-2.4 + 1.20 * etch_risky + 0.85 * chamber_risky + 0.50 * rework_route + 0.45 * a_latent)
    prob_b = sigmoid(-2.0 + 0.75 * (process_step == "PHOTO") + 0.45 * product_b + 0.35 * b_latent)
    prob_c = sigmoid(-2.2 + 0.80 * (process_step == "CMP") + 0.40 * product_c + 0.35 * c_latent)
    prob_d = sigmoid(-2.3 + 0.80 * (process_step == "CVD") + 0.40 * fast_route + 0.35 * d_latent)

    sim_a = rng.binomial(1, prob_a).astype("float32")
    sim_b = rng.binomial(1, prob_b).astype("float32")
    sim_c = rng.binomial(1, prob_c).astype("float32")
    sim_d = rng.binomial(1, prob_d).astype("float32")

    eds_bin_a_wf_mean = np.clip(rng.normal(0.035, 0.018, size=n_rows) + 0.145 * sim_a, 0, 0.60)
    eds_bin_b_wf_mean = np.clip(rng.normal(0.025, 0.015, size=n_rows) + 0.115 * sim_b, 0, 0.50)
    eds_bin_c_wf_mean = np.clip(rng.normal(0.022, 0.014, size=n_rows) + 0.095 * sim_c, 0, 0.45)
    eds_bin_no_wf_mean = np.clip(
        1.0
        + 2.8 * eds_bin_a_wf_mean
        + 2.0 * eds_bin_b_wf_mean
        + 1.7 * eds_bin_c_wf_mean
        + 0.35 * sim_d
        + rng.normal(0, 0.08, size=n_rows),
        0,
        4,
    )
    eds_yield_wf_mean = np.clip(
        0.965
        - 0.32 * eds_bin_a_wf_mean
        - 0.22 * eds_bin_b_wf_mean
        - 0.20 * eds_bin_c_wf_mean
        - 0.05 * sim_d
        + rng.normal(0, 0.012, size=n_rows),
        0.55,
        0.995,
    )

    target_bad_a = ((eds_bin_a_wf_mean >= 0.11) | ((sim_a == 1) & (rng.random(n_rows) < 0.20))).astype(int)
    false_positive = ((sim_b + sim_c + sim_d) > 0) & (rng.random(n_rows) < 0.035)
    target_bad_a = np.where(false_positive, 1, target_bad_a).astype(int)

    base_df = pd.DataFrame(
        {
            "sample_id": [f"O{order_id:03d}_W{i:05d}" for i in range(n_rows)],
            "order_id": order_id,
            "lot_id": lot_id,
            "wafer_id": wafer_id,
            "product_id": product_id,
            "route_id": route_id,
            "process_step": process_step,
            "equipment_name": equipment_name,
            "chamber_id": chamber_id,
            "process_time_sec": process_time_sec.round(2),
            "target_bad_a": target_bad_a,
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
        etch_risky=etch_risky,
        chamber_risky=chamber_risky,
        rework_route=rework_route,
        target_bad_a=target_bad_a,
    )

    feature_df = pd.DataFrame(features, columns=feature_names)
    return pd.concat([base_df, feature_df], axis=1)


def build_feature_names(features_per_order: int) -> list[str]:
    planted = []
    planted += [f"sensor_a_pressure_like_{i:03d}" for i in range(40)]
    planted += [f"sensor_a_rf_power_like_{i:03d}" for i in range(40)]
    planted += [f"sensor_a_plasma_instability_like_{i:03d}" for i in range(40)]
    planted += [f"measure_a_cd_shift_like_{i:03d}" for i in range(40)]
    planted += [f"measure_a_cd_uniformity_like_{i:03d}" for i in range(40)]
    planted += [f"midproc_a_residue_count_like_{i:03d}" for i in range(40)]
    planted += [f"midproc_a_microbridge_count_like_{i:03d}" for i in range(40)]
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
    etch_risky: np.ndarray,
    chamber_risky: np.ndarray,
    rework_route: np.ndarray,
    target_bad_a: np.ndarray,
) -> None:
    index = {name: idx for idx, name in enumerate(names)}

    def fill(prefix: str, signal: np.ndarray, strength: float, noise: float = 1.0) -> None:
        cols = [idx for name, idx in index.items() if name.startswith(prefix)]
        for offset, col in enumerate(cols):
            jitter = rng.normal(0, noise, size=features.shape[0])
            features[:, col] = (strength * signal + 0.05 * offset + jitter).astype("float32")

    fill("sensor_a_pressure_like_", sim_a + 0.35 * etch_risky, 1.15, 0.80)
    fill("sensor_a_rf_power_like_", sim_a + 0.25 * rework_route, 1.00, 0.90)
    fill("sensor_a_plasma_instability_like_", sim_a + 0.30 * chamber_risky, 1.35, 0.95)
    fill("measure_a_cd_shift_like_", sim_a, 1.25, 0.75)
    fill("measure_a_cd_uniformity_like_", sim_a, 0.95, 0.70)
    fill("midproc_a_residue_count_like_", sim_a + 0.15 * rework_route, 1.55, 0.95)
    fill("midproc_a_microbridge_count_like_", sim_a, 1.15, 1.00)
    fill("defect_b_overlay_like_", sim_b, 1.30, 0.85)
    fill("defect_c_particle_like_", sim_c, 1.25, 0.85)
    fill("defect_d_thickness_like_", sim_d, 1.20, 0.85)
    fill("tool_confounded_etch03_like_", etch_risky, 1.45, 0.70)

    for name, col in index.items():
        if name.startswith("bad_only_a_sparse_signature_"):
            values = rng.normal(8.0, 1.2, size=features.shape[0]).astype("float32")
            values[target_bad_a == 0] = np.nan
            features[:, col] = values
        elif name.startswith("good_only_stable_signature_"):
            values = rng.normal(1.0, 0.2, size=features.shape[0]).astype("float32")
            values[target_bad_a == 1] = np.nan
            features[:, col] = values


def build_feature_metadata(features_per_order: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in build_feature_names(features_per_order):
        if "_a_" in feature or feature.startswith("bad_only_a_"):
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
    rows: int = 240,
    features_per_order: int = 10000,
    seed: int = 2026,
    overwrite: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for order_id in range(1, orders + 1):
        path = output_dir / f"wide_order_{order_id:03d}.csv"
        if path.exists() and not overwrite:
            df = pd.read_csv(path, usecols=["order_id", "target_bad_a"])
        else:
            df = generate_order(order_id, rows, features_per_order, seed)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        summaries.append(
            {
                "order_id": order_id,
                "rows": len(df),
                "candidate_feature_count": features_per_order,
                "bad_count": int(df["target_bad_a"].sum()),
                "bad_rate": float(df["target_bad_a"].mean()),
                "file": str(path),
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
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--features-per-order", type=int, default=10000)
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
        seed=args.seed,
        overwrite=not args.no_overwrite,
    )
    print(f"Wrote wide toy dataset to {args.output_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
