from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_evidence_report(result_path: Path | str) -> pd.DataFrame:
    """Load a combined feature evidence CSV."""

    return pd.read_csv(result_path)


def select_top_features(
    evidence: pd.DataFrame,
    order_id: int | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return top-ranked evidence rows, optionally for one order."""

    work = evidence.copy()
    if order_id is not None:
        work = work[work["order_id"] == order_id]
    if "final_rank" in work.columns:
        work = work.sort_values(["order_id", "final_rank"])
    else:
        work = work.sort_values(["order_id", "final_score"], ascending=[True, False])
    return work.groupby("order_id", group_keys=False).head(top_n).reset_index(drop=True)


def load_order_slice(
    data_dir: Path | str,
    order_id: int,
    feature_names: Iterable[str],
    file_pattern: str = "wide_order_{order_id:03d}.csv",
    base_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Load only the columns needed for visual review from a wide order CSV."""

    data_dir = Path(data_dir)
    path = data_dir / file_pattern.format(order_id=int(order_id))
    if not path.exists():
        raise FileNotFoundError(f"Order file not found: {path}")

    header = pd.read_csv(path, nrows=0)
    available = set(header.columns)
    requested = list(dict.fromkeys([*base_cols, *feature_names]))
    usecols = [col for col in requested if col in available]
    missing = [col for col in requested if col not in available]
    if missing:
        print(f"Skipped missing columns in order {order_id}: {missing[:10]}")
    return pd.read_csv(path, usecols=usecols)


def plot_feature_review(
    df: pd.DataFrame,
    feature_name: str,
    y_col: str,
    label_col: str = "target_bad_a",
    facet_col: str = "equipment_name",
    evidence_row: pd.Series | None = None,
):
    """Create a compact 2x2 review chart for one feature.

    The plot checks:
    1. feature-vs-y scatter
    2. Good/Bad feature distribution
    3. facet-level median feature and bad rate
    4. feature presence by Good/Bad
    """

    plt = _require_matplotlib()

    if feature_name not in df.columns:
        raise ValueError(f"feature '{feature_name}' is missing from df")
    if y_col not in df.columns:
        raise ValueError(f"y_col '{y_col}' is missing from df")
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' is missing from df")

    feature = df[feature_name]
    y = pd.to_numeric(df[y_col], errors="coerce")
    label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    numeric_feature = pd.to_numeric(feature, errors="coerce")
    is_numeric = numeric_feature.notna().sum() >= max(3, int(feature.notna().sum() * 0.8))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(_title(feature_name, evidence_row), fontsize=13, fontweight="bold")

    _scatter_feature_y(axes[0, 0], feature, numeric_feature, is_numeric, y, label, feature_name, y_col)
    _distribution_by_label(axes[0, 1], feature, numeric_feature, is_numeric, label, feature_name)
    _facet_summary(axes[1, 0], df, feature_name, numeric_feature, is_numeric, label, facet_col)
    _presence_by_label(axes[1, 1], feature, label, feature_name)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_order_top_features(
    evidence: pd.DataFrame,
    data_dir: Path | str,
    order_id: int,
    y_col: str = "eds_bin_a_wf_mean",
    top_n: int = 5,
    file_pattern: str = "wide_order_{order_id:03d}.csv",
    label_col: str = "target_bad_a",
    facet_col: str = "equipment_name",
    extra_cols: Iterable[str] = (),
) -> list[object]:
    """Load one order and plot review charts for its top features."""

    top = select_top_features(evidence, order_id=order_id, top_n=top_n)
    feature_names = top["feature_name"].tolist()
    base_cols = [
        "sample_id",
        "lot_id",
        "wafer_id",
        "product_id",
        "route_id",
        "process_step",
        "equipment_name",
        "chamber_id",
        label_col,
        y_col,
        *extra_cols,
    ]
    df = load_order_slice(data_dir, order_id, feature_names, file_pattern=file_pattern, base_cols=base_cols)
    figures = []
    for _, row in top.iterrows():
        feature_name = row["feature_name"]
        figures.append(
            plot_feature_review(
                df,
                feature_name=feature_name,
                y_col=y_col,
                label_col=label_col,
                facet_col=facet_col,
                evidence_row=row,
            )
        )
    return figures


def plot_feature_process_window(
    df: pd.DataFrame,
    feature_name: str,
    y_col: str = "eds_bin_a_wf_mean",
    label_col: str = "target_bad_a",
    time_col: str = "process_run_seq",
    role_col: str = "booster_sample_role",
    evidence_row: pd.Series | None = None,
):
    """Plot one feature and y over process order to inspect excursion windows."""

    plt = _require_matplotlib()

    if feature_name not in df.columns:
        raise ValueError(f"feature '{feature_name}' is missing from df")
    if y_col not in df.columns:
        raise ValueError(f"y_col '{y_col}' is missing from df")

    if time_col in df.columns:
        x = pd.to_numeric(df[time_col], errors="coerce")
    else:
        x = pd.Series(np.arange(1, len(df) + 1), index=df.index)

    feature = pd.to_numeric(df[feature_name], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    roles = _review_roles(df, label_col=label_col, role_col=role_col)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(_title(feature_name, evidence_row), fontsize=13, fontweight="bold")

    _scatter_by_role(axes[0], x, feature, roles)
    axes[0].set_ylabel(feature_name)
    axes[0].set_title("feature by process sequence")
    axes[0].grid(alpha=0.25)

    _scatter_by_role(axes[1], x, y, roles)
    axes[1].set_xlabel(time_col if time_col in df.columns else "row_order")
    axes[1].set_ylabel(y_col)
    axes[1].set_title("y by process sequence")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_order_top_feature_process_windows(
    evidence: pd.DataFrame,
    data_dir: Path | str,
    order_id: int,
    y_col: str = "eds_bin_a_wf_mean",
    top_n: int = 3,
    file_pattern: str = "wide_order_{order_id:03d}_full_pool.csv",
    label_col: str = "target_bad_a",
    time_col: str = "process_run_seq",
    role_col: str = "booster_sample_role",
) -> list[object]:
    """Plot process-window views for top-ranked features using the full pool file."""

    top = select_top_features(evidence, order_id=order_id, top_n=top_n)
    feature_names = top["feature_name"].tolist()
    base_cols = [label_col, y_col, time_col, role_col]
    df = load_order_slice(data_dir, order_id, feature_names, file_pattern=file_pattern, base_cols=base_cols)
    figures = []
    for _, row in top.iterrows():
        figures.append(
            plot_feature_process_window(
                df,
                feature_name=row["feature_name"],
                y_col=y_col,
                label_col=label_col,
                time_col=time_col,
                role_col=role_col,
                evidence_row=row,
            )
        )
    return figures


def make_review_decision_table(
    evidence: pd.DataFrame,
    data_dir: Path | str,
    order_id: int,
    y_col: str = "eds_bin_a_wf_mean",
    top_n: int = 10,
    file_pattern: str = "wide_order_{order_id:03d}.csv",
    label_col: str = "target_bad_a",
    facet_col: str = "equipment_name",
) -> pd.DataFrame:
    """Build a compact table with plot-adjacent diagnostics for top features."""

    top = select_top_features(evidence, order_id=order_id, top_n=top_n)
    feature_names = top["feature_name"].tolist()
    df = load_order_slice(
        data_dir,
        order_id,
        feature_names,
        file_pattern=file_pattern,
        base_cols=[label_col, y_col, facet_col],
    )
    records = []
    label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    y = pd.to_numeric(df[y_col], errors="coerce")

    for _, row in top.iterrows():
        feature_name = row["feature_name"]
        x = pd.to_numeric(df[feature_name], errors="coerce")
        presence = df[feature_name].notna().astype(float)
        corr = _spearman_without_scipy(x, y)
        bad_presence = float(presence[label == 1].mean()) if (label == 1).any() else np.nan
        good_presence = float(presence[label == 0].mean()) if (label == 0).any() else np.nan
        confounding_hint = _facet_concentration(df, feature_name, facet_col)
        records.append(
            {
                "order_id": order_id,
                "final_rank": row.get("final_rank"),
                "feature_name": feature_name,
                "final_score": row.get("final_score"),
                "direction": row.get("direction"),
                "presence_type": row.get("presence_type"),
                f"spearman_{feature_name}_vs_{y_col}": corr,
                "bad_presence_rate": bad_presence,
                "good_presence_rate": good_presence,
                f"{facet_col}_concentration_hint": confounding_hint,
                "evidence_reason": row.get("evidence_reason"),
            }
        )
    return pd.DataFrame(records)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local notebook env
        raise ImportError(
            "matplotlib is required for visual review. Install it with: "
            "python -m pip install matplotlib"
        ) from exc
    return plt


def _spearman_without_scipy(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    x_rank = frame["x"].rank(method="average")
    y_rank = frame["y"].rank(method="average")
    if x_rank.nunique() < 2 or y_rank.nunique() < 2:
        return np.nan
    return float(x_rank.corr(y_rank, method="pearson"))


def _title(feature_name: str, evidence_row: pd.Series | None) -> str:
    if evidence_row is None:
        return feature_name
    bits = [
        f"rank={evidence_row.get('final_rank', '')}",
        f"score={float(evidence_row.get('final_score', np.nan)):.3f}"
        if pd.notna(evidence_row.get("final_score", np.nan))
        else "",
        f"direction={evidence_row.get('direction', '')}",
        f"presence={evidence_row.get('presence_type', '')}",
    ]
    suffix = " | ".join(bit for bit in bits if bit)
    return f"{feature_name}\n{suffix}"


def _scatter_feature_y(ax, feature, numeric_feature, is_numeric, y, label, feature_name, y_col):
    colors = np.where(label == 1, "#d62728", "#1f77b4")
    if is_numeric:
        x = numeric_feature
        ax.scatter(x[label == 0], y[label == 0], s=20, alpha=0.55, c="#1f77b4", label="Good")
        ax.scatter(x[label == 1], y[label == 1], s=24, alpha=0.70, c="#d62728", label="Bad")
        ax.set_xlabel(feature_name)
    else:
        codes, uniques = pd.factorize(feature.astype("object").where(feature.notna(), "__MISSING__"))
        jitter = np.random.default_rng(42).normal(0, 0.04, size=len(codes))
        ax.scatter(codes + jitter, y, s=20, alpha=0.60, c=colors)
        tick_count = min(len(uniques), 8)
        ax.set_xticks(range(tick_count))
        ax.set_xticklabels([str(v)[:14] for v in uniques[:tick_count]], rotation=30, ha="right")
        ax.set_xlabel(feature_name)
    ax.set_ylabel(y_col)
    ax.set_title("feature vs y")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)


def _distribution_by_label(ax, feature, numeric_feature, is_numeric, label, feature_name):
    if is_numeric:
        good = numeric_feature[label == 0].dropna()
        bad = numeric_feature[label == 1].dropna()
        ax.boxplot([good, bad], showfliers=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Good", "Bad"])
        ax.set_ylabel(feature_name)
        ax.set_title("Good/Bad distribution")
    else:
        frame = pd.DataFrame({"feature": feature.astype("object").where(feature.notna(), "__MISSING__"), "label": label})
        summary = frame.groupby("feature", observed=True)["label"].agg(["mean", "count"]).sort_values("count", ascending=False).head(12)
        ax.bar(range(len(summary)), summary["mean"], color="#9467bd")
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels([str(v)[:14] for v in summary.index], rotation=30, ha="right")
        ax.set_ylabel("Bad rate")
        ax.set_title("Category bad rate")
    ax.grid(alpha=0.25)


def _facet_summary(ax, df, feature_name, numeric_feature, is_numeric, label, facet_col):
    if facet_col not in df.columns:
        ax.text(0.5, 0.5, f"{facet_col} missing", ha="center", va="center")
        ax.set_axis_off()
        return
    frame = pd.DataFrame(
        {
            "facet": df[facet_col].astype("object").where(df[facet_col].notna(), "__MISSING__"),
            "label": label,
            "feature": numeric_feature if is_numeric else df[feature_name].notna().astype(float),
        }
    )
    summary = frame.groupby("facet", observed=True).agg(feature_median=("feature", "median"), bad_rate=("label", "mean"), count=("label", "size"))
    summary = summary.sort_values("count", ascending=False).head(10)
    ax.bar(range(len(summary)), summary["feature_median"], color="#2ca02c", alpha=0.75, label="feature median")
    ax2 = ax.twinx()
    ax2.plot(range(len(summary)), summary["bad_rate"], color="#d62728", marker="o", label="bad rate")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([str(v)[:14] for v in summary.index], rotation=30, ha="right")
    ax.set_title(f"{facet_col} check")
    ax.set_ylabel("feature median")
    ax2.set_ylabel("bad rate")
    ax.grid(alpha=0.25)


def _presence_by_label(ax, feature, label, feature_name):
    presence = feature.notna().astype(float)
    values = [
        float(presence[label == 0].mean()) if (label == 0).any() else np.nan,
        float(presence[label == 1].mean()) if (label == 1).any() else np.nan,
    ]
    ax.bar(["Good", "Bad"], values, color=["#1f77b4", "#d62728"], alpha=0.75)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("non-null rate")
    ax.set_title("Presence / missingness")
    ax.grid(alpha=0.25)
    for idx, value in enumerate(values):
        if pd.notna(value):
            ax.text(idx, min(value + 0.03, 1.02), f"{value:.1%}", ha="center")


def _review_roles(df: pd.DataFrame, label_col: str, role_col: str) -> pd.Series:
    if role_col in df.columns:
        return df[role_col].astype("object").where(df[role_col].notna(), "Ignored")
    if label_col in df.columns:
        label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        return pd.Series(np.where(label == 1, "Bad", "Good"), index=df.index)
    return pd.Series("All", index=df.index)


def _scatter_by_role(ax, x: pd.Series, y: pd.Series, roles: pd.Series) -> None:
    styles = [
        ("Ignored", "#9e9e9e", 14, 0.25),
        ("Good", "#1f77b4", 20, 0.65),
        ("Bad", "#d62728", 24, 0.75),
        ("All", "#4c4c4c", 18, 0.55),
    ]
    plotted = set()
    for role, color, size, alpha in styles:
        mask = roles == role
        if mask.any():
            ax.scatter(x[mask], y[mask], s=size, alpha=alpha, c=color, label=role)
            plotted.add(role)
    for role in sorted(set(roles.dropna()) - plotted):
        mask = roles == role
        ax.scatter(x[mask], y[mask], s=18, alpha=0.45, label=str(role))


def _facet_concentration(df: pd.DataFrame, feature_name: str, facet_col: str) -> float:
    if facet_col not in df.columns:
        return np.nan
    available = df[df[feature_name].notna()]
    if available.empty:
        return np.nan
    shares = available[facet_col].value_counts(normalize=True, dropna=False)
    return float(shares.iloc[0]) if not shares.empty else np.nan
