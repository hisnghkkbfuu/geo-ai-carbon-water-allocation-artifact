from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text
from PIL import Image
from pypdf import PdfReader


MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 135

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESULTS_DIR = PROJECT_ROOT / "04_results"
POLICY_INPUT = RESULTS_DIR / "main_policy_metrics_v1.csv"
INTERVAL_INPUT = RESULTS_DIR / "main_policy_paired_block_intervals_v1.csv"
ZERO_EXPOSURE_INPUT = RESULTS_DIR / "high_stress_water_pareto_v1.csv"

SOURCE_CSV = HERE / "fig02_source_data_v1.csv"
SOURCE_MANIFEST = HERE / "fig02_source_manifest_v1.json"
OUTPUT_STEM = HERE / "fig02_equal_service_tradeoff_v1"
EXPORT_MANIFEST = HERE / "fig02_export_manifest_v1.json"

COLORS = {
    "ink": "#202124",
    "muted": "#62676B",
    "light": "#D6DADD",
    "reference": "#9AA0A4",
    "e0": "#6E7378",
    "e1": "#225E91",
    "e2": "#00857C",
    "stress": "#B44A3E",
    "stress_pale": "#F6E3E0",
    "blue_pale": "#E6EFF6",
    "teal_pale": "#E2F1EF",
    "white": "#FFFFFF",
}

POLICY_STYLE = {
    "E0": {"color": COLORS["e0"], "marker": "o", "hatch": ""},
    "E1": {"color": COLORS["e1"], "marker": "s", "hatch": "///"},
    "E2": {"color": COLORS["e2"], "marker": "D", "hatch": "xx"},
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one {label} row; found {len(selected)}")
    return selected.iloc[0]


def assert_close(label: str, actual: float, expected: float, atol: float = 1e-9) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise ValueError(f"Frozen-value drift for {label}: actual={actual}, expected={expected}")


def build_source_data() -> tuple[pd.DataFrame, dict[str, object]]:
    policy = pd.read_csv(POLICY_INPUT)
    intervals = pd.read_csv(INTERVAL_INPUT)
    zero_exposure = pd.read_csv(ZERO_EXPOSURE_INPUT)

    require_columns(
        policy,
        {
            "policy",
            "served_accelerator_hours",
            "migration_share",
            "carbon_kgco2e",
            "carbon_kgco2e_change_pct_vs_status_quo",
            "high_stress_direct_water_l",
        },
        "policy metrics",
    )
    require_columns(
        intervals,
        {
            "metric",
            "estimate",
            "block_interval_low_2_5pct",
            "block_interval_high_97_5pct",
            "replicates",
            "block_hours",
            "seed",
        },
        "paired block intervals",
    )
    require_columns(
        zero_exposure,
        {
            "policy",
            "served_accelerator_hours",
            "migration_share",
            "carbon_kgco2e",
            "high_stress_direct_water_l",
            "cap_fraction_of_carbon_first",
        },
        "zero-exposure operating points",
    )

    e0 = one_row(policy, policy["policy"].eq("B0_true_no_migration_status_quo"), "E0")
    e1 = one_row(policy, policy["policy"].eq("B1_carbon_first"), "E1")
    e2 = one_row(
        zero_exposure,
        zero_exposure["cap_fraction_of_carbon_first"].eq(0.0)
        & zero_exposure["high_stress_direct_water_l"].eq(0.0),
        "E2 zero-exposure endpoint",
    )
    carbon_interval = one_row(intervals, intervals["metric"].eq("carbon_change_pct"), "carbon interval")
    exposure_interval = one_row(
        intervals,
        intervals["metric"].eq("high_stress_water_change_l"),
        "high-stress exposure interval",
    )

    e0_carbon = float(e0["carbon_kgco2e"])
    e1_carbon = float(e1["carbon_kgco2e"])
    e2_carbon = float(e2["carbon_kgco2e"])
    e1_change = (e1_carbon / e0_carbon - 1.0) * 100.0
    e2_change = (e2_carbon / e0_carbon - 1.0) * 100.0
    e2_vs_e1 = (e2_carbon / e1_carbon - 1.0) * 100.0
    reduction_penalty_pp = e2_change - e1_change

    assert_close("E1/E0 carbon change", e1_change, -13.235121293717944)
    assert_close("E2/E0 carbon change", e2_change, -13.083310808067905)
    assert_close("E2/E1 carbon change", e2_vs_e1, 0.1749676688466817)
    assert_close("E2 reduction penalty", reduction_penalty_pp, 0.15181048565003863)
    assert_close("E1 exposure", float(e1["high_stress_direct_water_l"]), 2.4250446709410443)
    assert_close("E2 exposure", float(e2["high_stress_direct_water_l"]), 0.0)
    assert_close("carbon interval estimate", float(carbon_interval["estimate"]), e1_change)
    assert_close("exposure interval estimate", float(exposure_interval["estimate"]), float(e1["high_stress_direct_water_l"]))

    served = {
        "E0": float(e0["served_accelerator_hours"]),
        "E1": float(e1["served_accelerator_hours"]),
        "E2": float(e2["served_accelerator_hours"]),
    }
    service_range = max(served.values()) - min(served.values())
    if service_range > 1e-8:
        raise ValueError(f"Equal-service check failed: range={service_range}")

    interval_meta = {
        "replicates": int(carbon_interval["replicates"]),
        "block_hours": int(carbon_interval["block_hours"]),
        "seed": int(carbon_interval["seed"]),
    }
    if interval_meta != {"replicates": 4000, "block_hours": 24, "seed": 20260719}:
        raise ValueError(f"Unexpected interval metadata: {interval_meta}")
    if any(int(exposure_interval[key]) != value for key, value in interval_meta.items()):
        raise ValueError("Carbon and exposure interval metadata do not match")

    definitions = {
        "E0": "strict no migration",
        "E1": "30% migration; latency ≤ 20 ms",
        "E2": "E1 plus zero high-stress direct-water cap",
    }
    point_values = {
        "E0": {
            "carbon_change_pct": 0.0,
            "exposure_l": float(e0["high_stress_direct_water_l"]),
            "migration_pct": float(e0["migration_share"]) * 100.0,
            "served": served["E0"],
        },
        "E1": {
            "carbon_change_pct": e1_change,
            "exposure_l": float(e1["high_stress_direct_water_l"]),
            "migration_pct": float(e1["migration_share"]) * 100.0,
            "served": served["E1"],
        },
        "E2": {
            "carbon_change_pct": e2_change,
            "exposure_l": float(e2["high_stress_direct_water_l"]),
            "migration_pct": float(e2["migration_share"]) * 100.0,
            "served": served["E2"],
        },
    }

    rows: list[dict[str, object]] = []
    for policy_name in ("E0", "E1", "E2"):
        values = point_values[policy_name]
        rows.append(
            {
                "panel": "a",
                "row_id": f"a_{policy_name.lower()}",
                "record_type": "reference_operating_point",
                "policy": policy_name,
                "policy_definition": definitions[policy_name],
                "carbon_change_pct": values["carbon_change_pct"],
                "high_stress_direct_water_l": values["exposure_l"],
                "migration_share_pct": values["migration_pct"],
                "served_accelerator_hours": values["served"],
                "interval_low": None,
                "interval_high": None,
                "interval_unit": "",
                "replicates": None,
                "block_hours": None,
                "seed": None,
                "uncertainty_status": "point estimate",
                "evidence_id": "T3C01;T3C02;T3C03",
            }
        )

    for policy_name in ("E1", "E2"):
        values = point_values[policy_name]
        is_e1 = policy_name == "E1"
        rows.append(
            {
                "panel": "b",
                "row_id": f"b_{policy_name.lower()}",
                "record_type": "carbon_direction_interval" if is_e1 else "carbon_point_only",
                "policy": policy_name,
                "policy_definition": definitions[policy_name],
                "carbon_change_pct": values["carbon_change_pct"],
                "high_stress_direct_water_l": values["exposure_l"],
                "migration_share_pct": values["migration_pct"],
                "served_accelerator_hours": values["served"],
                "interval_low": float(carbon_interval["block_interval_low_2_5pct"]) if is_e1 else None,
                "interval_high": float(carbon_interval["block_interval_high_97_5pct"]) if is_e1 else None,
                "interval_unit": "% vs E0",
                "replicates": interval_meta["replicates"] if is_e1 else None,
                "block_hours": interval_meta["block_hours"] if is_e1 else None,
                "seed": interval_meta["seed"] if is_e1 else None,
                "uncertainty_status": "paired circular-block interval" if is_e1 else "point estimate; no primary block interval",
                "evidence_id": "T3C01;T3C03",
            }
        )

    for policy_name in ("E1", "E2"):
        values = point_values[policy_name]
        is_e1 = policy_name == "E1"
        rows.append(
            {
                "panel": "c",
                "row_id": f"c_{policy_name.lower()}",
                "record_type": "exposure_direction_interval" if is_e1 else "exposure_fixed_zero",
                "policy": policy_name,
                "policy_definition": definitions[policy_name],
                "carbon_change_pct": values["carbon_change_pct"],
                "high_stress_direct_water_l": values["exposure_l"],
                "migration_share_pct": values["migration_pct"],
                "served_accelerator_hours": values["served"],
                "interval_low": float(exposure_interval["block_interval_low_2_5pct"]) if is_e1 else None,
                "interval_high": float(exposure_interval["block_interval_high_97_5pct"]) if is_e1 else None,
                "interval_unit": "L vs E0",
                "replicates": interval_meta["replicates"] if is_e1 else None,
                "block_hours": interval_meta["block_hours"] if is_e1 else None,
                "seed": interval_meta["seed"] if is_e1 else None,
                "uncertainty_status": "paired circular-block interval" if is_e1 else "fixed at zero by construction",
                "evidence_id": "T3C02;T3C03",
            }
        )

    for policy_name in ("E0", "E1", "E2"):
        values = point_values[policy_name]
        rows.append(
            {
                "panel": "d",
                "row_id": f"d_{policy_name.lower()}",
                "record_type": "equal_service_migration_control",
                "policy": policy_name,
                "policy_definition": definitions[policy_name],
                "carbon_change_pct": values["carbon_change_pct"],
                "high_stress_direct_water_l": values["exposure_l"],
                "migration_share_pct": values["migration_pct"],
                "served_accelerator_hours": values["served"],
                "interval_low": None,
                "interval_high": None,
                "interval_unit": "",
                "replicates": None,
                "block_hours": None,
                "seed": None,
                "uncertainty_status": "deterministic policy result",
                "evidence_id": "T3C01;T3C03",
            }
        )

    source = pd.DataFrame(rows)
    source.to_csv(SOURCE_CSV, index=False, float_format="%.15g")

    scientific_summary = {
        "e1_carbon_change_pct": e1_change,
        "e1_carbon_interval_pct": [
            float(carbon_interval["block_interval_low_2_5pct"]),
            float(carbon_interval["block_interval_high_97_5pct"]),
        ],
        "e1_exposure_change_l": float(e1["high_stress_direct_water_l"]),
        "e1_exposure_interval_l": [
            float(exposure_interval["block_interval_low_2_5pct"]),
            float(exposure_interval["block_interval_high_97_5pct"]),
        ],
        "e2_carbon_change_pct": e2_change,
        "e2_vs_e1_carbon_pct": e2_vs_e1,
        "carbon_reduction_penalty_pp": reduction_penalty_pp,
        "e2_exposure_l": float(e2["high_stress_direct_water_l"]),
        "served_accelerator_hours": served,
        "max_served_difference_accelerator_hours": service_range,
        "migration_share_pct": {name: values["migration_pct"] for name, values in point_values.items()},
        "interval_metadata": interval_meta,
    }
    manifest = {
        "figure": "Fig. 2",
        "derived_source": {
            "path": SOURCE_CSV.name,
            "rows": int(len(source)),
            "panel_counts": {key: int(value) for key, value in source.groupby("panel").size().to_dict().items()},
            "sha256": sha256(SOURCE_CSV),
        },
        "upstream_inputs": [
            {"role": "frozen policy metrics", "rows": int(len(policy)), "sha256": sha256(POLICY_INPUT)},
            {"role": "paired circular-block intervals", "rows": int(len(intervals)), "sha256": sha256(INTERVAL_INPUT)},
            {"role": "zero-exposure operating points", "rows": int(len(zero_exposure)), "sha256": sha256(ZERO_EXPOSURE_INPUT)},
        ],
        "derivation": {
            "E0": "strict no-migration row",
            "E1": "carbon-first row",
            "E2": "zero high-stress direct-water endpoint",
            "carbon_change_pct": "100 × (policy carbon / E0 carbon − 1)",
            "carbon_reduction_penalty_pp": "E2/E0 percentage change minus E1/E0 percentage change",
        },
        "scientific_summary": scientific_summary,
        "assertions": {
            "frozen_values_match": True,
            "equal_service_within_1e-8_accelerator_hours": service_range <= 1e-8,
            "E0_migration_is_zero": bool(np.isclose(point_values["E0"]["migration_pct"], 0.0)),
            "E1_and_E2_migration_are_30_pct": bool(
                np.isclose(point_values["E1"]["migration_pct"], 30.0)
                and np.isclose(point_values["E2"]["migration_pct"], 30.0)
            ),
            "E2_high_stress_exposure_is_zero": bool(np.isclose(point_values["E2"]["exposure_l"], 0.0)),
            "E1_carbon_interval_upper_below_zero": float(carbon_interval["block_interval_high_97_5pct"]) < 0.0,
            "E1_exposure_interval_lower_above_zero": float(exposure_interval["block_interval_low_2_5pct"]) > 0.0,
            "E2_has_no_inferred_primary_interval": True,
        },
    }
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return source, scientific_summary


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(direction="out", colors=COLORS["ink"], pad=2)
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])


def panel_heading(ax: plt.Axes, letter: str, title: str, x_letter: float = -0.12) -> None:
    ax.text(
        x_letter,
        1.025,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
        clip_on=False,
    )
    ax.text(
        0.0,
        1.025,
        title,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
        clip_on=False,
    )


def plot_policy_point(
    ax: plt.Axes,
    policy: str,
    x: float,
    y: float,
    *,
    open_marker: bool = False,
    size: float = 52,
    zorder: int = 5,
) -> None:
    style = POLICY_STYLE[policy]
    ax.scatter(
        [x],
        [y],
        s=size,
        marker=style["marker"],
        facecolor=COLORS["white"] if open_marker else style["color"],
        edgecolor=style["color"],
        linewidth=1.25,
        zorder=zorder,
        clip_on=False,
    )


def draw_panel_a(ax: plt.Axes, source: pd.DataFrame, summary: dict[str, object]) -> None:
    data = source.loc[source["panel"].eq("a")].set_index("policy")
    panel_heading(ax, "a", "Carbon–exposure operating space", x_letter=-0.10)
    style_axis(ax)
    ax.set_xlim(-14.25, 0.80)
    ax.set_ylim(-0.18, 2.88)
    ax.set_xticks([-14, -10, -6, -2, 0])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("Carbon change relative to E0 (%)", labelpad=5)
    ax.set_ylabel("High-stress direct water (L)", labelpad=5)
    ax.axhline(0, color=COLORS["reference"], linewidth=0.75, linestyle=(0, (3, 2)), zorder=0)
    ax.axvline(0, color=COLORS["reference"], linewidth=0.75, linestyle=(0, (3, 2)), zorder=0)

    points = {
        policy: (
            float(data.loc[policy, "carbon_change_pct"]),
            float(data.loc[policy, "high_stress_direct_water_l"]),
        )
        for policy in ("E0", "E1", "E2")
    }
    ax.add_patch(
        FancyArrowPatch(
            points["E0"],
            points["E1"],
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.15,
            color=COLORS["e1"],
            shrinkA=7,
            shrinkB=8,
            connectionstyle="arc3,rad=0.02",
            zorder=2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            points["E1"],
            points["E2"],
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.15,
            color=COLORS["e2"],
            shrinkA=7,
            shrinkB=8,
            connectionstyle="arc3,rad=-0.05",
            zorder=2,
        )
    )

    for policy in ("E0", "E1", "E2"):
        plot_policy_point(ax, policy, *points[policy], size=70)

    ax.text(-0.30, 0.16, "E0\nno migration", ha="right", va="bottom", fontsize=6.5, fontweight="bold", color=COLORS["e0"])
    ax.text(-13.92, 2.56, "E1  carbon first", ha="left", va="bottom", fontsize=6.5, fontweight="bold", color=COLORS["e1"])
    ax.text(-12.72, 0.16, "E2  zero exposure", ha="left", va="bottom", fontsize=6.5, fontweight="bold", color=COLORS["e2"])
    ax.text(-6.4, 1.42, "carbon-first\nallocation", ha="center", va="center", fontsize=6.2, color=COLORS["e1"], rotation=-9)
    ax.text(-12.45, 1.28, "zero-exposure\nconstraint", ha="left", va="center", fontsize=6.2, color=COLORS["e2"])

    penalty = float(summary["carbon_reduction_penalty_pp"])
    ax.text(
        -6.1,
        2.47,
        f"E2 vs E1\n{penalty:.3f}-pp carbon-reduction penalty",
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": COLORS["teal_pale"], "edgecolor": COLORS["e2"], "linewidth": 0.7},
        zorder=4,
    )
    ax.text(
        0.01,
        0.99,
        "Three reference operating points—not a continuous frontier",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color=COLORS["muted"],
    )


def draw_interval_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    panel: str,
    title: str,
    x_label: str,
    xlim: tuple[float, float],
    xticks: list[float],
    value_column: str,
    signal_color: str,
    show_zero_line: bool,
) -> None:
    data = source.loc[source["panel"].eq(panel)].set_index("policy")
    panel_heading(ax, panel, title, x_letter=-0.16)
    style_axis(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, 1.48)
    ax.set_xticks(xticks)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["E1", "E2"])
    ax.set_xlabel(x_label, labelpad=4)
    if show_zero_line:
        ax.axvline(0, color=COLORS["reference"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)

    e1 = data.loc["E1"]
    estimate = float(e1[value_column])
    low = float(e1["interval_low"])
    high = float(e1["interval_high"])
    ax.hlines(1, low, high, color=signal_color, linewidth=1.5, zorder=2)
    ax.vlines([low, high], 0.91, 1.09, color=signal_color, linewidth=0.9, zorder=2)
    plot_policy_point(ax, "E1", estimate, 1, size=42)

    e2_value = float(data.loc["E2", value_column])
    plot_policy_point(ax, "E2", e2_value, 0, open_marker=True, size=44)

    if panel == "b":
        ax.text(estimate, 1.24, f"{estimate:.3f}%", ha="center", va="bottom", fontsize=6.1, color=COLORS["e1"])
        ax.text(e2_value, 0.22, f"{e2_value:.3f}%", ha="center", va="bottom", fontsize=6.1, color=COLORS["e2"])
        ax.text(0.98, 0.08, "open diamond: point only", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.7, color=COLORS["muted"])
    else:
        ax.text(estimate, 1.24, f"+{estimate:.3f} L", ha="center", va="bottom", fontsize=6.1, color=COLORS["stress"])
        ax.text(0.08, 0.20, "fixed zero", ha="left", va="bottom", fontsize=5.9, color=COLORS["e2"])
    ax.text(
        0.98,
        0.96,
        "4,000 paired 24-h\ncircular-block resamples",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=COLORS["muted"],
        linespacing=1.1,
    )


def draw_panel_d(ax: plt.Axes, source: pd.DataFrame) -> None:
    data = source.loc[source["panel"].eq("d")].set_index("policy").loc[["E0", "E1", "E2"]]
    panel_heading(ax, "d", "Policy migration share", x_letter=-0.16)
    style_axis(ax)
    x = np.arange(3)
    values = data["migration_share_pct"].astype(float).to_numpy()
    bars = ax.bar(
        x,
        values,
        width=0.54,
        color=[COLORS["white"], COLORS["blue_pale"], COLORS["teal_pale"]],
        edgecolor=[POLICY_STYLE[name]["color"] for name in ("E0", "E1", "E2")],
        linewidth=0.9,
        zorder=1,
    )
    for bar, policy in zip(bars, ("E0", "E1", "E2")):
        bar.set_hatch(POLICY_STYLE[policy]["hatch"])
    for xi, value, policy in zip(x, values, ("E0", "E1", "E2")):
        plot_policy_point(ax, policy, float(xi), float(value), size=38, zorder=4)
        ax.text(xi, value + (1.6 if value > 0 else 1.8), f"{value:.0f}%", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color=POLICY_STYLE[policy]["color"])

    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-1.5, 46.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["E0", "E1", "E2"])
    ax.set_yticks([0, 15, 30])
    ax.set_ylabel("Migration share (%)", labelpad=4)
    ax.axhline(0, color=COLORS["reference"], linewidth=0.75, zorder=0)
    served = data["served_accelerator_hours"].astype(float)
    ax.text(
        0.52,
        0.95,
        f"All policies serve {served.mean():.3f} accelerator-h",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.7,
        color=COLORS["ink"],
    )


def build_figure(source: pd.DataFrame, summary: dict[str, object]) -> plt.Figure:
    figure = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH))
    grid = figure.add_gridspec(3, 2, width_ratios=[1.50, 1.0], height_ratios=[1.0, 1.0, 0.90])
    ax_a = figure.add_subplot(grid[:, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 1])
    ax_d = figure.add_subplot(grid[2, 1])

    draw_panel_a(ax_a, source, summary)
    draw_interval_panel(
        ax_b,
        source,
        "b",
        "Carbon direction",
        "Carbon change vs E0 (%)",
        (-14.15, -12.15),
        [-14.0, -13.5, -13.0, -12.5],
        "carbon_change_pct",
        COLORS["e1"],
        False,
    )
    draw_interval_panel(
        ax_c,
        source,
        "c",
        "Exposure direction",
        "High-stress direct-water change (L)",
        (-0.25, 5.15),
        [0, 2, 4],
        "high_stress_direct_water_l",
        COLORS["stress"],
        True,
    )
    draw_panel_d(ax_d, source)

    figure.subplots_adjust(left=0.075, right=0.982, bottom=0.095, top=0.925, wspace=0.42, hspace=0.72)
    return figure


def figure_layout_metadata(figure: plt.Figure) -> dict[str, object]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    outside: list[str] = []
    for artist in figure.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.width == 0 or bbox.height == 0:
            continue
        if bbox.x0 < canvas.x0 - 0.5 or bbox.y0 < canvas.y0 - 0.5 or bbox.x1 > canvas.x1 + 0.5 or bbox.y1 > canvas.y1 + 0.5:
            outside.append(artist.get_text())
    return {
        "canvas_width_px_at_render_dpi": round(float(canvas.width), 3),
        "canvas_height_px_at_render_dpi": round(float(canvas.height), 3),
        "visible_text_artist_count": sum(1 for artist in figure.findobj(match=Text) if artist.get_visible() and artist.get_text().strip()),
        "text_outside_canvas_count": len(outside),
        "text_outside_canvas": outside,
    }


def svg_metadata(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    text_nodes = sum(1 for node in root.iter() if node.tag.endswith("text"))
    return {
        "width": root.attrib.get("width"),
        "height": root.attrib.get("height"),
        "viewBox": root.attrib.get("viewBox"),
        "text_node_count": text_nodes,
    }


def pdf_metadata(path: Path) -> dict[str, object]:
    reader = PdfReader(str(path))
    page = reader.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    fonts = page["/Resources"].get("/Font", {})
    font_records = []
    for key, value in fonts.items():
        font = value.get_object()
        font_records.append(
            {
                "resource": str(key),
                "subtype": str(font.get("/Subtype")),
                "basefont": str(font.get("/BaseFont")),
            }
        )
    return {
        "width_mm": round(width_pt / 72 * MM_PER_INCH, 3),
        "height_mm": round(height_pt / 72 * MM_PER_INCH, 3),
        "fonts": font_records,
        "type3_font_count": sum(record["subtype"] == "/Type3" for record in font_records),
    }


def raster_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        return {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "format": image.format,
            "dpi": [round(float(value), 3) for value in image.info.get("dpi", (0, 0))],
            "compression": image.info.get("compression"),
        }


def export_figure(figure: plt.Figure, source: pd.DataFrame, summary: dict[str, object]) -> None:
    layout = figure_layout_metadata(figure)
    outputs = {
        "svg": OUTPUT_STEM.with_suffix(".svg"),
        "pdf": OUTPUT_STEM.with_suffix(".pdf"),
        "tiff": OUTPUT_STEM.with_suffix(".tiff"),
        "png": OUTPUT_STEM.with_suffix(".png"),
    }
    figure.savefig(outputs["svg"], format="svg")
    figure.savefig(outputs["pdf"], format="pdf", metadata={"Title": "Fig. 2 | Equal-service carbon and high-stress direct-water operating points"})
    figure.savefig(outputs["tiff"], format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(outputs["png"], format="png", dpi=300)
    plt.close(figure)

    file_manifest = {
        name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in outputs.items()
    }
    file_manifest["svg"]["metadata"] = svg_metadata(outputs["svg"])
    file_manifest["pdf"]["metadata"] = pdf_metadata(outputs["pdf"])
    file_manifest["tiff"]["metadata"] = raster_metadata(outputs["tiff"])
    file_manifest["png"]["metadata"] = raster_metadata(outputs["png"])

    pdf_meta = file_manifest["pdf"]["metadata"]
    tiff_meta = file_manifest["tiff"]["metadata"]
    png_meta = file_manifest["png"]["metadata"]
    manifest = {
        "figure": "Fig. 2",
        "title": "Equal-service carbon and high-stress direct-water operating points",
        "backend": "Python / matplotlib Agg",
        "intended_size_mm": {"width": FIGURE_WIDTH_MM, "height": FIGURE_HEIGHT_MM},
        "source_data": {
            "path": SOURCE_CSV.name,
            "rows": int(len(source)),
            "panel_counts": {key: int(value) for key, value in source.groupby("panel").size().to_dict().items()},
            "sha256": sha256(SOURCE_CSV),
        },
        "scientific_summary": summary,
        "layout": layout,
        "outputs": file_manifest,
        "qa_contract": {
            "svg_editable_text": file_manifest["svg"]["metadata"]["text_node_count"] > 0,
            "pdf_size_matches_mm": abs(pdf_meta["width_mm"] - FIGURE_WIDTH_MM) < 0.02
            and abs(pdf_meta["height_mm"] - FIGURE_HEIGHT_MM) < 0.02,
            "pdf_has_no_type3_fonts": pdf_meta["type3_font_count"] == 0,
            "tiff_nominal_600_dpi": all(abs(value - 600) < 1 for value in tiff_meta["dpi"]),
            "tiff_lzw_compression": tiff_meta["compression"] == "tiff_lzw",
            "png_nominal_300_dpi": all(abs(value - 300) < 1 for value in png_meta["dpi"]),
            "no_text_outside_canvas": layout["text_outside_canvas_count"] == 0,
            "source_has_expected_10_rows": len(source) == 10,
        },
    }
    EXPORT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    source, summary = build_source_data()
    export_figure(build_figure(source, summary), source, summary)
    print(f"Generated {SOURCE_CSV.name} from frozen inputs.")
    print(f"Generated {OUTPUT_STEM.name} in SVG, PDF, TIFF, and PNG formats.")
    print(f"Source manifest: {SOURCE_MANIFEST.name}")
    print(f"Export manifest: {EXPORT_MANIFEST.name}")


if __name__ == "__main__":
    main()
