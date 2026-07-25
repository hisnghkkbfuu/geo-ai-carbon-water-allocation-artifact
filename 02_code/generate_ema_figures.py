from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "04_figures"
SOURCE_DIR = FIGURE_DIR / "source_data"
EXPORT_DIR = FIGURE_DIR / "exports"

GRAPHITE = "#5F6B73"
TEAL = "#00857C"
BLUE = "#2F6B9A"
RUST = "#B35C44"
GREEN = "#5A8F6A"
PALE_BLUE = "#D9E7F2"
PALE_GREEN = "#DCECDF"
PALE_RUST = "#F1DDD6"
GRID = "#D5D9DC"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "svg.fonttype": "none",
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


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.15,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def apply_grid(axis: plt.Axes, *, y: bool = True) -> None:
    axis.grid(axis="y" if y else "both", color=GRID, linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)


def add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = GRAPHITE,
    fontsize: float = 6.6,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.75,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def add_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = GRAPHITE) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=axis.transAxes,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.8,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    axis.add_patch(arrow)


def figure_1() -> plt.Figure:
    fig = plt.figure(figsize=(7.09, 4.80))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 0.88], height_ratios=[1.0, 1.0], wspace=0.32, hspace=0.45)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    for axis in (ax_a, ax_b, ax_c):
        axis.set_axis_off()

    add_box(ax_a, 0.03, 0.78, 0.27, 0.14, "Hourly service\nproxy", facecolor=PALE_BLUE)
    add_box(ax_a, 0.03, 0.52, 0.27, 0.14, "Regional scenario\nlayers", facecolor="#F3F5F6")
    add_box(ax_a, 0.38, 0.62, 0.27, 0.18, "Feasible hourly\nallocation", facecolor="#E9F2F1", edgecolor=TEAL)
    add_box(ax_a, 0.73, 0.78, 0.22, 0.14, "Carbon", facecolor="#E8F0F4", edgecolor=BLUE)
    add_box(ax_a, 0.73, 0.52, 0.22, 0.14, "Direct water", facecolor="#EAF2EC", edgecolor=GREEN)
    add_box(ax_a, 0.38, 0.25, 0.57, 0.17, "Trace, proxy, and scenario inputs remain distinct", facecolor="#FFF9F2", edgecolor="#C79A59")
    add_arrow(ax_a, (0.30, 0.85), (0.38, 0.72))
    add_arrow(ax_a, (0.30, 0.59), (0.38, 0.70))
    add_arrow(ax_a, (0.65, 0.72), (0.73, 0.85), color=BLUE)
    add_arrow(ax_a, (0.65, 0.70), (0.73, 0.59), color=GREEN)
    ax_a.text(0.03, 0.97, "Inputs and environmental outputs", transform=ax_a.transAxes, fontweight="bold", va="top")
    ax_a.text(0.03, 0.08, "No named facility measurement or observed placement is assumed.", transform=ax_a.transAxes, fontsize=6.2, color=GRAPHITE)
    panel_label(ax_a, "a")

    add_box(ax_b, 0.02, 0.60, 0.26, 0.20, "Assigned\nservice", facecolor=PALE_BLUE)
    add_box(ax_b, 0.38, 0.60, 0.25, 0.20, "IT energy", facecolor="#F3F5F6")
    add_box(ax_b, 0.72, 0.72, 0.25, 0.18, "Facility energy\nthen carbon", facecolor="#E8F0F4", edgecolor=BLUE)
    add_box(ax_b, 0.72, 0.40, 0.25, 0.18, "Direct water\nthen stress screen", facecolor="#EAF2EC", edgecolor=GREEN)
    add_arrow(ax_b, (0.28, 0.70), (0.38, 0.70))
    add_arrow(ax_b, (0.63, 0.70), (0.72, 0.81), color=BLUE)
    add_arrow(ax_b, (0.63, 0.68), (0.72, 0.49), color=GREEN)
    ax_b.text(0.72, 0.26, "PUE enters carbon;\nWUE uses IT energy only.", transform=ax_b.transAxes, ha="center", fontsize=6.2, color=GRAPHITE)
    ax_b.text(0.02, 0.97, "Physical accounting", transform=ax_b.transAxes, fontweight="bold", va="top")
    panel_label(ax_b, "b")

    add_box(ax_c, 0.04, 0.61, 0.25, 0.23, "Stage 1\nminimize\nunserved service", facecolor=PALE_BLUE, fontsize=5.4)
    add_box(ax_c, 0.39, 0.61, 0.25, 0.23, "Fix maximum\nfeasible service", facecolor="#F3F5F6", fontsize=5.5)
    add_box(ax_c, 0.74, 0.61, 0.22, 0.23, "Stage 2\nminimize\ncarbon", facecolor="#E9F2F1", edgecolor=TEAL, fontsize=5.4)
    add_box(ax_c, 0.23, 0.22, 0.56, 0.20, "Epsilon constraint\nscreened direct-water cap", facecolor=PALE_RUST, edgecolor=RUST, fontsize=5.6)
    add_arrow(ax_c, (0.29, 0.725), (0.39, 0.725))
    add_arrow(ax_c, (0.64, 0.725), (0.74, 0.725), color=TEAL)
    add_arrow(ax_c, (0.85, 0.61), (0.70, 0.42), color=RUST)
    ax_c.text(0.02, 0.97, "Service-preserving environmental decision", transform=ax_c.transAxes, fontweight="bold", va="top")
    ax_c.text(0.05, 0.08, "No cross-hour delay, causal estimate, or hydrologic damage model.", transform=ax_c.transAxes, fontsize=6.1, color=GRAPHITE)
    panel_label(ax_c, "c")

    return fig


def figure_2(source_dir: Path) -> plt.Figure:
    frontier = pd.read_csv(source_dir / "fig02_reference_frontier_source_v1.csv")
    segments = pd.read_csv(source_dir / "fig02_reference_segment_source_v1.csv")
    allocation = pd.read_csv(source_dir / "fig02_selected_allocations_source_v1.csv")
    fig = plt.figure(figsize=(7.09, 5.25))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.05, 0.90], height_ratios=[1.05, 0.95], wspace=0.52, hspace=0.58)
    ax_a = fig.add_subplot(grid[0, :2])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1:])

    x = frontier["screened_direct_water_avoided_l_vs_carbon_first"]
    y = frontier["additional_carbon_kgco2e_vs_carbon_first"]
    ax_a.plot(x, y, color=BLUE, linewidth=1.35, zorder=2)
    ax_a.scatter(x, y, facecolor="white", edgecolor=BLUE, linewidth=0.9, s=20, zorder=3)
    for fraction, label, offset in [(0.0, "0% allowed", (-58, 5)), (0.5, "50% allowed", (5, -10)), (1.0, "100% allowed", (5, 7))]:
        row = frontier.loc[frontier["cap_fraction_of_carbon_first"].eq(fraction)].iloc[0]
        ax_a.annotate(
            label,
            (row["screened_direct_water_avoided_l_vs_carbon_first"], row["additional_carbon_kgco2e_vs_carbon_first"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.2,
            color=BLUE,
        )
    ax_a.set_xlabel("Screened direct water avoided vs carbon-first (L)")
    ax_a.set_ylabel("Additional model-accounted carbon (kgCO2e)")
    ax_a.set_title("Reference screened-water frontier", loc="left", fontweight="bold")
    apply_grid(ax_a)
    panel_label(ax_a, "a")

    ax_b.plot(
        100 * frontier["cap_fraction_of_carbon_first"],
        frontier["carbon_change_pct_vs_status_quo"],
        color=TEAL,
        marker="o",
        markersize=3.2,
        linewidth=1.15,
    )
    ax_b.axhline(0.0, color=GRAPHITE, linewidth=0.75, linestyle="--")
    ax_b.set_xlabel("Allowed screened water\n(% of carbon-first)")
    ax_b.set_ylabel("Carbon change vs no migration (%)")
    ax_b.set_xlim(-2, 102)
    ax_b.set_title("Carbon direction", loc="left", fontweight="bold")
    apply_grid(ax_b)
    panel_label(ax_b, "b")

    midpoint = 50.0 * (segments["from_cap_fraction"] + segments["to_cap_fraction"])
    ax_c.plot(
        midpoint,
        segments["marginal_additional_carbon_kgco2e_per_screened_l_avoided"],
        color=RUST,
        marker="s",
        markersize=3,
        linewidth=1.1,
    )
    ax_c.set_xlabel("Allowed screened water (%)")
    ax_c.set_ylabel("Marginal carbon cost\n(kgCO2e per L avoided)")
    ax_c.set_xlim(-2, 102)
    ax_c.set_title("Adjacent cap segments", loc="left", fontweight="bold")
    apply_grid(ax_c)
    panel_label(ax_c, "c")

    regions = ["east", "north", "northwest", "southwest"]
    colors = [GREEN, RUST, "#D18A75", "#6C9A8D"]
    hatches = ["", "///", "xxx", ""]
    labels = ["east (not screened)", "north (screened)", "northwest (screened)", "southwest (not screened)"]
    positions = np.arange(len(allocation))
    left = np.zeros(len(allocation))
    for region, color, hatch, label in zip(regions, colors, hatches, labels, strict=True):
        values = allocation[f"assigned_{region}_accelerator_hours"].to_numpy()
        bars = ax_d.barh(positions, values, left=left, color=color, edgecolor="white", linewidth=0.5, hatch=hatch, label=label)
        left += values
    ax_d.set_yticks(positions, [f"{int(100 * value)}% allowed" for value in allocation["cap_fraction_of_carbon_first"]])
    ax_d.set_xlabel("Assigned accelerator-hour-equivalent service")
    ax_d.set_title("Regional allocation at three cap levels", loc="left", fontweight="bold")
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.50, -0.34), ncol=2, columnspacing=0.8, handlelength=1.4)
    apply_grid(ax_d)
    panel_label(ax_d, "d")
    return fig


def _deterministic_range_plot(axis: plt.Axes, summary: pd.DataFrame, *, color: str, ylabel: str, title: str) -> None:
    x = 100 * summary["cap_fraction_of_carbon_first"].to_numpy()
    lower = summary["carbon_change_pct_vs_status_quo_minimum"].to_numpy()
    q25 = summary["carbon_change_pct_vs_status_quo_q25"].to_numpy()
    median = summary["carbon_change_pct_vs_status_quo_median"].to_numpy()
    q75 = summary["carbon_change_pct_vs_status_quo_q75"].to_numpy()
    upper = summary["carbon_change_pct_vs_status_quo_maximum"].to_numpy()
    axis.vlines(x, lower, upper, color=color, linewidth=1.0, alpha=0.8, zorder=2)
    axis.vlines(x, q25, q75, color=color, linewidth=4.0, zorder=3)
    axis.scatter(x, median, color="white", edgecolor=color, linewidth=1.0, s=22, zorder=4)
    axis.axhline(0.0, color=GRAPHITE, linewidth=0.75, linestyle="--")
    axis.set_xlim(-4, 104)
    axis.set_xlabel("Allowed screened water (% of carbon-first)")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    apply_grid(axis)


def figure_3(source_dir: Path) -> plt.Figure:
    summary = pd.read_csv(source_dir / "fig03_scenario_cap_summary_source_v1.csv")
    zero_cap = pd.read_csv(source_dir / "fig03_zero_cap_scenarios_source_v1.csv")
    segments = pd.read_csv(source_dir / "fig03_scenario_segment_source_v1.csv")
    qa = pd.read_csv(source_dir / "fig03_solver_qa_source_v1.csv")
    fig = plt.figure(figsize=(7.09, 5.25))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.23, 0.77], height_ratios=[1.0, 1.0], wspace=0.50, hspace=0.60)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _deterministic_range_plot(
        ax_a,
        summary,
        color=TEAL,
        ylabel="Carbon change vs no migration (%)",
        title="243-cell deterministic scenario envelope",
    )
    ax_a.text(
        0.03,
        0.06,
        "thin: min-max; thick: Q25-Q75; open circle: median\nComplete enumeration, not a confidence interval",
        transform=ax_a.transAxes,
        fontsize=5.8,
        color=GRAPHITE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        zorder=6,
    )
    panel_label(ax_a, "a")

    pue_order = ["hub_policy_compliant", "uniform_new_build", "legacy_stress"]
    display_names = ["hub policy", "uniform new build", "legacy stress"]
    rng = np.random.default_rng(20260725)
    for pos, pue in enumerate(pue_order):
        values = zero_cap.loc[zero_cap["pue_scenario"].eq(pue), "carbon_change_pct_vs_status_quo"].to_numpy()
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax_b.scatter(np.full(len(values), pos) + jitter, values, s=9, color=BLUE, alpha=0.42, linewidth=0)
        ax_b.hlines(np.median(values), pos - 0.22, pos + 0.22, color=RUST, linewidth=1.3)
    ax_b.axhline(0.0, color=GRAPHITE, linewidth=0.75, linestyle="--")
    ax_b.set_xticks(range(3), display_names, rotation=18, ha="right")
    ax_b.set_ylabel("Zero-cap carbon change (%)")
    ax_b.set_title("Zero-cap cells by PUE scenario", loc="left", fontweight="bold")
    apply_grid(ax_b)
    panel_label(ax_b, "b")

    valid = segments.dropna(subset=["marginal_additional_carbon_kgco2e_per_screened_l_avoided"]).copy()
    valid["transition"] = (100 * valid["from_cap_fraction"]).round().astype(int).astype(str) + " to " + (100 * valid["to_cap_fraction"]).round().astype(int).astype(str)
    transition_order = ["100 to 75", "75 to 50", "50 to 25", "25 to 0"]
    data = [valid.loc[valid["transition"].eq(name), "marginal_additional_carbon_kgco2e_per_screened_l_avoided"].to_numpy() for name in transition_order]
    box = ax_c.boxplot(data, positions=np.arange(4), widths=0.58, patch_artist=True, showfliers=False, medianprops={"color": "white", "linewidth": 1.1})
    for patch in box["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.7)
        patch.set_edgecolor(BLUE)
    for element in box["whiskers"] + box["caps"]:
        element.set_color(BLUE)
        element.set_linewidth(0.8)
    for position, values in enumerate(data):
        sample = values[:: max(1, len(values) // 60)]
        jitter = np.linspace(-0.18, 0.18, len(sample))
        ax_c.scatter(position + jitter, sample, s=5, color=GRAPHITE, alpha=0.30, linewidth=0)
    ax_c.set_xticks(np.arange(4), transition_order, rotation=18, ha="right")
    ax_c.set_ylabel("Marginal carbon cost\n(kgCO2e per L avoided)")
    ax_c.set_title("972 adjacent-cap segments", loc="left", fontweight="bold")
    apply_grid(ax_c)
    panel_label(ax_c, "c")

    ax_d.set_axis_off()
    labels = ["Scenario cells", "Optimal solves", "Constraint violations"]
    values = [243, 1215, 0]
    colors = [GRAPHITE, TEAL, RUST]
    y_positions = [0.72, 0.47, 0.22]
    for label, value, color, y in zip(labels, values, colors, y_positions, strict=True):
        ax_d.add_patch(Rectangle((0.05, y - 0.06), 0.60, 0.12, transform=ax_d.transAxes, facecolor="#F3F5F6", edgecolor=GRID, linewidth=0.6))
        width = 0.60 * value / max(values)
        ax_d.add_patch(Rectangle((0.05, y - 0.06), width, 0.12, transform=ax_d.transAxes, facecolor=color, edgecolor=color, linewidth=0.6))
        ax_d.text(0.70, y, f"{value}", transform=ax_d.transAxes, va="center", fontsize=9, fontweight="bold", color=color)
        ax_d.text(0.05, y + 0.09, label, transform=ax_d.transAxes, fontsize=6.4)
    ax_d.text(0.05, 0.93, "Solver and feasibility QA", transform=ax_d.transAxes, fontweight="bold")
    ax_d.text(0.05, 0.02, "All checks passed; values are audit counts, not performance metrics.", transform=ax_d.transAxes, fontsize=5.8, color=GRAPHITE)
    panel_label(ax_d, "d")
    return fig


def figure_4(source_dir: Path) -> plt.Figure:
    wue_summary = pd.read_csv(source_dir / "fig04_spatial_wue_cap_summary_source_v1.csv")
    wue = pd.read_csv(source_dir / "fig04_spatial_wue_frontier_source_v1.csv")
    attributes = pd.read_csv(source_dir / "fig04_wue_profile_attributes_source_v1.csv")
    energy = pd.read_csv(source_dir / "fig04_energy_scaling_source_v1.csv")
    fig = plt.figure(figsize=(7.09, 5.25))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 0.92], height_ratios=[1.0, 1.0], wspace=0.48, hspace=0.60)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    x = 100 * wue_summary["cap_fraction_of_carbon_first"].to_numpy()
    lower = wue_summary["additional_carbon_kgco2e_vs_carbon_first_minimum"].to_numpy()
    q25 = wue_summary["additional_carbon_kgco2e_vs_carbon_first_q25"].to_numpy()
    median = wue_summary["additional_carbon_kgco2e_vs_carbon_first_median"].to_numpy()
    q75 = wue_summary["additional_carbon_kgco2e_vs_carbon_first_q75"].to_numpy()
    upper = wue_summary["additional_carbon_kgco2e_vs_carbon_first_maximum"].to_numpy()
    ax_a.vlines(x, lower, upper, color=RUST, linewidth=1.0, alpha=0.9)
    ax_a.vlines(x, q25, q75, color=RUST, linewidth=4.0)
    ax_a.scatter(x, median, facecolor="white", edgecolor=RUST, linewidth=1.0, s=22, zorder=3)
    ax_a.set_xlim(-4, 104)
    ax_a.set_xlabel("Allowed screened water (% of carbon-first)")
    ax_a.set_ylabel("Additional carbon vs carbon-first (kgCO2e)")
    ax_a.set_title("81 reoptimized spatial WUE profiles", loc="left", fontweight="bold")
    ax_a.text(
        0.03,
        0.06,
        "thin: min-max; thick: Q25-Q75; open circle: median\nProfiles are complete factorial cases, not samples",
        transform=ax_a.transAxes,
        fontsize=5.7,
        color=GRAPHITE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        zorder=6,
    )
    apply_grid(ax_a)
    panel_label(ax_a, "a")

    intermediate = wue.loc[wue["cap_fraction_of_carbon_first"].eq(0.25)].merge(attributes, on="wue_profile", validate="one_to_one")
    screened_lower = intermediate["mean_unscreened_region_wue_l_per_kwh_it"] >= intermediate["mean_screened_region_wue_l_per_kwh_it"]
    for selector, color, marker, label in [
        (screened_lower, GREEN, "o", "screened mean <= unscreened"),
        (~screened_lower, RUST, "D", "screened mean > unscreened"),
    ]:
        subset = intermediate.loc[selector]
        ax_b.scatter(
            subset["mean_screened_region_wue_l_per_kwh_it"],
            subset["additional_carbon_kgco2e_vs_carbon_first"],
            s=23,
            c=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.85,
            label=label,
        )
    ax_b.set_xlabel("Mean WUE in screened regions\n(L per kWh-IT)")
    ax_b.set_ylabel("25%-allowed carbon penalty (kgCO2e)")
    ax_b.set_title("Intermediate-cap sensitivity", loc="left", fontweight="bold")
    ax_b.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.29),
        ncol=2,
        fontsize=4.8,
        handletextpad=0.3,
        columnspacing=0.8,
        frameon=False,
    )
    apply_grid(ax_b)
    panel_label(ax_b, "b")

    region_style = {
        "east": (GREEN, "o", "-", "east"),
        "north": (RUST, "s", "--", "north"),
        "northwest": ("#C98771", "D", ":", "northwest"),
        "southwest": ("#76A986", "^", "-.", "southwest"),
    }
    central = energy.loc[energy["energy_level"].eq("central")].sort_values("cap_fraction_of_carbon_first")
    for region, (color, marker, linestyle, label) in region_style.items():
        value = central[f"assigned_{region}_accelerator_hours"] / central["served_accelerator_hours"]
        ax_c.plot(
            100 * central["cap_fraction_of_carbon_first"],
            100 * value,
            color=color,
            linewidth=1.25,
            marker=marker,
            linestyle=linestyle,
            markersize=3,
            label=label,
        )
    ax_c.set_xlim(-4, 104)
    ax_c.set_xlabel("Allowed screened water (% of carbon-first)")
    ax_c.set_ylabel("Allocation share of served service (%)")
    ax_c.set_title("Allocation is invariant to energy scale", loc="left", fontweight="bold")
    ax_c.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.34), columnspacing=0.8, handlelength=1.4)
    ax_c.text(
        0.03,
        0.14,
        "Low, central, and high energy curves coincide\n(max allocation difference: 2.14e-14 accelerator-h)",
        transform=ax_c.transAxes,
        fontsize=5.7,
        color=GRAPHITE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        zorder=6,
    )
    apply_grid(ax_c)
    panel_label(ax_c, "c")

    energy_order = ["low", "central", "high"]
    energy_colors = ["#8FA7B7", TEAL, "#273E52"]
    energy_markers = ["o", "s", "D"]
    energy_linestyles = ["--", "-", ":"]
    for level, color, marker, linestyle in zip(energy_order, energy_colors, energy_markers, energy_linestyles, strict=True):
        subset = energy.loc[energy["energy_level"].eq(level)].sort_values("cap_fraction_of_carbon_first")
        ax_d.plot(
            100 * subset["cap_fraction_of_carbon_first"],
            subset["carbon_kgco2e"],
            marker=marker,
            linestyle=linestyle,
            markersize=3,
            linewidth=1.2,
            color=color,
            label=level,
        )
    ax_d.set_xlim(-4, 104)
    ax_d.set_xlabel("Allowed screened water (% of carbon-first)")
    ax_d.set_ylabel("Absolute modeled carbon (kgCO2e)")
    ax_d.set_title("Energy intensity rescales absolute carbon", loc="left", fontweight="bold")
    ax_d.legend(title="IT-energy level", loc="center right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
    apply_grid(ax_d)
    panel_label(ax_d, "d")
    return fig


def save_figure(fig: plt.Figure, stem: str, source_files: list[str]) -> list[dict[str, object]]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    paths = {
        "svg": EXPORT_DIR / f"{stem}.svg",
        "pdf": EXPORT_DIR / f"{stem}.pdf",
        "tiff": EXPORT_DIR / f"{stem}.tiff",
        "png": EXPORT_DIR / f"{stem}.png",
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    for kind, path in paths.items():
        outputs.append(
            {
                "figure": stem,
                "format": kind,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_files": source_files,
            }
        )
    plt.close(fig)
    return outputs


def main() -> None:
    global ROOT, FIGURE_DIR, SOURCE_DIR, EXPORT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root
    FIGURE_DIR = ROOT / "04_figures"
    SOURCE_DIR = FIGURE_DIR / "source_data"
    EXPORT_DIR = FIGURE_DIR / "exports"

    figures = [
        (
            "fig01_environmental_decision_model_v1",
            figure_1(),
            ["fig01_model_definition_source_v1.csv"],
        ),
        (
            "fig02_reference_screened_water_frontier_v1",
            figure_2(SOURCE_DIR),
            [
                "fig02_reference_frontier_source_v1.csv",
                "fig02_reference_segment_source_v1.csv",
                "fig02_reference_block_intervals_source_v1.csv",
                "fig02_selected_allocations_source_v1.csv",
            ],
        ),
        (
            "fig03_scenario_envelope_v1",
            figure_3(SOURCE_DIR),
            [
                "fig03_scenario_cap_summary_source_v1.csv",
                "fig03_zero_cap_scenarios_source_v1.csv",
                "fig03_scenario_segment_source_v1.csv",
                "fig03_solver_qa_source_v1.csv",
            ],
        ),
        (
            "fig04_wue_and_energy_boundaries_v1",
            figure_4(SOURCE_DIR),
            [
                "fig04_spatial_wue_frontier_source_v1.csv",
                "fig04_spatial_wue_cap_summary_source_v1.csv",
                "fig04_wue_profile_attributes_source_v1.csv",
                "fig04_energy_scaling_source_v1.csv",
            ],
        ),
    ]
    manifest: list[dict[str, object]] = []
    for stem, figure, source_files in figures:
        manifest.extend(save_figure(figure, stem, source_files))
    (FIGURE_DIR / "figure_export_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
