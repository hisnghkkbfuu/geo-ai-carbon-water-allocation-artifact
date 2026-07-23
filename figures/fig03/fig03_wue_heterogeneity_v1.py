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
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image
from pypdf import PdfReader


MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 135

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESULTS_DIR = PROJECT_ROOT / "04_results"
PROCESSED_DIR = PROJECT_ROOT / "01_data_processed"
ACCOUNTING_INPUT = RESULTS_DIR / "wue_factorial_accounting_v1.csv"
PROFILE_INPUT = PROCESSED_DIR / "wue_profiles.csv"

SOURCE_CSV = HERE / "fig03_source_data_v1.csv"
SOURCE_MANIFEST = HERE / "fig03_source_manifest_v1.json"
OUTPUT_STEM = HERE / "fig03_wue_heterogeneity_v1"
EXPORT_MANIFEST = HERE / "fig03_export_manifest_v1.json"

COLORS = {
    "ink": "#202124",
    "muted": "#62676B",
    "reference": "#9AA0A4",
    "e0": "#6E7378",
    "e1": "#225E91",
    "stress": "#B44A3E",
    "wue_low": "#5F7F7A",
    "wue_mid": "#A2773C",
    "wue_high": "#765A91",
    "white": "#FFFFFF",
}

WUE_STYLE = {
    0.36: {"color": COLORS["wue_low"], "marker": "o", "label": "0.36"},
    1.15: {"color": COLORS["wue_mid"], "marker": "^", "label": "1.15"},
    2.00: {"color": COLORS["wue_high"], "marker": "s", "label": "2.00"},
}

POLICY_STYLE = {
    "E0": {"color": COLORS["e0"], "marker": "o", "label": "E0"},
    "E1": {"color": COLORS["e1"], "marker": "s", "label": "E1"},
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
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


def assert_close(label: str, actual: float, expected: float, atol: float = 1e-9) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise ValueError(f"Frozen-value drift for {label}: actual={actual}, expected={expected}")


def build_source_data() -> tuple[pd.DataFrame, dict[str, object]]:
    accounting = pd.read_csv(ACCOUNTING_INPUT)
    profiles = pd.read_csv(PROFILE_INPUT)
    require_columns(
        accounting,
        {
            "water_risk_mapping",
            "wue_profile",
            "status_quo_direct_water_l",
            "carbon_first_direct_water_l",
            "direct_water_change_pct",
            "status_quo_high_stress_water_l",
            "carbon_first_high_stress_water_l",
            "high_stress_water_share_change_points",
        },
        "WUE factorial accounting",
    )
    require_columns(profiles, {"wue_profile", "region", "wue_l_per_kwh_it", "profile_type"}, "WUE profile definitions")

    factorial_profiles = profiles.loc[profiles["profile_type"].eq("full-factorial mechanism sensitivity")].copy()
    if len(factorial_profiles) != 324:
        raise ValueError(f"Expected 324 regional WUE assignments; found {len(factorial_profiles)}")
    profile_matrix = factorial_profiles.pivot(index="wue_profile", columns="region", values="wue_l_per_kwh_it").reset_index()
    required_regions = {"east", "north", "northwest", "southwest"}
    if set(profile_matrix.columns) != {"wue_profile", *required_regions}:
        raise ValueError(f"Unexpected regional WUE columns: {profile_matrix.columns.tolist()}")
    if len(profile_matrix) != 81:
        raise ValueError(f"Expected 81 WUE profiles; found {len(profile_matrix)}")
    if set(profile_matrix[sorted(required_regions)].stack().unique()) != {0.36, 1.15, 2.0}:
        raise ValueError("Factorial WUE levels drifted from the frozen 0.36/1.15/2.00 levels")
    if len(profile_matrix[["east", "north", "northwest", "southwest"]].drop_duplicates()) != 81:
        raise ValueError("WUE factorial profile tuples are not complete and unique")

    merged = accounting.merge(profile_matrix, on="wue_profile", how="left", validate="many_to_one")
    if len(merged) != 162:
        raise ValueError(f"Expected 162 accounting rows after profile join; found {len(merged)}")
    merged["status_quo_high_stress_share_pct"] = 100.0 * merged["status_quo_high_stress_water_l"] / merged["status_quo_direct_water_l"]
    merged["carbon_first_high_stress_share_pct"] = 100.0 * merged["carbon_first_high_stress_water_l"] / merged["carbon_first_direct_water_l"]
    merged["recomputed_share_change_pp"] = merged["carbon_first_high_stress_share_pct"] - merged["status_quo_high_stress_share_pct"]
    if not np.allclose(merged["recomputed_share_change_pp"], merged["high_stress_water_share_change_points"], atol=1e-9, rtol=0.0):
        raise ValueError("Stored high-stress share changes do not match recomputed shares")

    portfolio = merged.loc[merged["water_risk_mapping"].eq("portfolio_mean")].copy()
    conservative = merged.loc[merged["water_risk_mapping"].eq("conservative_max")].copy()
    if len(portfolio) != 81 or len(conservative) != 81:
        raise ValueError("Each water-risk mapping must contain all 81 WUE profiles")
    if not np.allclose(
        portfolio.sort_values("wue_profile")["direct_water_change_pct"].to_numpy(),
        conservative.sort_values("wue_profile")["direct_water_change_pct"].to_numpy(),
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("Physical total-water changes differ across screening mappings")

    rank_table = portfolio.sort_values(["direct_water_change_pct", "wue_profile"], kind="mergesort").reset_index(drop=True)
    rank_table["profile_rank"] = np.arange(1, len(rank_table) + 1)
    rank_map = rank_table.set_index("wue_profile")["profile_rank"].to_dict()
    merged["profile_rank"] = merged["wue_profile"].map(rank_map).astype(int)
    tolerance = 1e-9
    merged["direct_water_change_class"] = np.select(
        [merged["direct_water_change_pct"] > tolerance, merged["direct_water_change_pct"] < -tolerance],
        ["increase", "decrease"],
        default="unchanged",
    )
    merged["use_panel_a"] = merged["water_risk_mapping"].eq("portfolio_mean")
    merged["use_panel_b"] = merged["water_risk_mapping"].eq("portfolio_mean")
    merged["use_panel_c"] = merged["water_risk_mapping"].eq("portfolio_mean")
    merged["use_panel_d"] = True
    merged["evidence_id"] = "T3C04;T1R08;T1R09"

    source_columns = [
        "water_risk_mapping",
        "wue_profile",
        "profile_rank",
        "east",
        "north",
        "northwest",
        "southwest",
        "status_quo_direct_water_l",
        "carbon_first_direct_water_l",
        "direct_water_change_pct",
        "status_quo_high_stress_water_l",
        "carbon_first_high_stress_water_l",
        "status_quo_high_stress_share_pct",
        "carbon_first_high_stress_share_pct",
        "high_stress_water_share_change_points",
        "direct_water_change_class",
        "use_panel_a",
        "use_panel_b",
        "use_panel_c",
        "use_panel_d",
        "evidence_id",
    ]
    source = merged[source_columns].rename(
        columns={
            "east": "east_wue_l_per_kwh_it",
            "north": "north_wue_l_per_kwh_it",
            "northwest": "northwest_wue_l_per_kwh_it",
            "southwest": "southwest_wue_l_per_kwh_it",
            "high_stress_water_share_change_points": "high_stress_share_change_pp",
        }
    )
    source.to_csv(SOURCE_CSV, index=False, float_format="%.15g")

    pm = source.loc[source["water_risk_mapping"].eq("portfolio_mean")].copy()
    change_by_mapping = source.pivot(index="wue_profile", columns="water_risk_mapping", values="direct_water_change_pct")
    mapping_change_max_abs_difference = float((change_by_mapping["portfolio_mean"] - change_by_mapping["conservative_max"]).abs().max())
    increase = int((pm["direct_water_change_pct"] > tolerance).sum())
    decrease = int((pm["direct_water_change_pct"] < -tolerance).sum())
    unchanged = int((pm["direct_water_change_pct"].abs() <= tolerance).sum())
    east_summary = pm.groupby("east_wue_l_per_kwh_it")["direct_water_change_pct"].agg(["min", "median", "max", "count"]).reset_index().to_dict(orient="records")
    scientific_summary = {
        "profile_count": 81,
        "mapping_rows": {name: int((source["water_risk_mapping"] == name).sum()) for name in ("portfolio_mean", "conservative_max")},
        "portfolio_direct_water_counts": {"increase": increase, "decrease": decrease, "unchanged": unchanged},
        "portfolio_direct_water_range_pct": [float(pm["direct_water_change_pct"].min()), float(pm["direct_water_change_pct"].max())],
        "portfolio_direct_water_median_pct": float(pm["direct_water_change_pct"].median()),
        "portfolio_high_stress_share_change_range_pp": [float(pm["high_stress_share_change_pp"].min()), float(pm["high_stress_share_change_pp"].max())],
        "portfolio_high_stress_share_change_median_pp": float(pm["high_stress_share_change_pp"].median()),
        "portfolio_high_stress_positive_count": int((pm["high_stress_share_change_pp"] > 0).sum()),
        "east_wue_group_summary": east_summary,
        "portfolio_e0_share_range_pct": [float(pm["status_quo_high_stress_share_pct"].min()), float(pm["status_quo_high_stress_share_pct"].max())],
        "portfolio_e1_share_range_pct": [float(pm["carbon_first_high_stress_share_pct"].min()), float(pm["carbon_first_high_stress_share_pct"].max())],
        "conservative_e0_share_range_pct": [float(conservative["status_quo_high_stress_share_pct"].min()), float(conservative["status_quo_high_stress_share_pct"].max())],
        "conservative_e1_share_range_pct": [float(conservative["carbon_first_high_stress_share_pct"].min()), float(conservative["carbon_first_high_stress_share_pct"].max())],
        "profile_rank_range": [int(source["profile_rank"].min()), int(source["profile_rank"].max())],
        "mapping_change_max_abs_difference_pct": mapping_change_max_abs_difference,
    }
    assertions = {
        "no_missing_or_infinite_values": bool(not source.isna().any().any() and np.isfinite(source.select_dtypes(include="number")).all().all()),
        "two_mappings_and_162_rows": bool(len(source) == 162 and source["water_risk_mapping"].value_counts().to_dict() == {"portfolio_mean": 81, "conservative_max": 81}),
        "complete_factorial_81_tuples": bool(len(profile_matrix) == 81 and len(profile_matrix[["east", "north", "northwest", "southwest"]].drop_duplicates()) == 81),
        "physical_change_invariant_to_screening_mapping": bool(mapping_change_max_abs_difference <= 1e-9),
        "portfolio_counts_are_40_38_3": bool((increase, decrease, unchanged) == (40, 38, 3)),
        "portfolio_exposure_change_positive_81_of_81": bool((pm["high_stress_share_change_pp"] > 0).sum() == 81),
        "east_wue_levels_have_27_profiles_each": bool(pm["east_wue_l_per_kwh_it"].value_counts().sort_index().to_dict() == {0.36: 27, 1.15: 27, 2.0: 27}),
        "portfolio_e0_exposure_is_zero": bool(np.allclose(pm["status_quo_high_stress_share_pct"], 0.0, atol=1e-9)),
        "portfolio_e1_exposure_is_positive": bool((pm["carbon_first_high_stress_share_pct"] > 0).all()),
        "conservative_max_both_policies_are_100_pct": bool(np.allclose(conservative[["status_quo_high_stress_share_pct", "carbon_first_high_stress_share_pct"]], 100.0, atol=1e-9)),
        "profile_ranks_are_one_to_81": bool(sorted(source["profile_rank"].unique().tolist()) == list(range(1, 82))),
        "stored_share_change_matches_recomputed": bool(np.allclose(source["high_stress_share_change_pp"], source["carbon_first_high_stress_share_pct"] - source["status_quo_high_stress_share_pct"], atol=1e-9, rtol=0.0)),
    }
    upstream = [
        {"role": "WUE factorial accounting", "rows": int(len(accounting)), "sha256": sha256(ACCOUNTING_INPUT)},
        {"role": "regional WUE profile definitions", "rows": int(len(profiles)), "sha256": sha256(PROFILE_INPUT)},
    ]
    manifest = {
        "figure": "Fig. 3",
        "derived_source": {"path": SOURCE_CSV.name, "rows": int(len(source)), "panel_filters": {"a": "water_risk_mapping == portfolio_mean", "b": "water_risk_mapping == portfolio_mean", "c": "water_risk_mapping == portfolio_mean", "d": "both mappings"}, "sha256": sha256(SOURCE_CSV)},
        "upstream_inputs": upstream,
        "derivation": {
            "profile_join": "regional WUE assignments pivoted by profile and joined to factorial accounting",
            "portfolio_rank": "ascending portfolio-mean direct_water_change_pct; wue_profile is deterministic tie-break",
            "exposure_share_pct": "100 × high-stress direct water / total direct water",
            "exposure_change_pp": "carbon-first exposure share minus status-quo exposure share",
            "unchanged_tolerance": 1e-9,
        },
        "scientific_summary": scientific_summary,
        "assertions": assertions,
    }
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return source, scientific_summary


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(direction="out", colors=COLORS["ink"], pad=2)


def panel_heading(ax: plt.Axes, letter: str, title: str, x_letter: float = -0.12) -> None:
    ax.text(x_letter, 1.025, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom", color=COLORS["ink"], clip_on=False)
    ax.text(0.0, 1.025, title, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom", color=COLORS["ink"], clip_on=False)


def wue_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker=WUE_STYLE[level]["marker"], color="none", markerfacecolor=WUE_STYLE[level]["color"], markeredgecolor="white", markeredgewidth=0.4, markersize=5.2, label=WUE_STYLE[level]["label"])
        for level in (0.36, 1.15, 2.0)
    ]


def draw_panel_a(ax: plt.Axes, source: pd.DataFrame) -> None:
    panel_heading(ax, "a", "Ranked total-water changes")
    style_axis(ax)
    data = source.loc[source["use_panel_a"]].sort_values("profile_rank")
    y_min, y_max = -30.0, 145.0
    ax.set_xlim(0, 82)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([1, 20, 40, 60, 81])
    ax.set_yticks([-20, 0, 40, 80, 120])
    ax.set_xlabel("WUE-profile rank (sorted by total-water change)", labelpad=4)
    ax.set_ylabel("E1/E0 total direct-water change (%)", labelpad=4)
    ax.axhline(0, color=COLORS["reference"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    for level in (0.36, 1.15, 2.0):
        group = data.loc[np.isclose(data["east_wue_l_per_kwh_it"], level)]
        style = WUE_STYLE[level]
        ax.scatter(group["profile_rank"], group["direct_water_change_pct"], s=22, marker=style["marker"], facecolor=style["color"], edgecolor="white", linewidth=0.35, alpha=0.92, zorder=3, label=style["label"])
    ax.legend(handles=wue_handles(), title="East WUE (L/kWh-IT)", title_fontsize=5.8, fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, 0.99), ncol=3, handletextpad=0.25, columnspacing=0.65, borderaxespad=0.0)
    counts = data["direct_water_change_class"].value_counts()
    ax.text(0.98, 0.035, f"Complete profiles (n = 81)\n{int(counts['increase'])} increase  |  {int(counts['decrease'])} decrease  |  {int(counts['unchanged'])} unchanged", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=COLORS["ink"], bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": COLORS["reference"], "linewidth": 0.55})


def draw_panel_b(ax: plt.Axes, source: pd.DataFrame) -> None:
    panel_heading(ax, "b", "Matched-rank exposure changes")
    style_axis(ax)
    data = source.loc[source["use_panel_b"]].sort_values("profile_rank")
    ax.set_xlim(0, 82)
    ax.set_ylim(-0.35, 19.0)
    ax.set_xticks([1, 20, 40, 60, 81])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("WUE-profile rank (same order as a)", labelpad=4)
    ax.set_ylabel("High-stress share change (pp)", labelpad=4)
    ax.axhline(0, color=COLORS["reference"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.scatter(data["profile_rank"], data["high_stress_share_change_pp"], s=18, marker="o", facecolor=COLORS["stress"], edgecolor="white", linewidth=0.3, alpha=0.82, zorder=3)
    ax.text(0.98, 0.94, "81/81 profiles remain above zero", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=COLORS["stress"], fontweight="bold", bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": COLORS["stress"], "linewidth": 0.5})


def deterministic_jitter(n: int, width: float = 0.30) -> np.ndarray:
    index = np.arange(n, dtype=float)
    return (((index * 0.61803398875) % 1.0) - 0.5) * width


def draw_panel_c(ax: plt.Axes, source: pd.DataFrame) -> None:
    panel_heading(ax, "c", "Mechanism by east WUE")
    style_axis(ax)
    data = source.loc[source["use_panel_c"]].copy()
    levels = [0.36, 1.15, 2.0]
    values = [data.loc[np.isclose(data["east_wue_l_per_kwh_it"], level), "direct_water_change_pct"].to_numpy() for level in levels]
    positions = np.arange(3)
    box = ax.boxplot(values, positions=positions, widths=0.48, patch_artist=True, whis=(0, 100), showfliers=False, manage_ticks=False, medianprops={"color": COLORS["ink"], "linewidth": 1.1}, whiskerprops={"color": COLORS["muted"], "linewidth": 0.75}, capprops={"color": COLORS["muted"], "linewidth": 0.75}, boxprops={"linewidth": 0.75})
    for patch, level in zip(box["boxes"], levels):
        patch.set_facecolor(WUE_STYLE[level]["color"])
        patch.set_alpha(0.16)
        patch.set_edgecolor(WUE_STYLE[level]["color"])
    for position, level, group_values in zip(positions, levels, values):
        group_values = np.sort(group_values)
        ax.scatter(position + deterministic_jitter(len(group_values)), group_values, s=17, marker=WUE_STYLE[level]["marker"], facecolor=WUE_STYLE[level]["color"], edgecolor="white", linewidth=0.3, alpha=0.82, zorder=3)
    ax.axhline(0, color=COLORS["reference"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-30.0, 145.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(["0.36", "1.15", "2.00"])
    ax.set_yticks([-20, 0, 40, 80, 120])
    ax.set_xlabel("East WUE (L/kWh-IT)", labelpad=4)
    ax.set_ylabel("E1/E0 total-water change (%)", labelpad=4)
    ax.text(0.98, 0.96, "27 complete profiles per level\nboxes: IQR; whiskers: full range", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=COLORS["muted"], linespacing=1.1)


def draw_policy_marker(ax: plt.Axes, x: float, y: float, policy: str, alpha: float = 0.78) -> None:
    style = POLICY_STYLE[policy]
    ax.scatter([x], [y], s=22, marker=style["marker"], facecolor=style["color"], edgecolor="white", linewidth=0.3, alpha=alpha, zorder=3)


def draw_panel_d(ax: plt.Axes, source: pd.DataFrame) -> None:
    panel_heading(ax, "d", "Screening discrimination")
    style_axis(ax)
    positions = {"portfolio_mean": {"E0": -0.18, "E1": 0.18}, "conservative_max": {"E0": 0.82, "E1": 1.18}}
    for mapping in ("portfolio_mean", "conservative_max"):
        for policy, column in (("E0", "status_quo_high_stress_share_pct"), ("E1", "carbon_first_high_stress_share_pct")):
            values = source.loc[source["water_risk_mapping"].eq(mapping), column].astype(float).sort_values().to_numpy()
            center = positions[mapping][policy]
            jitter = deterministic_jitter(len(values), width=0.20)
            draw_policy_marker(ax, center + jitter, values, policy, alpha=0.62)
            ax.vlines(center, float(values.min()), float(values.max()), color=POLICY_STYLE[policy]["color"], linewidth=0.9, alpha=0.8, zorder=1)
            ax.hlines(float(np.median(values)), center - 0.09, center + 0.09, color=POLICY_STYLE[policy]["color"], linewidth=1.5, zorder=4)

    ax.axhline(0, color=COLORS["reference"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-4, 105)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Portfolio\nmean", "Conservative\nmax"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("High-stress direct-water share (%)", labelpad=4)
    handles = [Line2D([0], [0], marker=POLICY_STYLE[policy]["marker"], color="none", markerfacecolor=POLICY_STYLE[policy]["color"], markeredgecolor="white", markeredgewidth=0.3, markersize=5.2, label=policy) for policy in ("E0", "E1")]
    ax.legend(handles=handles, labels=["E0", "E1"], loc="upper left", bbox_to_anchor=(0.0, 0.99), ncol=2, fontsize=5.8, handletextpad=0.25, columnspacing=0.55, borderaxespad=0.0)
    ax.text(0.02, 0.22, "Portfolio mean:\nE0 = 0%; E1 > 0% in 81/81", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4, color=COLORS["ink"], bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": COLORS["reference"], "linewidth": 0.45})
    ax.text(0.98, 0.76, "Conservative max:\n81/81 profiles at 100%\nclassification degenerates", transform=ax.transAxes, ha="right", va="center", fontsize=5.4, color=COLORS["muted"], linespacing=1.05, bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": COLORS["reference"], "linewidth": 0.45})


def build_figure(source: pd.DataFrame, summary: dict[str, object]) -> plt.Figure:
    figure = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.62, 1.0], height_ratios=[1.08, 1.0])
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[1, 0], sharex=ax_a)
    ax_c = figure.add_subplot(grid[0, 1])
    ax_d = figure.add_subplot(grid[1, 1])
    draw_panel_a(ax_a, source)
    draw_panel_b(ax_b, source)
    draw_panel_c(ax_c, source)
    draw_panel_d(ax_d, source)
    figure.subplots_adjust(left=0.080, right=0.985, bottom=0.095, top=0.925, wspace=0.36, hspace=0.62)
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
    return {"canvas_width_px_at_render_dpi": round(float(canvas.width), 3), "canvas_height_px_at_render_dpi": round(float(canvas.height), 3), "visible_text_artist_count": sum(1 for artist in figure.findobj(match=Text) if artist.get_visible() and artist.get_text().strip()), "text_outside_canvas_count": len(outside), "text_outside_canvas": outside}


def svg_metadata(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    return {"width": root.attrib.get("width"), "height": root.attrib.get("height"), "viewBox": root.attrib.get("viewBox"), "text_node_count": sum(1 for node in root.iter() if node.tag.endswith("text"))}


def pdf_metadata(path: Path) -> dict[str, object]:
    page = PdfReader(str(path)).pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    fonts = page["/Resources"].get("/Font", {})
    records = []
    for key, value in fonts.items():
        font = value.get_object()
        records.append({"resource": str(key), "subtype": str(font.get("/Subtype")), "basefont": str(font.get("/BaseFont"))})
    return {"width_mm": round(width_pt / 72 * MM_PER_INCH, 3), "height_mm": round(height_pt / 72 * MM_PER_INCH, 3), "fonts": records, "type3_font_count": sum(record["subtype"] == "/Type3" for record in records)}


def raster_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        return {"width_px": image.width, "height_px": image.height, "mode": image.mode, "format": image.format, "dpi": [round(float(value), 3) for value in image.info.get("dpi", (0, 0))], "compression": image.info.get("compression")}


def export_figure(figure: plt.Figure, source: pd.DataFrame, summary: dict[str, object]) -> None:
    layout = figure_layout_metadata(figure)
    outputs = {"svg": OUTPUT_STEM.with_suffix(".svg"), "pdf": OUTPUT_STEM.with_suffix(".pdf"), "tiff": OUTPUT_STEM.with_suffix(".tiff"), "png": OUTPUT_STEM.with_suffix(".png")}
    figure.savefig(outputs["svg"], format="svg")
    figure.savefig(outputs["pdf"], format="pdf", metadata={"Title": "Fig. 3 | Spatial WUE heterogeneity separates total direct water from high-stress exposure"})
    figure.savefig(outputs["tiff"], format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(outputs["png"], format="png", dpi=300)
    plt.close(figure)
    file_manifest = {name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in outputs.items()}
    file_manifest["svg"]["metadata"] = svg_metadata(outputs["svg"])
    file_manifest["pdf"]["metadata"] = pdf_metadata(outputs["pdf"])
    file_manifest["tiff"]["metadata"] = raster_metadata(outputs["tiff"])
    file_manifest["png"]["metadata"] = raster_metadata(outputs["png"])
    pdf_meta = file_manifest["pdf"]["metadata"]
    tiff_meta = file_manifest["tiff"]["metadata"]
    png_meta = file_manifest["png"]["metadata"]
    manifest = {
        "figure": "Fig. 3",
        "title": "Spatial WUE heterogeneity separates total direct water from high-stress exposure",
        "backend": "Python / matplotlib Agg",
        "intended_size_mm": {"width": FIGURE_WIDTH_MM, "height": FIGURE_HEIGHT_MM},
        "source_data": {"path": SOURCE_CSV.name, "rows": int(len(source)), "panel_filters": {"a": "portfolio_mean rows", "b": "portfolio_mean rows", "c": "portfolio_mean rows", "d": "both water-risk mappings"}, "sha256": sha256(SOURCE_CSV)},
        "scientific_summary": summary,
        "layout": layout,
        "outputs": file_manifest,
        "qa_contract": {
            "svg_editable_text": file_manifest["svg"]["metadata"]["text_node_count"] > 0,
            "pdf_size_matches_mm": abs(pdf_meta["width_mm"] - FIGURE_WIDTH_MM) < 0.02 and abs(pdf_meta["height_mm"] - FIGURE_HEIGHT_MM) < 0.02,
            "pdf_has_no_type3_fonts": pdf_meta["type3_font_count"] == 0,
            "tiff_nominal_600_dpi": all(abs(value - 600) < 1 for value in tiff_meta["dpi"]),
            "tiff_lzw_compression": tiff_meta["compression"] == "tiff_lzw",
            "png_nominal_300_dpi": all(abs(value - 300) < 1 for value in png_meta["dpi"]),
            "no_text_outside_canvas": layout["text_outside_canvas_count"] == 0,
            "source_has_expected_162_rows": len(source) == 162,
        },
    }
    EXPORT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    source, summary = build_source_data()
    export_figure(build_figure(source, summary), source, summary)
    print(f"Generated {SOURCE_CSV.name} from frozen WUE inputs.")
    print(f"Generated {OUTPUT_STEM.name} in SVG, PDF, TIFF, and PNG formats.")
    print(f"Source manifest: {SOURCE_MANIFEST.name}")
    print(f"Export manifest: {EXPORT_MANIFEST.name}")


if __name__ == "__main__":
    main()
