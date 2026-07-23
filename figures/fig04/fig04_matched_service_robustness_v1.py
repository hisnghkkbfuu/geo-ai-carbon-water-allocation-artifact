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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from PIL import Image
from pypdf import PdfReader


MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 170

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESULTS_DIR = PROJECT_ROOT / "04_results"

MATRIX_INPUT = RESULTS_DIR / "carbon_service_sensitivity_matrix_v1.csv"
MAPPING_INTERVAL_INPUT = RESULTS_DIR / "external_mapping_sensitivity_intervals_v1.csv"
FROZEN_INTERVAL_INPUT = RESULTS_DIR / "external_confirmation_paired_block_intervals_v1.csv"
DAILY_INPUT = RESULTS_DIR / "external_daily_stratified_effects_v1.csv"
DAILY_AUDIT_INPUT = RESULTS_DIR / "external_daily_stratified_audit_v1.json"
EXTERNAL_QA_INPUT = RESULTS_DIR / "external_confirmation_experiment_qa_v1.json"
STAGE1_QA_INPUT = RESULTS_DIR / "stage1_experiment_qa.json"

SOURCE_CSV = HERE / "fig04_source_data_v1.csv"
SOURCE_MANIFEST = HERE / "fig04_source_manifest_v1.json"
OUTPUT_STEM = HERE / "fig04_matched_service_robustness_v1"
EXPORT_MANIFEST = HERE / "fig04_export_manifest_v1.json"

COLORS = {
    "ink": "#202124",
    "muted": "#62676B",
    "reference": "#9AA0A4",
    "pale": "#E9EDF0",
    "blue": "#225E91",
    "blue_light": "#70A2C6",
    "e2": "#00857C",
    "mapping_teal": "#4F7F7A",
    "red": "#B44A3E",
    "gold": "#A2773C",
    "white": "#FFFFFF",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "carbon_reduction",
    ["#F4F7F9", "#C9DBE8", "#77A6C8", "#225E91", "#143E66"],
)

CARBON_ORDER = [
    "conservative_site_selection",
    "equal_site_portfolio",
    "optimistic_site_selection",
]
CAPACITY_ORDER = ["ample", "central", "tight"]
MIGRATION_ORDER = [0.15, 0.30, 0.50]
LATENCY_ORDER = [15.0, 20.0, 35.0]
MAPPING_ORDER = ["frozen_global_upper", "triangulated_linear", "local_simplex_upper"]

CARBON_LABEL = {
    "conservative_site_selection": "Conservative",
    "equal_site_portfolio": "Equal",
    "optimistic_site_selection": "Optimistic",
}
CAPACITY_LABEL = {"ample": "ample", "central": "central", "tight": "tight"}
MAPPING_LABEL = {
    "frozen_global_upper": "Frozen upper",
    "triangulated_linear": "Triangulated",
    "local_simplex_upper": "Local simplex",
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_source_data() -> tuple[pd.DataFrame, dict[str, object]]:
    matrix = pd.read_csv(MATRIX_INPUT)
    mapping_intervals = pd.read_csv(MAPPING_INTERVAL_INPUT)
    frozen_intervals = pd.read_csv(FROZEN_INTERVAL_INPUT)
    daily = pd.read_csv(DAILY_INPUT)
    daily_audit = read_json(DAILY_AUDIT_INPUT)
    external_qa = read_json(EXTERNAL_QA_INPUT)
    stage1_qa = read_json(STAGE1_QA_INPUT)

    require_columns(
        matrix,
        {
            "carbon_mapping",
            "capacity_scenario",
            "migration_limit",
            "max_latency_ms",
            "carbon_change_pct_vs_matched_scenario_status_quo",
            "service_rate_difference_vs_status_quo",
            "reoptimized_matched_carbon_change_pct",
        },
        "carbon-service sensitivity matrix",
    )
    require_columns(
        mapping_intervals,
        {
            "service_mapping",
            "metric",
            "estimate",
            "block_interval_low_2_5pct",
            "block_interval_high_97_5pct",
            "replicates",
            "block_hours",
            "seed",
        },
        "external mapping sensitivity intervals",
    )
    require_columns(
        frozen_intervals,
        {
            "metric",
            "estimate",
            "block_interval_low_2_5pct",
            "block_interval_high_97_5pct",
            "replicates",
            "block_hours",
            "seed",
        },
        "frozen external intervals",
    )
    require_columns(
        daily,
        {
            "service_mapping",
            "date_utc",
            "e1_vs_e0_carbon_change_pct",
            "e2_vs_e0_carbon_change_pct",
            "e1_vs_e0_high_stress_water_change_l",
        },
        "external daily effects",
    )

    expected_axes = {
        "carbon_mapping": set(CARBON_ORDER),
        "capacity_scenario": set(CAPACITY_ORDER),
        "migration_limit": set(MIGRATION_ORDER),
        "max_latency_ms": set(LATENCY_ORDER),
    }
    if len(matrix) != 81:
        raise ValueError(f"Expected 81 sensitivity runs; found {len(matrix)}")
    for column, expected in expected_axes.items():
        actual = set(matrix[column].unique())
        if actual != expected:
            raise ValueError(f"Unexpected {column} levels: {sorted(actual)}")
    if matrix[["carbon_mapping", "capacity_scenario", "migration_limit", "max_latency_ms"]].duplicated().any():
        raise ValueError("Sensitivity matrix contains duplicate factorial cells")
    if matrix["reoptimized_matched_carbon_change_pct"].ge(0).any():
        raise ValueError("Matched-service robustness no longer has a negative carbon change in every cell")

    sensitivity = matrix.copy()
    sensitivity["panel"] = "a,b"
    sensitivity["record_type"] = "sensitivity"
    sensitivity["migration_limit_pct"] = 100.0 * sensitivity["migration_limit"]
    sensitivity["service_rate_difference_pp"] = 100.0 * sensitivity["service_rate_difference_vs_status_quo"]
    sensitivity["raw_carbon_change_pct"] = sensitivity["carbon_change_pct_vs_matched_scenario_status_quo"]
    sensitivity["matched_carbon_change_pct"] = sensitivity["reoptimized_matched_carbon_change_pct"]
    sensitivity["raw_positive"] = sensitivity["raw_carbon_change_pct"] > 0
    sensitivity["row_order"] = [
        CARBON_ORDER.index(mapping) * 3 + CAPACITY_ORDER.index(capacity)
        for mapping, capacity in zip(sensitivity["carbon_mapping"], sensitivity["capacity_scenario"], strict=True)
    ]
    sensitivity["column_order"] = [
        MIGRATION_ORDER.index(migration) * 3 + LATENCY_ORDER.index(latency)
        for migration, latency in zip(sensitivity["migration_limit"], sensitivity["max_latency_ms"], strict=True)
    ]

    frozen_row = frozen_intervals.loc[frozen_intervals["metric"].eq("e1_vs_e0_carbon_change_pct")].copy()
    if len(frozen_row) != 1:
        raise ValueError("Expected one frozen E1/E0 carbon interval")
    frozen_row["service_mapping"] = "frozen_global_upper"
    mapping_rows = mapping_intervals.loc[mapping_intervals["metric"].eq("e1_vs_e0_carbon_change_pct")].copy()
    forest = pd.concat([frozen_row, mapping_rows], ignore_index=True)
    if set(forest["service_mapping"]) != set(MAPPING_ORDER) or len(forest) != 3:
        raise ValueError("External forest must contain the frozen and two sensitivity mappings")
    forest["panel"] = "c"
    forest["record_type"] = "external_interval"
    forest["external_mapping"] = forest["service_mapping"]
    forest["interval_low_pct"] = forest["block_interval_low_2_5pct"]
    forest["interval_high_pct"] = forest["block_interval_high_97_5pct"]
    forest["estimate_pct"] = forest["estimate"]
    forest["row_order"] = forest["external_mapping"].map({name: i for i, name in enumerate(MAPPING_ORDER)})
    if forest["interval_high_pct"].ge(0).any():
        raise ValueError("At least one external carbon interval now reaches or exceeds zero")
    if set(forest["replicates"].astype(int)) != {4000} or set(forest["block_hours"].astype(int)) != {24} or set(forest["seed"].astype(int)) != {20260720}:
        raise ValueError("External interval method drifted from 4,000 x 24 h, seed 20260720")

    direction_records: list[dict[str, object]] = []
    for order, mapping in enumerate(MAPPING_ORDER):
        subset = daily.loc[daily["service_mapping"].eq(mapping)].copy()
        if len(subset) != 7 or subset["date_utc"].nunique() != 7:
            raise ValueError(f"Expected seven unique UTC days for {mapping}")
        summary = {
            "panel": "d",
            "record_type": "daily_direction",
            "external_mapping": mapping,
            "row_order": order,
            "days_total": int(len(subset)),
            "negative_e1_carbon_days": int(subset["e1_vs_e0_carbon_change_pct"].lt(0).sum()),
            "negative_e2_carbon_days": int(subset["e2_vs_e0_carbon_change_pct"].lt(0).sum()),
            "positive_e1_exposure_days": int(subset["e1_vs_e0_high_stress_water_change_l"].gt(0).sum()),
        }
        audit = daily_audit["mappings"][mapping]
        if summary["negative_e1_carbon_days"] != audit["days_e1_carbon_lower_than_e0"]:
            raise ValueError(f"Daily E1 carbon audit mismatch for {mapping}")
        if summary["negative_e2_carbon_days"] != audit["days_e2_carbon_lower_than_e0"]:
            raise ValueError(f"Daily E2 carbon audit mismatch for {mapping}")
        if summary["positive_e1_exposure_days"] != audit["days_e1_high_stress_water_above_e0"]:
            raise ValueError(f"Daily E1 exposure audit mismatch for {mapping}")
        direction_records.append(summary)
    directions = pd.DataFrame(direction_records)

    fallback_share = float(external_qa["calibration_limitation"]["boundary_fallback_service_share"])
    fallback = pd.DataFrame(
        [
            {
                "panel": "e",
                "record_type": "external_service_origin",
                "segment": "Boundary fallback",
                "share_pct": 100.0 * fallback_share,
                "row_order": 0,
            },
            {
                "panel": "e",
                "record_type": "external_service_origin",
                "segment": "In-grid interpolation",
                "share_pct": 100.0 * (1.0 - fallback_share),
                "row_order": 1,
            },
        ]
    )
    reversal_values = stage1_qa["reversal_test"]
    reversal = pd.DataFrame(
        [
            {
                "panel": "f",
                "record_type": "carbon_reversal",
                "condition": "Original carbon field",
                "migration_share_pct": 100.0 * float(reversal_values["original_migration_share"]),
                "allocation_l1_accelerator_hours": float(reversal_values["allocation_l1_difference_accelerator_hours"]),
                "row_order": 0,
            },
            {
                "panel": "f",
                "record_type": "carbon_reversal",
                "condition": "Reversed carbon field",
                "migration_share_pct": 100.0 * float(reversal_values["reversed_migration_share"]),
                "allocation_l1_accelerator_hours": float(reversal_values["allocation_l1_difference_accelerator_hours"]),
                "row_order": 1,
            },
        ]
    )

    columns = [
        "panel",
        "record_type",
        "row_order",
        "column_order",
        "carbon_mapping",
        "capacity_scenario",
        "migration_limit_pct",
        "max_latency_ms",
        "raw_carbon_change_pct",
        "service_rate_difference_pp",
        "matched_carbon_change_pct",
        "raw_positive",
        "external_mapping",
        "estimate_pct",
        "interval_low_pct",
        "interval_high_pct",
        "replicates",
        "block_hours",
        "seed",
        "days_total",
        "negative_e1_carbon_days",
        "negative_e2_carbon_days",
        "positive_e1_exposure_days",
        "segment",
        "share_pct",
        "condition",
        "migration_share_pct",
        "allocation_l1_accelerator_hours",
        "evidence_id",
    ]
    sensitivity_source = sensitivity.assign(evidence_id="T3C01")
    forest_source = forest.assign(evidence_id="T3C05")
    directions = directions.assign(evidence_id="T3C05")
    fallback = fallback.assign(evidence_id="T1R03;T3C06")
    reversal = reversal.assign(evidence_id="T3C07")
    source = pd.concat(
        [sensitivity_source, forest_source, directions, fallback, reversal],
        ignore_index=True,
        sort=False,
    ).reindex(columns=columns)
    source.to_csv(SOURCE_CSV, index=False, float_format="%.15g")

    raw_positive_count = int(sensitivity["raw_positive"].sum())
    max_service_difference_pp = float(sensitivity["service_rate_difference_pp"].max())
    summary = {
        "source_rows": int(len(source)),
        "sensitivity_runs": int(len(sensitivity)),
        "matched_carbon_change_range_pct": [
            float(sensitivity["matched_carbon_change_pct"].min()),
            float(sensitivity["matched_carbon_change_pct"].max()),
        ],
        "matched_carbon_negative_count": int(sensitivity["matched_carbon_change_pct"].lt(0).sum()),
        "raw_carbon_change_range_pct": [
            float(sensitivity["raw_carbon_change_pct"].min()),
            float(sensitivity["raw_carbon_change_pct"].max()),
        ],
        "raw_positive_count": raw_positive_count,
        "max_service_difference_pp": max_service_difference_pp,
        "external_e1_carbon_intervals": forest.sort_values("row_order")[["external_mapping", "estimate_pct", "interval_low_pct", "interval_high_pct"]].to_dict(orient="records"),
        "daily_direction_counts": directions.sort_values("row_order")[["external_mapping", "negative_e1_carbon_days", "negative_e2_carbon_days", "positive_e1_exposure_days", "days_total"]].to_dict(orient="records"),
        "external_service_origin_pct": {row["segment"]: float(row["share_pct"]) for _, row in fallback.iterrows()},
        "reversal_migration_share_pct": {row["condition"]: float(row["migration_share_pct"]) for _, row in reversal.iterrows()},
        "reversal_allocation_l1_accelerator_hours": float(reversal_values["allocation_l1_difference_accelerator_hours"]),
    }
    assert_close("matched minimum", summary["matched_carbon_change_range_pct"][0], -48.07968977449586)
    assert_close("matched maximum", summary["matched_carbon_change_range_pct"][1], -0.3876706182690426)
    assert_close("max service difference pp", max_service_difference_pp, 1.7700947562941605)
    assert_close("fallback pct", summary["external_service_origin_pct"]["Boundary fallback"], 97.17081756044886)
    assert_close("reversed migration pct", summary["reversal_migration_share_pct"]["Reversed carbon field"], 4.701838770355915)

    assertions = {
        "source_has_91_rows": bool(len(source) == 91),
        "complete_9_by_9_sensitivity_grid": bool(len(sensitivity) == 81 and sensitivity["row_order"].nunique() == 9 and sensitivity["column_order"].nunique() == 9),
        "matched_carbon_negative_81_of_81": bool(summary["matched_carbon_negative_count"] == 81),
        "raw_positive_count_is_3": bool(raw_positive_count == 3),
        "raw_positive_cases_are_15_ms": bool(sensitivity.loc[sensitivity["raw_positive"], "max_latency_ms"].eq(15.0).all()),
        "external_interval_upper_bounds_below_zero": bool(forest["interval_high_pct"].lt(0).all()),
        "external_interval_method_frozen": bool(set(forest["replicates"].astype(int)) == {4000} and set(forest["block_hours"].astype(int)) == {24} and set(forest["seed"].astype(int)) == {20260720}),
        "daily_counts_are_7_7_4_for_all_mappings": bool(((directions["negative_e1_carbon_days"] == 7) & (directions["negative_e2_carbon_days"] == 7) & (directions["positive_e1_exposure_days"] == 4)).all()),
        "external_origin_sums_to_100_pct": bool(np.isclose(fallback["share_pct"].sum(), 100.0, atol=1e-9, rtol=0.0)),
        "reversal_responded": bool(reversal_values["responded_to_reversal"]),
        "panel_evidence_ids_match_frozen_manifest": bool(
            sensitivity_source["evidence_id"].eq("T3C01").all()
            and forest_source["evidence_id"].eq("T3C05").all()
            and directions["evidence_id"].eq("T3C05").all()
            and fallback["evidence_id"].eq("T1R03;T3C06").all()
            and reversal["evidence_id"].eq("T3C07").all()
        ),
        "no_absolute_external_footprint_is_plotted": True,
    }
    if not all(assertions.values()):
        failed = [key for key, passed in assertions.items() if not passed]
        raise ValueError(f"Fig. 4 source assertions failed: {failed}")

    inputs = [
        (MATRIX_INPUT, "81-run carbon/service sensitivity matrix"),
        (MAPPING_INTERVAL_INPUT, "post-confirmation mapping intervals"),
        (FROZEN_INTERVAL_INPUT, "frozen external paired-block intervals"),
        (DAILY_INPUT, "seven-day external direction records"),
        (DAILY_AUDIT_INPUT, "external daily direction audit"),
        (EXTERNAL_QA_INPUT, "external confirmation calibration limitation"),
        (STAGE1_QA_INPUT, "reversal diagnostic"),
    ]
    manifest = {
        "figure": "Fig. 4",
        "derived_source": {
            "path": SOURCE_CSV.name,
            "rows": int(len(source)),
            "panel_filters": {
                "a": "record_type == sensitivity; matched-service carbon field",
                "b": "record_type == sensitivity; raw carbon and service-rate fields",
                "c": "record_type == external_interval",
                "d": "record_type == daily_direction",
                "e": "record_type == external_service_origin",
                "f": "record_type == carbon_reversal",
            },
            "sha256": sha256(SOURCE_CSV),
        },
        "upstream_inputs": [
            {"role": role, "path": path.name, "sha256": sha256(path)} for path, role in inputs
        ],
        "derivation": {
            "panel_a": "negative reoptimized_matched_carbon_change_pct values arranged as carbon mapping x capacity rows and migration limit x latency columns",
            "panel_b": "raw carbon change plotted against 100 x service_rate_difference_vs_status_quo; diagnostic only, with no fitted causal relation",
            "panel_c": "E1/E0 carbon paired circular-block intervals from the frozen mapping and two post-confirmation mapping sensitivities",
            "panel_d": "counts of UTC days satisfying each directional comparison, recomputed from 21 daily records and cross-checked against the audit JSON",
            "panel_e": "boundary fallback share from the frozen external calibration limitation; complement is in-grid interpolation",
            "panel_f": "migration shares and allocation L1 response from the deterministic carbon-field reversal diagnostic",
        },
        "scientific_summary": summary,
        "assertions": assertions,
    }
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return source, summary


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(direction="out", colors=COLORS["ink"], pad=2)


def panel_heading(ax: plt.Axes, letter: str, title: str, x_letter: float = -0.12) -> None:
    ax.text(x_letter, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom", color=COLORS["ink"], clip_on=False)
    ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom", color=COLORS["ink"], clip_on=False)


def draw_figure(source: pd.DataFrame, summary: dict[str, object]) -> tuple[plt.Figure, dict[str, object]]:
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH))
    grid = fig.add_gridspec(
        4,
        6,
        left=0.145,
        right=0.985,
        bottom=0.065,
        top=0.950,
        wspace=1.10,
        hspace=0.90,
        height_ratios=[1.10, 0.95, 0.95, 0.72],
    )
    ax_a = fig.add_subplot(grid[0:3, 0:3])
    ax_b = fig.add_subplot(grid[0, 3:6])
    ax_c = fig.add_subplot(grid[1, 3:6])
    ax_d = fig.add_subplot(grid[2, 3:6])
    ax_e = fig.add_subplot(grid[3, 0:3])
    ax_f = fig.add_subplot(grid[3, 3:6])

    sensitivity = source.loc[source["record_type"].eq("sensitivity")].copy()
    sensitivity["row_order"] = sensitivity["row_order"].astype(int)
    sensitivity["column_order"] = sensitivity["column_order"].astype(int)
    heat = sensitivity.pivot(index="row_order", columns="column_order", values="matched_carbon_change_pct").sort_index().sort_index(axis=1)
    reduction = -heat.to_numpy()
    image = ax_a.imshow(reduction, cmap=HEATMAP_CMAP, vmin=0, vmax=50, interpolation="nearest", aspect="equal")
    ax_a.set_anchor("N")
    row_labels = [f"{CARBON_LABEL[carbon]} | {CAPACITY_LABEL[capacity]}" for carbon in CARBON_ORDER for capacity in CAPACITY_ORDER]
    column_labels = [f"{migration * 100:.0f}%\n{latency:.0f} ms" for migration in MIGRATION_ORDER for latency in LATENCY_ORDER]
    ax_a.set_xticks(np.arange(9), column_labels)
    ax_a.set_yticks(np.arange(9), row_labels)
    ax_a.set_xlabel("Migration limit / latency")
    ax_a.set_ylabel("Carbon mapping / capacity")
    ax_a.tick_params(length=0, pad=2)
    for boundary in (2.5, 5.5):
        ax_a.axhline(boundary, color=COLORS["white"], lw=1.8)
        ax_a.axvline(boundary, color=COLORS["white"], lw=1.8)
    for row in range(9):
        for column in range(9):
            value = float(heat.iloc[row, column])
            text_color = COLORS["white"] if -value >= 24 else COLORS["ink"]
            ax_a.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=5.25, color=text_color)
    panel_heading(ax_a, "a", "Matched-service carbon robustness", x_letter=-0.21)
    colorbar_axis = ax_a.inset_axes([0.0, -0.235, 1.0, 0.035])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Magnitude of carbon reduction (%)", labelpad=2)
    colorbar.set_ticks([0, 10, 20, 30, 40, 50])
    colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    colorbar.outline.set_linewidth(0.6)

    raw_positive = sensitivity["raw_positive"].astype(str).str.lower().eq("true")
    ax_b.scatter(
        sensitivity.loc[~raw_positive, "service_rate_difference_pp"],
        sensitivity.loc[~raw_positive, "raw_carbon_change_pct"],
        s=18,
        color=COLORS["blue"],
        alpha=0.72,
        edgecolor=COLORS["white"],
        linewidth=0.35,
        zorder=2,
    )
    ax_b.scatter(
        sensitivity.loc[raw_positive, "service_rate_difference_pp"],
        sensitivity.loc[raw_positive, "raw_carbon_change_pct"],
        s=32,
        marker="^",
        color=COLORS["red"],
        edgecolor=COLORS["ink"],
        linewidth=0.45,
        zorder=3,
    )
    ax_b.axhline(0, color=COLORS["reference"], lw=0.8, ls="--", zorder=1)
    ax_b.axvline(0, color=COLORS["reference"], lw=0.7, zorder=1)
    ax_b.set_xlim(-0.08, 1.88)
    ax_b.set_ylim(-50.5, 5.4)
    ax_b.set_xlabel("Service-rate difference vs E0 (pp)", labelpad=2)
    ax_b.set_ylabel("Raw carbon change (%)", labelpad=2)
    ax_b.set_xticks([0, 0.5, 1.0, 1.5])
    ax_b.legend(
        handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["blue"], markeredgecolor="none", markersize=4.5, label="Raw carbon <= 0"),
            Line2D([], [], marker="^", color="none", markerfacecolor=COLORS["red"], markeredgecolor=COLORS["ink"], markeredgewidth=0.4, markersize=5, label="Raw carbon > 0 (n=3)"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.01),
        fontsize=5.8,
        borderaxespad=0,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    ax_b.text(0.60, 0.88, "Max service difference: 1.770 pp", transform=ax_b.transAxes, ha="center", va="top", fontsize=5.8, color=COLORS["muted"])
    panel_heading(ax_b, "b", "Raw carbon-service diagnostic", x_letter=-0.16)
    style_axis(ax_b)

    forest = source.loc[source["record_type"].eq("external_interval")].copy().sort_values("row_order")
    y = np.arange(len(forest))[::-1]
    forest_colors = [COLORS["blue"], COLORS["mapping_teal"], COLORS["gold"]]
    forest_markers = ["o", "s", "D"]
    for ypos, (_, row), color, marker in zip(y, forest.iterrows(), forest_colors, forest_markers, strict=True):
        estimate = float(row["estimate_pct"])
        low = float(row["interval_low_pct"])
        high = float(row["interval_high_pct"])
        ax_c.errorbar(
            estimate,
            ypos,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt=marker,
            ms=4.3,
            mfc=color,
            mec=COLORS["ink"],
            mew=0.35,
            ecolor=color,
            elinewidth=1.15,
            capsize=2.2,
            capthick=0.8,
            zorder=3,
        )
    ax_c.set_yticks(y, [MAPPING_LABEL[name] for name in MAPPING_ORDER])
    ax_c.set_xlim(-16.20, -13.75)
    ax_c.set_xticks([-16, -15, -14])
    ax_c.set_xlabel("E1 vs E0 carbon change (%)", labelpad=2)
    ax_c.grid(axis="x", color=COLORS["pale"], lw=0.6, zorder=0)
    ax_c.text(0.99, 1.04, "All interval upper bounds < 0", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=5.8, color=COLORS["muted"], clip_on=False)
    panel_heading(ax_c, "c", "External paired-block intervals", x_letter=-0.16)
    style_axis(ax_c)

    direction = source.loc[source["record_type"].eq("daily_direction")].copy().sort_values("row_order")
    x = np.arange(len(direction))
    width = 0.23
    metrics = [
        ("negative_e1_carbon_days", "E1 carbon < E0", COLORS["blue"], None),
        ("negative_e2_carbon_days", "E2 carbon < E0", COLORS["e2"], "//"),
        ("positive_e1_exposure_days", "E1 exposure > E0", COLORS["red"], ".."),
    ]
    for offset, (column, label, color, hatch) in zip((-width, 0, width), metrics, strict=True):
        values = direction[column].astype(float).to_numpy()
        bars = ax_d.bar(x + offset, values, width=width, color=color, edgecolor=COLORS["ink"], linewidth=0.35, hatch=hatch, label=label)
        for bar, value in zip(bars, values, strict=True):
            ax_d.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{int(value)}/7", ha="center", va="bottom", fontsize=5.3, color=COLORS["ink"])
    ax_d.set_xticks(x, ["Frozen", "Triang.", "Simplex"])
    ax_d.set_ylim(0, 9.4)
    ax_d.set_yticks([0, 2, 4, 6, 7])
    ax_d.set_ylabel("UTC days")
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncols=3, fontsize=5.35, handlelength=1.2, columnspacing=0.65, handletextpad=0.3)
    panel_heading(ax_d, "d", "Daily directional consistency", x_letter=-0.16)
    style_axis(ax_d)

    origin = source.loc[source["record_type"].eq("external_service_origin")].copy().sort_values("row_order")
    left = 0.0
    segment_colors = [COLORS["muted"], COLORS["blue_light"]]
    for (_, row), color in zip(origin.iterrows(), segment_colors, strict=True):
        share = float(row["share_pct"])
        ax_e.barh([0], [share], left=left, height=0.42, color=color, edgecolor=COLORS["white"], linewidth=0.7)
        if share > 12:
            ax_e.text(left + share / 2, 0, f"{row['segment']}  {share:.3f}%", ha="center", va="center", fontsize=6.1, color=COLORS["white"])
        else:
            ax_e.annotate(
                f"In-grid {share:.3f}%",
                xy=(left + share / 2, 0.21),
                xytext=(98.5, 0.58),
                ha="right",
                va="bottom",
                fontsize=5.8,
                color=COLORS["blue"],
                arrowprops={"arrowstyle": "-", "color": COLORS["blue"], "lw": 0.6},
            )
        left += share
    ax_e.set_xlim(0, 100)
    ax_e.set_ylim(-0.55, 0.85)
    ax_e.set_yticks([])
    ax_e.set_xticks([0, 50, 100])
    ax_e.set_xlabel("Share of externally served accelerator-hours (%)", labelpad=2)
    ax_e.text(0.0, -0.36, "Absolute external footprint scale is not interpretable", transform=ax_e.transAxes, ha="left", va="top", fontsize=5.8, color=COLORS["red"])
    panel_heading(ax_e, "e", "External service origin", x_letter=-0.21)
    style_axis(ax_e)

    reversal = source.loc[source["record_type"].eq("carbon_reversal")].copy().sort_values("row_order")
    rx = np.arange(2)
    values = reversal["migration_share_pct"].astype(float).to_numpy()
    bars = ax_f.bar(rx, values, width=0.52, color=[COLORS["blue"], COLORS["gold"]], edgecolor=COLORS["ink"], linewidth=0.45)
    for bar, value in zip(bars, values, strict=True):
        ax_f.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.3f}%", ha="center", va="bottom", fontsize=6.0)
    ax_f.set_xticks(rx, ["Original field", "Reversed field"])
    ax_f.set_ylim(0, 35.5)
    ax_f.set_yticks([0, 10, 20, 30])
    ax_f.set_ylabel("Migration share (%)")
    ax_f.text(0.98, 0.92, "Allocation L1 = 111.466 acc-h", transform=ax_f.transAxes, ha="right", va="top", fontsize=5.8, color=COLORS["muted"])
    panel_heading(ax_f, "f", "Carbon-field reversal response", x_letter=-0.16)
    style_axis(ax_f)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    outside: list[dict[str, object]] = []
    visible_text_count = 0
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        visible_text_count += 1
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > canvas_width + 1 or bbox.y1 > canvas_height + 1:
            outside.append({"text": artist.get_text(), "bbox_px": [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]})
    layout = {
        "canvas_width_px_at_render_dpi": float(canvas_width),
        "canvas_height_px_at_render_dpi": float(canvas_height),
        "visible_text_artist_count": visible_text_count,
        "text_outside_canvas_count": len(outside),
        "text_outside_canvas": outside,
    }
    if outside:
        raise ValueError(f"Text outside canvas: {outside}")
    return fig, layout


def save_outputs(fig: plt.Figure) -> None:
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), dpi=300, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def inspect_outputs(layout: dict[str, object], summary: dict[str, object]) -> dict[str, object]:
    svg_path = OUTPUT_STEM.with_suffix(".svg")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    png_path = OUTPUT_STEM.with_suffix(".png")
    tiff_path = OUTPUT_STEM.with_suffix(".tiff")

    svg_root = ET.parse(svg_path).getroot()
    svg_text_nodes = svg_root.findall(".//{http://www.w3.org/2000/svg}text")
    pdf = PdfReader(str(pdf_path))
    page = pdf.pages[0]
    width_mm = float(page.mediabox.width) / 72.0 * MM_PER_INCH
    height_mm = float(page.mediabox.height) / 72.0 * MM_PER_INCH
    fonts: list[dict[str, str]] = []
    type3_count = 0
    resources = page.get("/Resources")
    if resources and "/Font" in resources:
        for resource_name, font_ref in resources["/Font"].items():
            font = font_ref.get_object()
            subtype = str(font.get("/Subtype", ""))
            basefont = str(font.get("/BaseFont", ""))
            fonts.append({"resource": str(resource_name), "subtype": subtype, "basefont": basefont})
            if subtype == "/Type3":
                type3_count += 1
    with Image.open(png_path) as image:
        png_metadata = {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "format": image.format,
            "dpi": [float(value) for value in image.info.get("dpi", (0, 0))],
        }
        rgb = image.convert("RGB")
        extrema = rgb.getextrema()
        nonblank = any(low < 250 for low, _ in extrema)
    with Image.open(tiff_path) as image:
        tiff_metadata = {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "format": image.format,
            "dpi": [float(value) for value in image.info.get("dpi", (0, 0))],
            "compression": image.info.get("compression"),
        }

    outputs = {
        "svg": {
            "path": svg_path.name,
            "bytes": svg_path.stat().st_size,
            "sha256": sha256(svg_path),
            "metadata": {
                "width": svg_root.attrib.get("width"),
                "height": svg_root.attrib.get("height"),
                "viewBox": svg_root.attrib.get("viewBox"),
                "text_node_count": len(svg_text_nodes),
            },
        },
        "pdf": {
            "path": pdf_path.name,
            "bytes": pdf_path.stat().st_size,
            "sha256": sha256(pdf_path),
            "metadata": {"width_mm": width_mm, "height_mm": height_mm, "fonts": fonts, "type3_font_count": type3_count},
        },
        "tiff": {"path": tiff_path.name, "bytes": tiff_path.stat().st_size, "sha256": sha256(tiff_path), "metadata": tiff_metadata},
        "png": {"path": png_path.name, "bytes": png_path.stat().st_size, "sha256": sha256(png_path), "metadata": png_metadata},
    }
    qa = {
        "svg_editable_text": len(svg_text_nodes) > 0,
        "pdf_size_matches_mm": abs(width_mm - FIGURE_WIDTH_MM) < 0.02 and abs(height_mm - FIGURE_HEIGHT_MM) < 0.02,
        "pdf_has_no_type3_fonts": type3_count == 0,
        "tiff_nominal_600_dpi": all(abs(value - 600) < 0.1 for value in tiff_metadata["dpi"]),
        "tiff_lzw_compression": tiff_metadata["compression"] == "tiff_lzw",
        "png_nominal_300_dpi": all(abs(value - 300) < 0.1 for value in png_metadata["dpi"]),
        "png_nonblank": nonblank,
        "no_text_outside_canvas": layout["text_outside_canvas_count"] == 0,
        "source_has_expected_91_rows": summary["source_rows"] == 91,
        "matched_carbon_negative_81_of_81": summary["matched_carbon_negative_count"] == 81,
    }
    if not all(qa.values()):
        failed = [key for key, passed in qa.items() if not passed]
        raise ValueError(f"Fig. 4 export QA failed: {failed}")
    return {"outputs": outputs, "qa_contract": qa}


def main() -> None:
    source, summary = build_source_data()
    fig, layout = draw_figure(source, summary)
    save_outputs(fig)
    plt.close(fig)
    inspected = inspect_outputs(layout, summary)
    manifest = {
        "figure": "Fig. 4",
        "title": "Matched-service and external diagnostics bound the robustness of carbon-aware routing",
        "backend": "Python / matplotlib Agg",
        "intended_size_mm": {"width": FIGURE_WIDTH_MM, "height": FIGURE_HEIGHT_MM},
        "source_data": {
            "path": SOURCE_CSV.name,
            "rows": int(len(source)),
            "panel_filters": {
                "a_b": "81 sensitivity records",
                "c": "3 external interval records",
                "d": "3 daily direction summaries",
                "e": "2 external service-origin segments",
                "f": "2 reversal conditions",
            },
            "sha256": sha256(SOURCE_CSV),
        },
        "scientific_summary": summary,
        "layout": layout,
        **inspected,
    }
    EXPORT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "source_rows": len(source), "outputs": {key: value["path"] for key, value in inspected["outputs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
