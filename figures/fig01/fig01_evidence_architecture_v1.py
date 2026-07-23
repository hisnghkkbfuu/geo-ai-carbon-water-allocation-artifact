from __future__ import annotations

import hashlib
import json
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from pypdf import PdfReader


MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 150

HERE = Path(__file__).resolve().parent
REWRITE_DIR = HERE.parents[1]
SOURCE_CSV = HERE / "fig01_source_data_v1.csv"
SOURCE_MANIFEST_JSON = HERE / "fig01_source_manifest_v1.json"
OUTPUT_STEM = HERE / "fig01_evidence_architecture_v1"
MANIFEST_JSON = HERE / "fig01_export_manifest_v1.json"

EXPECTED_IDS = {
    "a_gen", "a_azure", "a_service", "a_vidur", "a_scenarios", "a_alloc", "a_outputs",
    "b_service", "b_it", "b_facility", "b_carbon", "b_water", "b_exposure",
    "c_stage1", "c_stage2", "c_e0", "c_e1", "c_e2",
}

COLORS = {
    "ink": "#202124",
    "muted": "#5F6368",
    "line": "#7A8085",
    "observed": "#EAF2F7",
    "observed_edge": "#5B7C91",
    "proxy": "#F3F0E6",
    "proxy_edge": "#8A7951",
    "scenario": "#F7EEE8",
    "scenario_edge": "#A66B4F",
    "model": "#EEF1F0",
    "model_edge": "#596C67",
    "output": "#F4F4F4",
    "e0": "#6E7378",
    "e1": "#225E91",
    "e2": "#00857C",
    "stress": "#B44A3E",
    "white": "#FFFFFF",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 0.9,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def row_map(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    return frame.set_index("element_id").fillna("").to_dict(orient="index")


def write_source_manifest(source: pd.DataFrame) -> None:
    panel_counts = source["panel"].value_counts().sort_index().astype(int).to_dict()
    evidence_counts = source["evidence_class"].value_counts().sort_index().astype(int).to_dict()
    assertions = {
        "source_has_18_rows": bool(len(source) == 18),
        "panel_counts_are_7_6_5": bool(panel_counts == {"a": 7, "b": 6, "c": 5}),
        "element_ids_are_unique": bool(source["element_id"].is_unique),
        "element_ids_match_frozen_contract": bool(set(source["element_id"]) == EXPECTED_IDS),
        "required_fields_are_nonempty": bool(source[["panel", "element_id", "evidence_class", "label", "detail", "boundary"]].ne("").all().all()),
        "policy_rows_are_e0_e1_e2": bool(source.loc[source["evidence_class"].eq("policy_definition"), "label"].tolist() == ["E0", "E1", "E2"]),
    }
    if not all(assertions.values()):
        failed = [key for key, passed in assertions.items() if not passed]
        raise ValueError(f"Fig. 1 source assertions failed: {failed}")
    upstream_paths = [
        REWRITE_DIR / "01_claim_ledger_v1.md",
        REWRITE_DIR / "30_tables_1_3_english_evidence_v1.md",
        REWRITE_DIR / "34_figures_1_4_contract_v1.md",
        REWRITE_DIR / "35_figure_source_data_manifest_v1.csv",
    ]
    manifest = {
        "figure": "Fig. 1",
        "derived_source": {
            "path": SOURCE_CSV.name,
            "rows": int(len(source)),
            "panel_counts": panel_counts,
            "sha256": sha256(SOURCE_CSV),
        },
        "upstream_inputs": [
            {"path": path.name, "sha256": sha256(path)} for path in upstream_paths
        ],
        "derivation": {
            "status": "curated schematic source table",
            "panel_a": "frozen evidence-layer and system-boundary nodes",
            "panel_b": "frozen physical accounting chain and screening boundary",
            "panel_c": "frozen lexicographic stages and E0/E1/E2 policy definitions",
        },
        "scientific_summary": {
            "panels": ["a", "b", "c"],
            "panel_counts": panel_counts,
            "evidence_class_counts": evidence_counts,
            "policy_encoding": {"E0": "neutral grey circle", "E1": "deep blue square", "E2": "teal diamond"},
        },
        "assertions": assertions,
    }
    SOURCE_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def add_group(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    fill: str,
    edge: str,
    dashed: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=fill,
        edgecolor=edge,
        linewidth=0.75,
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.012,
        y + height - 0.035,
        title.upper(),
        ha="left",
        va="top",
        fontsize=5.8,
        fontweight="bold",
        color=edge,
        zorder=3,
    )


def add_node(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    *,
    facecolor: str = COLORS["white"],
    edgecolor: str = COLORS["line"],
    title_color: str = COLORS["ink"],
    detail_color: str = COLORS["muted"],
    title_size: float = 6.7,
    detail_size: float = 5.6,
    wrap_width: int = 24,
    linewidth: float = 0.7,
    linestyle: str | tuple = "solid",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.009",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.012,
        y + height - 0.038,
        wrap(title, wrap_width),
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        linespacing=1.05,
        zorder=3,
    )
    ax.text(
        x + 0.012,
        y + 0.035,
        wrap(detail.replace(" | ", "\n"), wrap_width),
        ha="left",
        va="bottom",
        fontsize=detail_size,
        color=detail_color,
        linespacing=1.12,
        zorder=3,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
    label_xy: tuple[float, float] | None = None,
    color: str = COLORS["line"],
    connectionstyle: str = "arc3,rad=0",
    dashed: bool = False,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "mutation_scale": 7,
            "linewidth": 0.75,
            "color": color,
            "linestyle": (0, (3, 2)) if dashed else "solid",
            "shrinkA": 1.5,
            "shrinkB": 1.5,
            "connectionstyle": connectionstyle,
        },
        zorder=1,
    )
    if label and label_xy is not None:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=5.5,
            color=COLORS["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
            zorder=4,
        )


def panel_heading(ax: plt.Axes, letter: str, title: str, subtitle: str = "") -> None:
    ax.text(
        0.0,
        1.025,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )
    ax.text(
        0.035,
        1.025,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )
    if subtitle:
        ax.text(
            0.035,
            0.982,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=COLORS["muted"],
        )


def setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_panel_a(ax: plt.Axes, rows: dict[str, dict[str, str]]) -> None:
    setup_axis(ax)
    panel_heading(ax, "a", "Evidence pipeline")

    add_group(ax, 0.008, 0.08, 0.166, 0.82, "Observed traces", COLORS["observed"], COLORS["observed_edge"])
    add_group(ax, 0.191, 0.08, 0.195, 0.82, "Calibrated proxies", COLORS["proxy"], COLORS["proxy_edge"])
    add_group(ax, 0.404, 0.08, 0.200, 0.82, "Scenario inputs", COLORS["scenario"], COLORS["scenario_edge"], dashed=True)
    add_group(ax, 0.622, 0.08, 0.176, 0.82, "Model", COLORS["model"], COLORS["model_edge"])
    add_group(ax, 0.816, 0.08, 0.176, 0.82, "Outputs", COLORS["output"], COLORS["line"])

    add_arrow(ax, (0.156, 0.665), (0.210, 0.665))
    add_arrow(ax, (0.156, 0.305), (0.210, 0.535), connectionstyle="arc3,rad=-0.12")
    add_arrow(ax, (0.289, 0.355), (0.289, 0.495), label="calibrates", label_xy=(0.335, 0.425), color=COLORS["proxy_edge"])
    add_arrow(ax, (0.368, 0.665), (0.640, 0.665))
    add_arrow(ax, (0.586, 0.480), (0.640, 0.570), color=COLORS["scenario_edge"], dashed=True)
    add_arrow(ax, (0.780, 0.575), (0.834, 0.575), color=COLORS["model_edge"])

    add_node(ax, 0.026, 0.535, 0.130, 0.255, rows["a_gen"]["label"], rows["a_gen"]["detail"], edgecolor=COLORS["observed_edge"], wrap_width=19)
    add_node(ax, 0.026, 0.175, 0.130, 0.255, rows["a_azure"]["label"], rows["a_azure"]["detail"], edgecolor=COLORS["observed_edge"], title_size=6.2, wrap_width=19)
    add_node(ax, 0.210, 0.495, 0.158, 0.295, rows["a_service"]["label"], rows["a_service"]["detail"], edgecolor=COLORS["proxy_edge"], wrap_width=22)
    add_node(ax, 0.210, 0.175, 0.158, 0.180, rows["a_vidur"]["label"], rows["a_vidur"]["detail"], edgecolor=COLORS["proxy_edge"], title_size=6.2, detail_size=5.2, wrap_width=25)
    add_node(ax, 0.422, 0.305, 0.164, 0.350, rows["a_scenarios"]["label"], rows["a_scenarios"]["detail"], edgecolor=COLORS["scenario_edge"], linestyle=(0, (3, 2)), wrap_width=23)
    add_node(ax, 0.640, 0.390, 0.140, 0.370, rows["a_alloc"]["label"], rows["a_alloc"]["detail"], edgecolor=COLORS["model_edge"], wrap_width=19)
    add_node(ax, 0.834, 0.350, 0.140, 0.450, rows["a_outputs"]["label"], rows["a_outputs"]["detail"], edgecolor=COLORS["line"], wrap_width=20)
    ax.text(0.904, 0.200, "Directional mechanism\nclaims only", ha="center", va="center", fontsize=5.8, color=COLORS["muted"], fontweight="bold")


def draw_panel_b(ax: plt.Axes, rows: dict[str, dict[str, str]]) -> None:
    setup_axis(ax)
    panel_heading(ax, "b", "Physical unit chain")

    nodes = {
        "b_service": (0.015, 0.575, 0.175, 0.205),
        "b_it": (0.275, 0.575, 0.175, 0.205),
        "b_facility": (0.555, 0.695, 0.175, 0.205),
        "b_carbon": (0.815, 0.695, 0.170, 0.205),
        "b_water": (0.555, 0.255, 0.175, 0.235),
        "b_exposure": (0.815, 0.255, 0.170, 0.235),
    }

    add_arrow(ax, (0.190, 0.678), (0.275, 0.678), label="x power", label_xy=(0.232, 0.723))
    add_arrow(ax, (0.450, 0.690), (0.555, 0.785), label="x PUE", label_xy=(0.500, 0.780))
    add_arrow(ax, (0.730, 0.798), (0.815, 0.798), label="x CEF", label_xy=(0.773, 0.842))
    add_arrow(ax, (0.450, 0.645), (0.555, 0.378), label="x WUE", label_xy=(0.495, 0.485))
    add_arrow(ax, (0.730, 0.378), (0.815, 0.378), label="screen\nonly", label_xy=(0.773, 0.455), color=COLORS["stress"])

    for key, (x, y, width, height) in nodes.items():
        edge = COLORS["stress"] if key == "b_exposure" else COLORS["line"]
        face = "#FCEFED" if key == "b_exposure" else COLORS["white"]
        title_wrap_width = {
            "b_service": 10,
            "b_it": 14,
            "b_facility": 10,
            "b_carbon": 14,
            "b_water": 12,
            "b_exposure": 12,
        }[key]
        add_node(
            ax,
            x,
            y,
            width,
            height,
            rows[key]["label"],
            rows[key]["detail"],
            facecolor=face,
            edgecolor=edge,
            title_size=6.1,
            detail_size=5.5,
            wrap_width=title_wrap_width,
        )

    ax.add_patch(Rectangle((0.545, 0.645), 0.450, 0.305, fill=False, edgecolor=COLORS["scenario_edge"], linewidth=0.6, linestyle=(0, (3, 2)), zorder=0))
    ax.text(0.555, 0.945, "CARBON BRANCH", ha="left", va="top", fontsize=5.5, fontweight="bold", color=COLORS["scenario_edge"])
    ax.add_patch(Rectangle((0.545, 0.225), 0.450, 0.305, fill=False, edgecolor=COLORS["stress"], linewidth=0.6, linestyle=(0, (3, 2)), zorder=0))
    ax.text(0.555, 0.525, "WATER + AQUEDUCT SCREEN", ha="left", va="top", fontsize=5.5, fontweight="bold", color=COLORS["stress"])
    ax.text(0.500, 0.080, "PUE excluded from water  |  Aqueduct excluded from liters", ha="center", va="center", fontsize=5.8, color=COLORS["ink"], fontweight="bold")


def policy_row(
    ax: plt.Axes,
    y: float,
    label: str,
    detail: str,
    color: str,
    marker: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.055, y),
            0.890,
            0.120,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            facecolor="white",
            edgecolor=color,
            linewidth=0.75,
            zorder=1,
        )
    )
    ax.scatter([0.095], [y + 0.060], s=24, marker=marker, facecolor=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(0.140, y + 0.060, label, ha="left", va="center", fontsize=6.5, fontweight="bold", color=color)
    ax.text(0.245, y + 0.060, detail, ha="left", va="center", fontsize=5.8, color=COLORS["ink"])


def draw_panel_c(ax: plt.Axes, rows: dict[str, dict[str, str]]) -> None:
    setup_axis(ax)
    panel_heading(ax, "c", "Lexicographic objective and policy ladder", "Same-hour allocation; no cross-hour delay")

    add_arrow(ax, (0.440, 0.755), (0.560, 0.755), color=COLORS["model_edge"])
    add_node(ax, 0.055, 0.650, 0.385, 0.210, rows["c_stage1"]["label"], rows["c_stage1"]["detail"], facecolor=COLORS["model"], edgecolor=COLORS["model_edge"], title_size=6.4, detail_size=5.8, wrap_width=31)
    add_node(ax, 0.560, 0.650, 0.385, 0.210, rows["c_stage2"]["label"], rows["c_stage2"]["detail"], facecolor=COLORS["model"], edgecolor=COLORS["model_edge"], title_size=6.4, detail_size=5.5, wrap_width=32)

    ax.text(0.055, 0.565, "REFERENCE OPERATING POINTS", ha="left", va="center", fontsize=5.5, fontweight="bold", color=COLORS["muted"])
    policy_row(ax, 0.400, rows["c_e0"]["label"], rows["c_e0"]["detail"], COLORS["e0"], "o")
    policy_row(ax, 0.235, rows["c_e1"]["label"], rows["c_e1"]["detail"].replace(" | ", "  |  "), COLORS["e1"], "s")
    policy_row(ax, 0.070, rows["c_e2"]["label"], rows["c_e2"]["detail"], COLORS["e2"], "D")


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    rows = row_map(frame)
    figure = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.23, 1.0], width_ratios=[1.03, 0.97])
    draw_panel_a(figure.add_subplot(grid[0, :]), rows)
    draw_panel_b(figure.add_subplot(grid[1, 0]), rows)
    draw_panel_c(figure.add_subplot(grid[1, 1]), rows)
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.045, top=0.960, hspace=0.235, wspace=0.105)
    return figure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def svg_metadata(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    text_nodes = sum(1 for node in root.iter() if node.tag.endswith("text"))
    return {
        "width": root.attrib.get("width"),
        "height": root.attrib.get("height"),
        "viewBox": root.attrib.get("viewBox"),
        "text_node_count": text_nodes,
    }


def pdf_metadata(path: Path) -> dict[str, float]:
    page = PdfReader(str(path)).pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    return {
        "width_mm": round(width_pt / 72 * MM_PER_INCH, 3),
        "height_mm": round(height_pt / 72 * MM_PER_INCH, 3),
    }


def raster_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        return {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "format": image.format,
            "dpi": [round(float(value), 3) for value in image.info.get("dpi", (0, 0))],
        }


def export_figure(figure: plt.Figure, source: pd.DataFrame) -> None:
    outputs = {
        "svg": OUTPUT_STEM.with_suffix(".svg"),
        "pdf": OUTPUT_STEM.with_suffix(".pdf"),
        "tiff": OUTPUT_STEM.with_suffix(".tiff"),
        "png": OUTPUT_STEM.with_suffix(".png"),
    }
    figure.savefig(outputs["svg"], format="svg")
    figure.savefig(outputs["pdf"], format="pdf", metadata={"Title": "Fig. 1 | Evidence and accounting architecture"})
    figure.savefig(outputs["tiff"], format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(outputs["png"], format="png", dpi=300)
    plt.close(figure)

    file_manifest = {
        name: {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in outputs.items()
    }
    file_manifest["svg"]["metadata"] = svg_metadata(outputs["svg"])
    file_manifest["pdf"]["metadata"] = pdf_metadata(outputs["pdf"])
    file_manifest["tiff"]["metadata"] = raster_metadata(outputs["tiff"])
    file_manifest["png"]["metadata"] = raster_metadata(outputs["png"])

    manifest = {
        "figure": "Fig. 1",
        "title": "Evidence and accounting architecture for spatial AI-inference allocation",
        "backend": "Python / matplotlib Agg",
        "intended_size_mm": {"width": FIGURE_WIDTH_MM, "height": FIGURE_HEIGHT_MM},
        "source_data": {
            "path": SOURCE_CSV.name,
            "rows": int(len(source)),
            "panels": sorted(source["panel"].unique().tolist()),
            "sha256": sha256(SOURCE_CSV),
        },
        "outputs": file_manifest,
        "qa_contract": {
            "svg_editable_text": file_manifest["svg"]["metadata"]["text_node_count"] > 0,
            "pdf_size_matches_mm": abs(file_manifest["pdf"]["metadata"]["width_mm"] - FIGURE_WIDTH_MM) < 0.02
            and abs(file_manifest["pdf"]["metadata"]["height_mm"] - FIGURE_HEIGHT_MM) < 0.02,
            "tiff_nominal_600_dpi": all(abs(value - 600) < 1 for value in file_manifest["tiff"]["metadata"]["dpi"]),
            "png_nominal_300_dpi": all(abs(value - 300) < 1 for value in file_manifest["png"]["metadata"]["dpi"]),
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    source = pd.read_csv(SOURCE_CSV, dtype=str, keep_default_na=False)
    actual_ids = set(source["element_id"])
    if actual_ids != EXPECTED_IDS:
        raise ValueError(f"Source-data element mismatch: missing={EXPECTED_IDS - actual_ids}, extra={actual_ids - EXPECTED_IDS}")
    write_source_manifest(source)
    export_figure(build_figure(source), source)
    print(f"Generated {OUTPUT_STEM.name} in SVG, PDF, TIFF, and PNG formats.")
    print(f"QA manifest: {MANIFEST_JSON.name}")


if __name__ == "__main__":
    main()
