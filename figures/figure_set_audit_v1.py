from __future__ import annotations

import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops
from pypdf import PdfReader


MM_PER_INCH = 25.4
HERE = Path(__file__).resolve().parent
REWRITE_DIR = HERE.parent
PANEL_CONTRACT = REWRITE_DIR / "35_figure_source_data_manifest_v1.csv"
CLAIM_TABLE = REWRITE_DIR / "30_tables_1_3_english_evidence_v1.md"
MANUSCRIPT = REWRITE_DIR / "27_full_english_manuscript_numbered_citations_v1.md"
OUTPUT_CSV = REWRITE_DIR / "50_figure_set_manifest_v1.csv"
OUTPUT_JSON = HERE / "figure_set_manifest_v1.json"

EXPECTED = {
    1: {
        "dir": "fig01",
        "stem": "fig01_evidence_architecture_v1",
        "source": "fig01_source_data_v1.csv",
        "source_manifest": "fig01_source_manifest_v1.json",
        "export_manifest": "fig01_export_manifest_v1.json",
        "script": "fig01_evidence_architecture_v1.py",
        "legend": "38_fig01_legend_v1.md",
        "rows": 18,
        "panels": ["a", "b", "c"],
        "width_mm": 183,
        "height_mm": 150,
    },
    2: {
        "dir": "fig02",
        "stem": "fig02_equal_service_tradeoff_v1",
        "source": "fig02_source_data_v1.csv",
        "source_manifest": "fig02_source_manifest_v1.json",
        "export_manifest": "fig02_export_manifest_v1.json",
        "script": "fig02_equal_service_tradeoff_v1.py",
        "legend": "41_fig02_legend_v1.md",
        "rows": 10,
        "panels": ["a", "b", "c", "d"],
        "width_mm": 183,
        "height_mm": 135,
    },
    3: {
        "dir": "fig03",
        "stem": "fig03_wue_heterogeneity_v1",
        "source": "fig03_source_data_v1.csv",
        "source_manifest": "fig03_source_manifest_v1.json",
        "export_manifest": "fig03_export_manifest_v1.json",
        "script": "fig03_wue_heterogeneity_v1.py",
        "legend": "44_fig03_legend_v1.md",
        "rows": 162,
        "panels": ["a", "b", "c", "d"],
        "width_mm": 183,
        "height_mm": 135,
    },
    4: {
        "dir": "fig04",
        "stem": "fig04_matched_service_robustness_v1",
        "source": "fig04_source_data_v1.csv",
        "source_manifest": "fig04_source_manifest_v1.json",
        "export_manifest": "fig04_export_manifest_v1.json",
        "script": "fig04_matched_service_robustness_v1.py",
        "legend": "47_fig04_legend_v1.md",
        "rows": 91,
        "panels": ["a", "b", "c", "d", "e", "f"],
        "width_mm": 183,
        "height_mm": 170,
    },
}

PANEL_INTEGRATION = {
    (1, "a"): {"role": "system boundary", "source_selector": "panel == 'a'", "callout_anchor": "M01", "callout_text": "Fig. 1a"},
    (1, "b"): {"role": "physical accounting", "source_selector": "panel == 'b'", "callout_anchor": "M08", "callout_text": "Fig. 1b"},
    (1, "c"): {"role": "policy and optimization logic", "source_selector": "panel == 'c'", "callout_anchor": "M10", "callout_text": "Fig. 1c"},
    (2, "a"): {"role": "primary trade-off", "source_selector": "panel == 'a'", "callout_anchor": "R05", "callout_text": "Fig. 2a"},
    (2, "b"): {"role": "primary carbon interval", "source_selector": "panel == 'b'", "callout_anchor": "R02", "callout_text": "Fig. 2b"},
    (2, "c"): {"role": "primary exposure interval", "source_selector": "panel == 'c'", "callout_anchor": "R04", "callout_text": "Fig. 2c"},
    (2, "d"): {"role": "equal-service control", "source_selector": "panel == 'd'", "callout_anchor": "R01", "callout_text": "Fig. 2d"},
    (3, "a"): {"role": "total-water heterogeneity", "source_selector": "use_panel_a == True", "callout_anchor": "R06", "callout_text": "Fig. 3a"},
    (3, "b"): {"role": "exposure-direction robustness", "source_selector": "use_panel_b == True", "callout_anchor": "R07", "callout_text": "Fig. 3b"},
    (3, "c"): {"role": "WUE mechanism", "source_selector": "use_panel_c == True", "callout_anchor": "R06", "callout_text": "Fig. 3c"},
    (3, "d"): {"role": "screening degeneration", "source_selector": "use_panel_d == True", "callout_anchor": "R07", "callout_text": "Fig. 3d"},
    (4, "a"): {"role": "matched-service robustness", "source_selector": "record_type == 'sensitivity'", "callout_anchor": "R08", "callout_text": "Fig. 4a"},
    (4, "b"): {"role": "service-confounding diagnostic", "source_selector": "record_type == 'sensitivity'", "callout_anchor": "R08", "callout_text": "Fig. 4b"},
    (4, "c"): {"role": "external mapping intervals", "source_selector": "record_type == 'external_interval'", "callout_anchor": "R10", "callout_text": "Fig. 4c"},
    (4, "d"): {"role": "daily direction audit", "source_selector": "record_type == 'daily_direction'", "callout_anchor": "R12", "callout_text": "Fig. 4d"},
    (4, "e"): {"role": "external calibration boundary", "source_selector": "record_type == 'external_service_origin'", "callout_anchor": "R10", "callout_text": "Fig. 4e"},
    (4, "f"): {"role": "carbon-reversal falsification", "source_selector": "record_type == 'carbon_reversal'", "callout_anchor": "R09", "callout_text": "Fig. 4f"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def legend_details(path: Path, figure_number: int) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## Final legend\s+(.*?)\s+## Interpretation boundary", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Final legend block not found in {path.name}")
    block = match.group(1).strip()
    title_match = re.search(rf"\*\*Fig\. {figure_number} \| (.*?)\.\*\*", block)
    return {
        "word_count": len(re.findall(r"\S+", block)),
        "title": title_match.group(1) if title_match else "",
        "source_data_statement": "Source data are provided" in block,
        "interpretation_boundary_present": "## Interpretation boundary" in text,
    }


def pdf_details(path: Path) -> dict[str, object]:
    page = PdfReader(str(path)).pages[0]
    width_mm = float(page.mediabox.width) / 72.0 * MM_PER_INCH
    height_mm = float(page.mediabox.height) / 72.0 * MM_PER_INCH
    fonts: list[dict[str, str]] = []
    type3_count = 0
    resources = page.get("/Resources")
    if resources:
        resources = resources.get_object()
        font_resources = resources.get("/Font", {})
        if hasattr(font_resources, "get_object"):
            font_resources = font_resources.get_object()
        for resource_name, font_ref in font_resources.items():
            font = font_ref.get_object()
            subtype = str(font.get("/Subtype", ""))
            basefont = str(font.get("/BaseFont", ""))
            fonts.append({"resource": str(resource_name), "subtype": subtype, "basefont": basefont})
            if subtype == "/Type3":
                type3_count += 1
    return {"width_mm": width_mm, "height_mm": height_mm, "fonts": fonts, "type3_font_count": type3_count}


def raster_details(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        dpi = [float(value) for value in image.info.get("dpi", (0, 0))]
        compression = image.info.get("compression")
        details = {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "format": image.format,
            "dpi": dpi,
            "compression": compression,
        }
        if image.format == "PNG":
            rgb = image.convert("RGB")
            bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
            if bbox is None:
                details["nonwhite_bbox"] = None
                details["canvas_clearance_ltrb_px"] = None
            else:
                details["nonwhite_bbox"] = list(bbox)
                details["canvas_clearance_ltrb_px"] = [bbox[0], bbox[1], rgb.width - bbox[2], rgb.height - bbox[3]]
    return details


def svg_details(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    text_nodes = root.findall(".//{http://www.w3.org/2000/svg}text")
    return {
        "width": root.attrib.get("width"),
        "height": root.attrib.get("height"),
        "viewBox": root.attrib.get("viewBox"),
        "text_node_count": len(text_nodes),
        "private_absolute_path_hits": len(re.findall(r"[A-Za-z]:\\Users\\", text)),
    }


def assigned_dict(source_text: str, name: str) -> dict[str, str]:
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError(f"{name} is not a dictionary")
            return value
    raise ValueError(f"{name} not found")


def panel_rows(frame: pd.DataFrame, figure_number: int, panel: str) -> int:
    if figure_number in (1, 2):
        return int(frame["panel"].eq(panel).sum())
    if figure_number == 3:
        return int(frame[f"use_panel_{panel}"].astype(str).str.lower().eq("true").sum())
    if panel in ("a", "b"):
        return int(frame["record_type"].eq("sensitivity").sum())
    record_type = {
        "c": "external_interval",
        "d": "daily_direction",
        "e": "external_service_origin",
        "f": "carbon_reversal",
    }[panel]
    return int(frame["record_type"].eq(record_type).sum())


def main() -> None:
    contract = pd.read_csv(PANEL_CONTRACT)
    if len(contract) != 17:
        raise ValueError(f"Expected 17 frozen panel contracts; found {len(contract)}")
    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    claim_text = CLAIM_TABLE.read_text(encoding="utf-8")
    valid_evidence_ids = set(re.findall(r"^\|\s*(T[123][A-Z]\d{2})\s*\|", claim_text, flags=re.MULTILINE))

    panel_records: list[dict[str, object]] = []
    figure_records: list[dict[str, object]] = []
    source_assertions_pass = True
    source_hashes_match = True
    output_hashes_match = True
    scripts_compile = True
    svg_editable = True
    no_private_paths = True
    pdf_size_match = True
    no_type3_fonts = True
    tiff_600_lzw = True
    png_300_nonblank = True
    legend_contract_pass = True
    policy_color_checks: list[bool] = []

    for number, spec in EXPECTED.items():
        figure_label = f"Fig{number}"
        figure_dir = HERE / spec["dir"]
        source_path = figure_dir / spec["source"]
        source_manifest_path = figure_dir / spec["source_manifest"]
        export_manifest_path = figure_dir / spec["export_manifest"]
        script_path = figure_dir / spec["script"]
        legend_path = REWRITE_DIR / spec["legend"]
        source = pd.read_csv(source_path)
        source_manifest = read_json(source_manifest_path)
        export_manifest = read_json(export_manifest_path)
        script_text = script_path.read_text(encoding="utf-8")
        try:
            compile(script_text, str(script_path), "exec")
            compiled = True
        except SyntaxError:
            compiled = False
        scripts_compile = scripts_compile and compiled

        source_assertions = source_manifest.get("assertions", {})
        local_source_assertions_pass = bool(source_assertions) and all(bool(value) for value in source_assertions.values())
        source_assertions_pass = source_assertions_pass and local_source_assertions_pass
        actual_source_hash = sha256(source_path)
        source_manifest_hash = source_manifest["derived_source"]["sha256"]
        export_source_hash = export_manifest["source_data"]["sha256"]
        local_source_hash_match = actual_source_hash == source_manifest_hash == export_source_hash
        source_hashes_match = source_hashes_match and local_source_hash_match

        legend = legend_details(legend_path, number)
        export_title = export_manifest["title"].rstrip(".")
        local_legend_pass = bool(
            legend["word_count"] <= 300
            and legend["title"].rstrip(".") == export_title
            and legend["source_data_statement"]
            and legend["interpretation_boundary_present"]
        )
        legend_contract_pass = legend_contract_pass and local_legend_pass

        svg_path = figure_dir / f"{spec['stem']}.svg"
        pdf_path = figure_dir / f"{spec['stem']}.pdf"
        tiff_path = figure_dir / f"{spec['stem']}.tiff"
        png_path = figure_dir / f"{spec['stem']}.png"
        paths_by_format = {"svg": svg_path, "pdf": pdf_path, "tiff": tiff_path, "png": png_path}
        local_output_hashes_match = True
        for output_format, path in paths_by_format.items():
            local_output_hashes_match = local_output_hashes_match and sha256(path) == export_manifest["outputs"][output_format]["sha256"]
        output_hashes_match = output_hashes_match and local_output_hashes_match

        svg = svg_details(svg_path)
        pdf = pdf_details(pdf_path)
        tiff = raster_details(tiff_path)
        png = raster_details(png_path)
        local_svg_editable = svg["text_node_count"] > 0
        local_no_private_paths = svg["private_absolute_path_hits"] == 0
        local_pdf_size = abs(pdf["width_mm"] - spec["width_mm"]) < 0.02 and abs(pdf["height_mm"] - spec["height_mm"]) < 0.02
        local_no_type3 = pdf["type3_font_count"] == 0
        local_tiff = all(abs(value - 600) < 0.1 for value in tiff["dpi"]) and tiff["compression"] == "tiff_lzw"
        local_png = all(abs(value - 300) < 0.1 for value in png["dpi"]) and png["nonwhite_bbox"] is not None
        svg_editable = svg_editable and local_svg_editable
        no_private_paths = no_private_paths and local_no_private_paths
        pdf_size_match = pdf_size_match and local_pdf_size
        no_type3_fonts = no_type3_fonts and local_no_type3
        tiff_600_lzw = tiff_600_lzw and local_tiff
        png_300_nonblank = png_300_nonblank and local_png

        colors = assigned_dict(script_text, "COLORS")
        if number in (1, 2):
            local_policy_color = colors.get("e0") == "#6E7378" and colors.get("e1") == "#225E91" and colors.get("e2") == "#00857C" and colors.get("stress") == "#B44A3E"
        elif number == 3:
            local_policy_color = colors.get("e0") == "#6E7378" and colors.get("e1") == "#225E91" and colors.get("stress") == "#B44A3E"
        else:
            local_policy_color = colors.get("blue") == "#225E91" and colors.get("e2") == "#00857C" and colors.get("red") == "#B44A3E"
        policy_color_checks.append(local_policy_color)

        manuscript_callout_count = len(re.findall(rf"(?:Fig\.|Figure)\s*{number}(?!\d)", manuscript_text, flags=re.IGNORECASE))
        figure_records.append(
            {
                "figure": f"Fig. {number}",
                "title": export_title,
                "source_rows": int(len(source)),
                "expected_source_rows": spec["rows"],
                "panel_count": len(spec["panels"]),
                "legend_words": legend["word_count"],
                "width_mm": pdf["width_mm"],
                "height_mm": pdf["height_mm"],
                "svg_text_nodes": svg["text_node_count"],
                "pdf_type3_fonts": pdf["type3_font_count"],
                "tiff_dpi": tiff["dpi"],
                "tiff_compression": tiff["compression"],
                "png_dpi": png["dpi"],
                "png_canvas_clearance_ltrb_px": png["canvas_clearance_ltrb_px"],
                "source_assertions_pass": local_source_assertions_pass,
                "source_hash_match": local_source_hash_match,
                "output_hashes_match": local_output_hashes_match,
                "script_compiles": compiled,
                "policy_color_contract_pass": local_policy_color,
                "manuscript_callout_count": manuscript_callout_count,
                "files": {
                    "legend": spec["legend"],
                    "source_data": f"figures/{spec['dir']}/{spec['source']}",
                    "source_manifest": f"figures/{spec['dir']}/{spec['source_manifest']}",
                    "script": f"figures/{spec['dir']}/{spec['script']}",
                    "export_manifest": f"figures/{spec['dir']}/{spec['export_manifest']}",
                    "svg": f"figures/{spec['dir']}/{svg_path.name}",
                    "pdf": f"figures/{spec['dir']}/{pdf_path.name}",
                    "tiff": f"figures/{spec['dir']}/{tiff_path.name}",
                    "png": f"figures/{spec['dir']}/{png_path.name}",
                },
            }
        )

        figure_contract = contract.loc[contract["figure"].eq(figure_label)].copy()
        actual_panels = sorted(figure_contract["panel"].tolist())
        if actual_panels != spec["panels"]:
            raise ValueError(f"Panel contract mismatch for {figure_label}: {actual_panels}")
        for _, row in figure_contract.sort_values("panel").iterrows():
            panel = row["panel"]
            integration = PANEL_INTEGRATION[(number, panel)]
            evidence_ids = row["source_evidence_ids"].split(";")
            panel_records.append(
                {
                    "figure": f"Fig. {number}",
                    "panel": panel,
                    "role": integration["role"],
                    "question": row["question"],
                    "evidence_ids": row["source_evidence_ids"],
                    "evidence_ids_valid": all(evidence_id in valid_evidence_ids for evidence_id in evidence_ids),
                    "source_data_path": f"figures/{spec['dir']}/{spec['source']}",
                    "source_selector": integration["source_selector"],
                    "source_rows_used": panel_rows(source, number, panel),
                    "statistical_definition": row["statistics_or_n"],
                    "reviewer_risk": row["interpretation_limit"],
                    "legend_path": spec["legend"],
                    "recommended_callout_anchor": integration["callout_anchor"],
                    "recommended_callout_text": integration["callout_text"],
                    "anchor_exists_in_manuscript": f"[{integration['callout_anchor']}]" in manuscript_text,
                    "current_manuscript_callout_present": bool(re.search(rf"(?:Fig\.|Figure)\s*{number}{panel}\b", manuscript_text, flags=re.IGNORECASE)),
                    "status": "FROZEN",
                }
            )

    panel_frame = pd.DataFrame(panel_records)
    panel_frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", lineterminator="\n")
    all_callout_anchors_exist = bool(panel_frame["anchor_exists_in_manuscript"].all())
    all_manuscript_callouts_present = bool(panel_frame["current_manuscript_callout_present"].all())
    assertions = {
        "four_figure_packages_present": len(figure_records) == 4,
        "seventeen_frozen_panel_contracts": len(panel_frame) == 17 and panel_frame["status"].eq("FROZEN").all(),
        "panel_letters_are_complete_and_sequential": all(
            panel_frame.loc[panel_frame["figure"].eq(f"Fig. {number}"), "panel"].tolist() == spec["panels"]
            for number, spec in EXPECTED.items()
        ),
        "all_evidence_ids_exist_in_frozen_claim_table": bool(panel_frame["evidence_ids_valid"].all()),
        "all_source_manifests_present_and_assertions_pass": source_assertions_pass,
        "all_source_hashes_match_manifests": source_hashes_match,
        "all_output_hashes_match_manifests": output_hashes_match,
        "all_python_generators_compile": scripts_compile,
        "all_figures_are_183_mm_wide_with_contract_heights": pdf_size_match,
        "all_svgs_have_editable_text": svg_editable,
        "all_pdfs_have_no_type3_fonts": no_type3_fonts,
        "all_tiffs_are_600_dpi_lzw": tiff_600_lzw,
        "all_pngs_are_300_dpi_and_nonblank": png_300_nonblank,
        "all_legends_match_titles_are_self_contained_and_below_300_words": legend_contract_pass,
        "policy_and_stress_colors_match_cross_figure_contract": all(policy_color_checks),
        "no_private_absolute_paths_in_svgs": no_private_paths,
        "all_recommended_callout_anchors_exist": all_callout_anchors_exist,
    }
    assertions = {key: bool(value) for key, value in assertions.items()}
    if not all(assertions.values()):
        failed = [key for key, passed in assertions.items() if not passed]
        raise ValueError(f"Figure-set audit failed: {failed}")
    status = "PASS" if all_manuscript_callouts_present else "AUDIT_PASS_WITH_CALLOUT_INSERTION_PENDING"
    report = {
        "status": status,
        "date": "2026-07-21",
        "backend": "Python only",
        "core_conclusion": "The four-figure sequence establishes the analysis architecture, equal-service carbon-water operating points, WUE-dependent water heterogeneity, and robustness with explicit calibration boundaries.",
        "evidence_sequence": [
            "Fig. 1: system and accounting architecture",
            "Fig. 2: primary equal-service carbon-exposure result",
            "Fig. 3: spatial WUE mechanism and screening sensitivity",
            "Fig. 4: matched-service robustness, external direction, calibration boundary, and falsification",
        ],
        "figure_count": len(figure_records),
        "panel_count": int(len(panel_frame)),
        "figures": figure_records,
        "panel_manifest": {"path": OUTPUT_CSV.name, "rows": int(len(panel_frame)), "sha256": sha256(OUTPUT_CSV)},
        "assertions": assertions,
        "manuscript_integration": {
            "manuscript": MANUSCRIPT.name,
            "current_panel_callouts_present": int(panel_frame["current_manuscript_callout_present"].sum()),
            "required_panel_callouts": int(len(panel_frame)),
            "all_callout_anchors_exist": all_callout_anchors_exist,
            "next_action": "Insert the recommended callouts during the manuscript-figure integration stage; do not modify the frozen evidence meaning.",
        },
        "interpretation_boundary": "DIRECTIONAL_MECHANISM_ONLY",
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"status": status, "figures": len(figure_records), "panels": len(panel_frame), "assertions_passed": sum(assertions.values()), "assertions_total": len(assertions), "current_panel_callouts": int(panel_frame["current_manuscript_callout_present"].sum())}, indent=2))


if __name__ == "__main__":
    main()
