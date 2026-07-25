from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STEMS = (
    "fig01_environmental_decision_model_v1",
    "fig02_reference_screened_water_frontier_v1",
    "fig03_scenario_envelope_v1",
    "fig04_wue_and_energy_boundaries_v1",
)


def image_is_nonblank(image: Image.Image) -> bool:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return bool(np.any(pixels < 245) and np.any(pixels > 10))


def create_contact_sheet(images: list[tuple[str, Image.Image]], path: Path, *, grayscale: bool) -> None:
    thumb_width = 1050
    thumb_height = 780
    label_height = 42
    canvas = Image.new("RGB", (thumb_width * 2, (thumb_height + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(images):
        rendered = image.convert("L").convert("RGB") if grayscale else image.convert("RGB")
        rendered.thumbnail((thumb_width - 36, thumb_height - 36), Image.Resampling.LANCZOS)
        x = (index % 2) * thumb_width
        y = (index // 2) * (thumb_height + label_height)
        offset_x = x + (thumb_width - rendered.width) // 2
        offset_y = y + 24 + (thumb_height - rendered.height) // 2
        canvas.paste(rendered, (offset_x, offset_y))
        draw.text((x + 14, y + 10), name, fill="black")
    canvas.save(path, dpi=(150, 150))


def pdf_font_subtypes(path: Path) -> set[str]:
    subtypes: set[str] = set()
    reader = PdfReader(path)
    for page in reader.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue
        for reference in fonts.values():
            font = reference.get_object()
            subtype = font.get("/Subtype")
            if subtype is not None:
                subtypes.add(str(subtype))
    return subtypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--keep-contact-sheets", action="store_true")
    args = parser.parse_args()
    root = args.root
    export_dir = root / "04_figures" / "exports"
    qa_dir = root / "06_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    previews: list[tuple[str, Image.Image]] = []

    for stem in EXPECTED_STEMS:
        svg = export_dir / f"{stem}.svg"
        pdf = export_dir / f"{stem}.pdf"
        tiff = export_dir / f"{stem}.tiff"
        png = export_dir / f"{stem}.png"
        required = all(path.exists() and path.stat().st_size > 0 for path in (svg, pdf, tiff, png))
        checks.append({"check": f"{stem}_all_exports_present", "passed": required, "detail": "svg,pdf,tiff,png"})

        text_nodes = len(re.findall(r"<text(?:\s|>)", svg.read_text(encoding="utf-8"))) if svg.exists() else 0
        checks.append({"check": f"{stem}_svg_has_editable_text", "passed": text_nodes > 0, "detail": f"text_nodes={text_nodes}"})

        font_subtypes = pdf_font_subtypes(pdf)
        checks.append({"check": f"{stem}_pdf_has_no_type3_font", "passed": "/Type3" not in font_subtypes, "detail": f"font_subtypes={sorted(font_subtypes)}"})

        with Image.open(tiff) as image:
            tiff_dpi = image.info.get("dpi", (0, 0))
            compression = image.tag_v2.get(259)
            tiff_size = image.size
            tiff_nonblank = image_is_nonblank(image)
        checks.append({"check": f"{stem}_tiff_lzw", "passed": compression == 5, "detail": f"compression_tag={compression}"})
        checks.append({"check": f"{stem}_tiff_600_dpi", "passed": all(abs(value - 600.0) <= 1.0 for value in tiff_dpi), "detail": f"dpi={tiff_dpi}"})
        checks.append({"check": f"{stem}_tiff_nonblank", "passed": tiff_nonblank, "detail": f"pixels={tiff_size}"})

        with Image.open(png) as image:
            png_dpi = image.info.get("dpi", (0, 0))
            png_size = image.size
            png_nonblank = image_is_nonblank(image)
            preview = image.copy()
        checks.append({"check": f"{stem}_png_300_dpi", "passed": all(abs(value - 300.0) <= 1.0 for value in png_dpi), "detail": f"dpi={png_dpi}"})
        checks.append({"check": f"{stem}_png_nonblank", "passed": png_nonblank, "detail": f"pixels={png_size}"})
        previews.append((stem, preview))

    contact = qa_dir / "temporary_ema_figure_contact_sheet.png"
    grayscale = qa_dir / "temporary_ema_figure_contact_sheet_grayscale.png"
    create_contact_sheet(previews, contact, grayscale=False)
    create_contact_sheet(previews, grayscale, grayscale=True)
    checks.append({"check": "temporary_contact_sheets_created", "passed": contact.exists() and grayscale.exists(), "detail": "color and grayscale"})

    payload = {
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "checks": checks,
        "temporary_contact_sheets": [contact.name, grayscale.name],
        "temporary_contact_sheets_retained": bool(args.keep_contact_sheets),
    }
    if payload["status"] != "PASS":
        (qa_dir / "ema_figure_export_qa_v1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("Figure export QA failed")
    if not args.keep_contact_sheets:
        contact.unlink(missing_ok=True)
        grayscale.unlink(missing_ok=True)
    (qa_dir / "ema_figure_export_qa_v1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
