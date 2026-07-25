# EMA figure-set visual QA v1

## Scope and backend

- Figure set: Fig. 1-Fig. 4.
- Backend: Python only for drawing, previewing, exporting, contact-sheet construction, and grayscale conversion.
- Claim ceiling: `DIRECTIONAL_MECHANISM_ONLY`.
- Archetypes: Fig. 1, schematic-led composite; Fig. 2-Fig. 4, quantitative grids.
- Review date: 2026-07-25.

## Scientific figure contract

- Fig. 1 conclusion: the study is a service-preserving environmental allocation model that keeps physical direct-water accounting separate from the water-stress screen.
- Fig. 2 conclusion: progressively tighter screened-water limits create a nonlinear but scenario-specific carbon penalty in the reference case.
- Fig. 3 conclusion: the sign is stable across the complete deterministic scenario grid, but the magnitude is heterogeneous and can approach zero.
- Fig. 4 conclusion: spatial WUE affects intermediate-cap penalties, whereas a region-invariant energy multiplier rescales absolute outputs without changing allocation.

## Programmatic export QA

Status: **PASS**.

- Four figures each have SVG, PDF, TIFF, and PNG exports.
- SVG text remains editable.
- PDF exports contain no Type 3 fonts.
- TIFF exports are LZW-compressed at 600 dpi.
- PNG previews are 300 dpi.
- All raster exports are nonblank.
- Source-data and export manifests were regenerated after the final figure-code revision.

Machine-readable result: `ema_figure_export_qa_v1.json`.

## Human color review

Status: **PASS after correction and reinspection**.

Corrections made during the human review loop:

1. Fig. 1c flow-box labels were rewrapped and reduced to fit fully inside their boxes.
2. Fig. 2b received a two-line x-axis title to prevent right-edge clipping.
3. Fig. 2a endpoint labels were moved inside the plotting area.
4. Fig. 2d uses different hatch patterns for the two screened regions.
5. Fig. 3a and Fig. 4a/c explanatory notes received opaque white backgrounds so data marks do not pass through text.
6. Fig. 4b legend was moved below the panel, outside the data region.
7. Fig. 4d legend was moved into an unused central-right region with an opaque background.

Final reinspection found no clipped text, overlapping annotations, obscured data marks, inconsistent panel labels, or legends covering data.

## Human grayscale review

Status: **PASS after redundant-encoding revision**.

- Fig. 2d distinguishes screened regions by different hatches.
- Fig. 4b distinguishes classes by circle versus diamond markers in addition to color.
- Fig. 4c distinguishes regions with separate marker and line-style combinations.
- Fig. 4d distinguishes energy levels with separate marker and line-style combinations.
- All primary trends, medians, ranges, zero reference lines, and audit counts remain interpretable in grayscale.

## Interpretation and integrity checks

- No panel presents a confidence interval where the data are a complete deterministic enumeration.
- The resampling interval is described as conditional trace-segment stability, not population uncertainty.
- Carbon and direct-water units remain distinct.
- Water stress is shown as a screening context, not a multiplier or hydrologic-damage estimate.
- No panel makes a facility-specific, causal, deployed-system, watershed, or population claim.

## Temporary file policy

The color and grayscale contact sheets were created only for inspection. They are deleted by the final QA run and are not part of the submission or archive package.
