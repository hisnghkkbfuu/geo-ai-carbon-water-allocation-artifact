# Figure contract for the EMA retargeted manuscript

Status: PRE-PLOTTING CONTRACT  
Backend: Python only  
Claim level: `DIRECTIONAL_MECHANISM_ONLY`  
Target: Environmental Modeling & Assessment

## Shared visual system

- Final working width: 180 mm, suitable for a double-column placement; final
  journal sizing will be checked against the accepted target's guide.
- Fonts: Arial/Helvetica-compatible sans serif; 6.5-7 pt body text at final
  size; bold lowercase panel labels.
- Status quo: neutral graphite `#5F6B73`.
- Carbon-first allocation: frozen teal `#00857C`.
- Tighter water-screening caps: blue `#2F6B9A` with marker shape and direct cap
  labels; color is never the only encoding.
- High-stress regions: muted rust `#B35C44`; low- or medium-stress regions:
  muted green `#5A8F6A`.
- No rainbow scales, no dual y axes, no unlabelled probability-like error bars.
- All quantitative values have a CSV source manifest. Scenario ranges are
  deterministic enumerations, not confidence intervals.

## Fig. 1 | Environmental decision model and accounting boundary

Core conclusion: The allocation model separates service preservation, carbon
accounting, physical direct-water accounting, and categorical water-stress
screening rather than collapsing them into a single score.

Archetype: schematic-led composite.

Panel map:

- a: trace-derived hourly demand, regional scenario layers, and feasible
  allocation structure.
- b: physically distinct IT-energy, facility-energy/carbon, and direct-water
  accounting branches, showing that PUE is excluded from direct water.
- c: two-stage epsilon-constraint sequence: preserve maximum service, then
  minimize carbon under a screened-water cap.

Evidence hierarchy: model definition and data-material passport. No results are
encoded as observations in this figure. Interpretation boundaries are included
as text within panels a and c.

Reviewer risk: a schematic could look like an operational deployment design.
Mitigation: label all inputs as trace/proxy/scenario, retain limitations in the
legend, and avoid control-room or facility imagery.

Exports: SVG with editable text, Type-3-free PDF, 600-dpi LZW TIFF, 300-dpi PNG.

## Fig. 2 | Reference screened-water frontier at equal service

Core conclusion: Under the reference scenario, progressively tighter screened-
water caps produce a monotone carbon penalty while retaining maximum feasible
service and a carbon value below the strict no-migration status quo.

Archetype: asymmetric mixed-modality quantitative figure.

Panel map:

- a (hero): additional model-accounted carbon versus screened direct water
  avoided across 21 cap fractions; direct labels identify 0%, 50%, and 100%
  caps.
- b: carbon change versus status quo across the same cap grid; the zero line is
  shown as a reference, not a hypothesis-test threshold.
- c: adjacent-segment marginal additional carbon per screened liter avoided;
  missing values are not imputed.
- d: stacked regional allocations at 0%, 50%, and 100% allowed fractions, with
  water-stress category encoded redundantly by fill and hatch/label.

Evidence hierarchy: 21 reoptimized reference points; equal-service and
constraint-residual source data accompany panels a-d.

Statistics: no inferential test. Values are deterministic scenario outputs.
Source data: Fig. 2 source CSV entries in the figure source manifest.

Reviewer risk: calling a finite cap grid a universal Pareto frontier.
Mitigation: use "reference screened-water frontier" and state all feasibility,
input, and scenario boundaries in legend and text.

## Fig. 3 | Deterministic scenario envelope

Core conclusion: Across the complete 243-cell carbon/PUE/capacity/migration/
latency factorial, all tested caps remain at or below the matched-service
no-migration carbon value, but the magnitude of the modeled change is highly
heterogeneous.

Archetype: quantitative grid.

Panel map:

- a (hero): every scenario point by cap fraction, with deterministic min-to-max
  ranges and median markers for carbon change versus status quo.
- b: every zero-cap scenario point grouped by PUE scenario; show the near-zero
  cases rather than hiding them.
- c: adjacent-segment marginal carbon penalties across the 972 valid scenario
  segments, displayed as a distribution with all points or range summaries.
- d: QA strip showing 243 cells, 1,215 optimal solves, matched service, and no
  observed cap/latency/capacity/migration violation. This is a compact
  diagnostic, not a performance claim.

Evidence hierarchy: complete deterministic factorial and solver/audit outputs.

Statistics: no p values or confidence intervals. Min/max and quartiles describe
the enumerated scenarios only.

Reviewer risk: deterministic ranges being read as uncertainty intervals.
Mitigation: call them "enumerated scenario ranges" in panel text and legend.

## Fig. 4 | Spatial WUE and scale-boundary checks

Core conclusion: Spatial WUE heterogeneity changes intermediate cap costs,
whereas a region-invariant energy-intensity multiplier rescales absolute
footprints without changing allocation or relative carbon conclusions.

Archetype: quantitative grid.

Panel map:

- a: 81-profile deterministic range of additional carbon by cap fraction after
  full reoptimization.
- b: profile-level intermediate-cap outcomes to show spatial WUE sensitivity
  without treating profiles as sampled observations.
- c: regional allocation comparison across low, central, and high energy
  calibrations at each cap fraction; allocations overlay exactly within numeric
  tolerance.
- d: absolute carbon and screened-water scaling at low, central, and high
  energy intensity, explicitly labelled as a scale check rather than a new
  empirical result.

Evidence hierarchy: 405 WUE reoptimized points and 15 energy scaling checks.

Statistics: complete deterministic enumeration; no inferential test.

Reviewer risk: energy calibration is misrepresented as a site measurement.
Mitigation: retain "measured proxy plus scenario server overhead" in the data
passport and show only the tested scale invariance conclusion.

## Cross-figure QA requirements

- Python is the only plotting, preview, export, and visual-QA backend.
- Every panel has source data, a source evidence ID, and a conclusion-level
  assertion.
- Greyscale renders must preserve policy/cap identity through marker, line, or
  hatch encoding.
- SVG text must remain editable; PDF must contain no Type 3 fonts; TIFF uses
  600 dpi LZW; PNG preview uses 300 dpi.
- A contact sheet and greyscale contact sheet are inspected before figures are
  frozen, then temporary QA images are removed.
