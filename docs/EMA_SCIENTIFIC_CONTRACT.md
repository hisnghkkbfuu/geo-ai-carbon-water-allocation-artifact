# Scientific contract for Environmental Modeling & Assessment

Status: FROZEN BEFORE NEW ANALYSIS  
Version: v1  
Date: 2026-07-25  
Claim level: `DIRECTIONAL_MECHANISM_ONLY`

## 1. Working title

**Environmental screening of carbon-water trade-offs in geo-distributed
computing workload allocation**

The title may be shortened after results are frozen, but it must retain both the
environmental-screening scope and the allocation-model context. It must not
claim an operational scheduler, universal sustainability improvement, or
measured facility footprints.

## 2. Decision problem

For an hourly workload with regional capacity, latency, and migration
constraints, determine how progressively tighter limits on modeled direct water
allocated to High or Extremely high baseline-water-stress regions change
model-accounted carbon while maximum feasible delivered service is preserved.

## 3. Primary question

Under the reference environmental and infrastructure scenario, what carbon
penalty is associated with avoiding successive fractions of the high-stress
direct-water exposure produced by unconstrained carbon-minimizing allocation?

## 4. Secondary questions

1. Does the direction and magnitude of the cap-carbon trade-off remain stable
   across the frozen carbon-mapping, capacity, migration, and latency scenarios?
2. Which constraints bind along the environmental frontier, and when does the
   cap cease to change allocation?
3. How do alternative spatial WUE profiles change physical-water accounting
   without changing the meaning of the categorical water-stress screen?
4. Does the directional conclusion persist under the independent demand-timing
   trace, subject to its existing execution-time fallback limitation?

## 5. Model contract

### 5.1 Variables and feasibility

- `x[t,r]`: accelerator-hour-equivalent service assigned in hour `t` to region
  `r`.
- `u[t]`: unserved accelerator-hour-equivalent service.
- Hourly demand conservation, regional capacity, migration-share, and latency-
  eligibility constraints remain unchanged from the frozen model.
- No cross-hour deferral, queuing, replication, demand response, or migration
  energy is modeled.

### 5.2 Environmental accounting

- IT electricity: assigned service multiplied by the frozen IT-energy proxy.
- Facility electricity: IT electricity multiplied by regional PUE.
- Carbon: facility electricity multiplied by the regional scenario carbon
  factor.
- Direct water: IT electricity multiplied by regional WUE. PUE is excluded.
- High-stress direct-water exposure: physical direct water counted only where
  the binary Aqueduct screen is High or Extremely high.
- Water-stress scores are never used as multiplicative damage weights.

### 5.3 Optimization hierarchy

1. Stage 1 minimizes total unserved service.
2. Stage 2 fixes total unserved service to the Stage-1 optimum within `1e-8` and
   minimizes carbon under an explicit high-stress direct-water cap.

This is an epsilon-constraint environmental decision formulation. It is not
presented as a novel optimization algorithm. Its methodological value is the
separation of unlike physical units and the preservation of service before
environmental optimization.

## 6. Prespecified analyses

### A1. Reference frontier

Use the reference case and caps equal to 0%, 5%, 10%, ..., 100% of the
unconstrained carbon-first high-stress direct-water exposure. For every point,
save allocation, carbon, total direct water, screened direct water, migration,
service, solver status, objective value, cap slack, and feasibility residuals.

The cap fraction is an analytical control scale. It is not a probability or a
policy recommendation.

### A2. Scenario frontier envelope

Repeat a coarser cap grid of 0%, 25%, 50%, 75%, and 100% for the complete frozen
factorial of:

- three carbon mappings;
- three PUE scenarios;
- three capacity scenarios;
- three migration limits: 15%, 30%, 50%;
- three latency thresholds: 15, 20, 35 ms.

This yields 243 scenario cells and 1,215 cap-constrained solves, plus the
required cell-specific baselines. Run and validate one PUE block at a time. A
cell may be summarized only after service matching, feasibility, and solver
optimality pass.

### A3. Marginal trade-off accounting

For adjacent feasible points, report:

- screened direct water avoided relative to unconstrained carbon-first
  allocation;
- additional carbon relative to unconstrained carbon-first allocation;
- marginal additional carbon per liter of screened direct water avoided.

The ratio is descriptive and scenario-specific. Undefined zero-denominator
segments remain missing and are not imputed.

### A4. Binding-constraint diagnostics

Record cap slack, migration slack, eligible destination set, regional capacity
utilization, and changes in regional allocation. Identify numerical plateaus and
piecewise-linear transitions without assigning causal interpretation.

### A5. Spatial WUE sensitivity

Retain the complete 81-profile WUE factorial for physical accounting. For the
frontier, reoptimize the five-point cap grid under all 81 profiles at the
reference carbon, PUE, capacity, migration, latency, and risk settings. This
yields 405 cap-constrained solves. Revaluation of a fixed dispatch may answer
accounting questions but must not be mislabeled as reoptimization.

### A6. Energy-scaling invariance

Rerun the reference five-point frontier at the frozen low, central, and high IT-
energy intensities. Because energy intensity is a region-invariant scalar in
the current model, test rather than assume that allocation and percentage
changes are invariant while absolute carbon and water scale. Treat failure of
this check as an implementation problem.

### A7. Conditional trace uncertainty

Retain paired circular 24-hour block resampling only for statistics that can be
computed from the solved supplied trace. Label all intervals as conditional
trace-segment intervals, not population confidence intervals and not parameter
uncertainty. Do not combine them with deterministic scenario ranges into a
single probabilistic interval.

### A8. External trace

Use the Azure trace only for directional replication. The existing 4096-token
boundary fallback contributes 97.171% of mapped service, so absolute external
accelerator-hours, carbon, and water remain uninterpretable.

## 7. Primary estimands and reporting hierarchy

Primary estimands:

1. Carbon change relative to the no-migration status quo at matched maximum
   service.
2. Additional carbon relative to unconstrained carbon-first allocation.
3. Screened direct water avoided relative to unconstrained carbon-first
   allocation.
4. Scenario-specific marginal carbon per screened liter avoided.

Reporting order:

1. Reference-case values.
2. Complete deterministic scenario envelope.
3. Conditional trace-segment intervals.
4. External directional check.

## 8. Permitted claims

- The model exposes a scenario-specific carbon-water-screening trade-off while
  preserving maximum feasible delivered service.
- A stricter screened-water cap changes modeled allocation and may impose a
  quantifiable modeled-carbon penalty.
- The direction of the trade-off is robust only over the explicitly enumerated
  frozen scenarios that pass all QA gates.
- Spatial WUE heterogeneity can change total direct-water accounting separately
  from categorical high-stress screening.

## 9. Prohibited claims

- measured carbon or water savings at real data centers;
- causal environmental effects;
- universal water sustainability or watershed benefit;
- real-time scheduler performance, latency, throughput, or energy-efficiency
  gains;
- representative population inference from one trace segment;
- independent environmental validation by the Azure trace;
- novel optimization algorithm or state-of-the-art computing system;
- recommendation of a single universally optimal cap.

## 10. Figure contract

All plotting, preview, export, and visual QA use Python only.

Provisional figures after analysis freeze:

1. Environmental accounting and epsilon-constraint decision architecture.
2. Reference cap-carbon frontier with service and binding-constraint checks.
3. Scenario envelope and marginal-carbon-per-screened-liter diagnostics.
4. Spatial WUE and external-trace boundary diagnostics.

Each main figure must have a conclusion, evidence ID map, source manifest,
editable SVG text, Type-3-free PDF, 600-dpi LZW TIFF, 300-dpi PNG, grayscale QA,
and a self-contained legend. The final figure count may change only if the
claim-to-evidence map is updated first.

## 11. Manuscript gate

Do not start the retargeted Word manuscript until:

- A1-A4 have run from a clean copied workspace;
- all solutions are optimal and feasibility residuals pass;
- the scenario envelope is complete or exclusions are documented;
- primary numbers are independently recomputed from output tables;
- the claim-to-evidence map is frozen;
- visual QA of the new figure set passes.

## 12. Revision decision rule

If the dense frontier is effectively flat, report the plateau as a result and
do not exaggerate novelty. If the environmental cap is infeasible in material
scenario cells at matched service, the manuscript must center feasibility
limits rather than claim robust co-benefits. If cap penalties change sign or
direction across the factorial, report heterogeneity rather than a single
headline percentage.
