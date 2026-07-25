# Figure legends

## Fig. 1 | Environmental decision model and accounting boundary

**a,** Trace-derived hourly service proxies and regional proxy/scenario layers
enter a feasible allocation model; carbon and direct-water outputs are kept
separate. **b,** Assigned service is converted to IT electricity. PUE enters
facility electricity and carbon, whereas direct water is calculated from IT
electricity and WUE; water-stress categories screen the location of physical
liters rather than multiply them. **c,** Stage 1 minimizes unserved service.
Stage 2 fixes maximum feasible service and minimizes modeled carbon subject to a
screened direct-water cap. The diagram is a model definition, not an observed
data-center process map. It omits cross-hour queueing, facility metering, causal
effects, and hydrologic damage. Source data: Fig. 1 model-definition table.

## Fig. 2 | Reference screened-water frontier at equal service

**a,** Twenty-one reoptimized cap fractions from 0% to 100% of the unconstrained
carbon-first screened direct-water exposure are shown. Tighter caps avoid more
screened direct water and impose a monotone modeled-carbon penalty. **b,** The
model-accounted carbon change relative to matched-service no migration remains
below zero across the reference cap grid. **c,** Adjacent-cap marginal additional
carbon per screened liter avoided rises as the cap approaches zero. **d,** The
regional allocation shifts from screened north/northwest locations toward the
unscreened southwest location as the cap tightens; distinct hatches distinguish
the two screened regions. Every point preserves 211.947 accelerator-hour-equivalent service;
maximum feasibility residual is 1e-8. Values are deterministic reference-case
outputs, not inferential statistics. For the zero-cap endpoint, paired 24-h
circular-block quantiles across 4,000 resamples are -13.666% to -12.270% for
carbon change versus no migration and 0.535 to 4.775 L for screened direct water
avoided; these quantify temporal composition within the supplied fixed solved
trace segment only. Source data: Fig. 2 source tables.

## Fig. 3 | PUE-aware deterministic scenario envelope

**a,** For 243 carbon/PUE/capacity/migration/latency scenario cells, thin ranges
show the minimum and maximum, thick ranges show the 25th to 75th percentiles, and
open circles show medians across the enumerated scenarios. They are not
confidence intervals. **b,** All 81 zero-cap cells per PUE scenario remain at or
below their matched-service no-migration carbon value, including cells with
changes close to zero. **c,** Marginal carbon costs across the 780 defined
adjacent-cap segments are shown by cap transition; 192 zero-water-avoidance
segments have undefined ratios and are omitted. **d,** Solver and constraint QA:
243 scenario cells and 1,215 optimal cap-constrained solves, with no observed
demand-balance, capacity, migration, latency, cap, or matched-service violation
above tolerance. Source data: Fig. 3 source tables.

## Fig. 4 | Spatial WUE and energy-scale boundary checks

**a,** Full reoptimization across 81 spatial WUE profiles changes the intermediate
carbon penalty, while the 0%- and 100%-allowed-screened-water endpoints coincide
for the reference settings. Thin, thick, and open marks denote deterministic min-max, interquartile,
and median summaries of the complete profile enumeration. **b,** Profile-level
25%-allowed penalties are plotted against the mean WUE of screened regions; marker
shape and color both encode whether the screened-region mean is less than or
equal to the unscreened-region mean. No fit or sampling interpretation is used.
**c,** Regional allocation shares vary with the cap but are identical to numerical
tolerance for the low, central, and high region-invariant energy-intensity
levels. **d,** Those energy levels rescale absolute modeled carbon without
changing allocation or relative carbon conclusions. Source data: Fig. 4 source
tables.
