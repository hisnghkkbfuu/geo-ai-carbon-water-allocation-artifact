# Environmental screening of carbon-water trade-offs

This is the `v2.0.0` reproducibility artifact for the trace-driven scenario model described by
the accompanying manuscript. The model preserves maximum feasible service before minimizing
model-accounted carbon under an allowed screened direct-water constraint.

## Scientific boundary

The artifact supports scenario-specific, directional mechanism analysis only. It does not report
measurements from a named data center, estimate watershed damage, identify causal policy effects,
or validate a production scheduler. Carbon, direct water, and categorical water-stress screening
remain separate model layers.

## Contents

- `01_data_processed/`: redistributable derived inputs used by the released analysis.
- `02_code/`: model, experiment, resampling, figure, and verification code.
- `03_tests/`: deterministic unit and integration tests.
- `03_results/`: frozen reference, factorial, WUE, energy-scale, and resampling outputs.
- `figures/source_data/`: figure-level source tables and manifest.
- `figures/exports/`: editable/raster publication exports.
- `06_qa/`: frozen scientific and figure QA results.
- `docs/`: data provenance, derived-data, reproduction, and scientific-boundary notes.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Quick verification

This checks the frozen result invariants, runs all tests, rebuilds figure source data and exports,
and verifies the exports:

```bash
python run_all.py --quick
```

## Full reproduction

This reruns the 21-point reference frontier, all three PUE blocks of the 243-cell deterministic
factorial, aggregation, 81-profile WUE analysis, energy-scale check, 4,000 paired circular-block
resamples, figure generation, tests, and QA:

```bash
python run_all.py --full
```

## Licenses and excluded sources

Author-written code and documentation are MIT licensed. Author-created derived tables and figure
source data are CC BY 4.0. Third-party raw request traces are excluded; see
`docs/DATA_SOURCES.md`.

## Version links

- Repository: https://github.com/hisnghkkbfuu/geo-ai-carbon-water-allocation-artifact
- GitHub release: https://github.com/hisnghkkbfuu/geo-ai-carbon-water-allocation-artifact/releases/tag/v2.0.0
- Zenodo concept DOI for this second-paper artifact family: https://doi.org/10.5281/zenodo.21512280

The immutable `v2.0.0` Zenodo version DOI is recorded on the GitHub release page after Zenodo
publishes the new version.
