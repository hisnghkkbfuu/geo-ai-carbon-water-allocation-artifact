# Carbon-Water Trade-offs in Geo-distributed AI Inference Allocation

This repository is the reproducibility archive for the manuscript *Carbon-Water Trade-offs in Geo-distributed AI Inference Allocation under Water-Stress Screening*, prepared for *Sustainable Computing: Informatics and Systems*.

Repository: https://github.com/hisnghkkbfuu/geo-ai-carbon-water-allocation-artifact

## Scope

This package reproduces a scenario-bounded, trace-driven lexicographic allocation analysis. It supports directional mechanism and constraint-accounting results only. It does not contain operator facility telemetry, hydrologic-impact estimates, causal policy effects, or raw request traces.

## Contents

- `02_code/`: portable analysis and verification scripts.
- `03_tests/`: 25 deterministic unit tests.
- `01_data_processed/`: processed inputs and scenario tables.
- `04_results/`: frozen derived outputs and QA records.
- `figures/`: source data, generators, manifests, and SVG/PDF/TIFF/PNG exports for Fig. 1-4.
- `docs/`: source-data, derived-data, reproduction, and licence notes.
- `SHA256SUMS.csv`: exact file hashes for this release.

## Quick start

Install the packages in `requirements.txt`, then from repository root run:

```text
python -m unittest discover -s 03_tests -p "test_*.py" -v
```

The source datasets are not redistributed. The included Azure input builder requires explicit paths to official source files; the GenTD26 input-construction method is documented while the exact processed tables are supplied. See `docs/DATA_SOURCES.md` and `docs/REPRODUCIBILITY.md`.

## Citation

See `CITATION.cff`. A Zenodo DOI is added only after Zenodo archives the GitHub `v1.0.0` release.
