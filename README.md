# Reproducibility Artifact for Carbon-Water Trade-offs in Geo-distributed AI Inference Allocation

This repository is the independent v1.0.0 reproducibility artifact for the manuscript *Carbon-Water Trade-offs in Geo-distributed AI Inference Allocation under Water-Stress Screening*, prepared for *Sustainable Computing: Informatics and Systems*.

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
- `DATA_LICENSE.md`: CC BY 4.0 boundary for author-created derived tables and figure source CSVs.
- `SHA256SUMS.csv`: exact file hashes for this release.

## Quick start

Install the packages in `requirements.txt`, then from repository root run the deterministic test gate:

```text
python -m unittest discover -s 03_tests -p "test_*.py" -v
```

For the core reconstruction entry, use a clean copy of the repository and run:

```text
python 02_code/run_stage1_reconstruction_experiments.py
```

This regenerates the core policy metrics, dispatch table, sensitivity matrix,
paired-block intervals, WUE accounting table, and `stage1_experiment_qa.json`
under `04_results/`. Run it from a disposable clean copy because it refreshes
those derived outputs. The included source datasets are not redistributed. The
source-input builders require explicit paths to official provider files and are
not part of the core no-raw-data reproduction path. See `docs/DATA_SOURCES.md`,
`docs/DERIVED_DATA.md`, and `docs/REPRODUCIBILITY.md`.

## License boundary

Author-written code and documentation are MIT licensed. Author-created derived
tables and figure source CSVs are CC BY 4.0. Third-party raw data and source
materials remain governed by their upstream terms and are not redistributed.

## Citation

See `CITATION.cff`. Zenodo version and concept DOIs are added to the release
metadata only after the independent Zenodo record is published.
