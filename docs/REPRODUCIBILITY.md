# Reproducibility guide

Use Python 3.11 or later and install `requirements.txt`. From repository root run:

```text
python -m unittest discover -s 03_tests -p "test_*.py" -v
```

The frozen validation suite contains 25 deterministic tests. The included Azure source-input builder requires explicit user-supplied official source paths and contains no local-machine or other-project defaults. The GenTD26 input builder is not distributed because it depends on raw sources whose redistribution terms differ; the exact processed tables used for this manuscript are included.

The analysis remains a public-data and explicit-scenario mechanism study, not measured operational performance, site-specific hydrological impact, or causal policy evaluation. Code and documentation are MIT licensed; author-created derived tables and figure-source CSVs are CC BY 4.0; third-party source terms remain controlling.
