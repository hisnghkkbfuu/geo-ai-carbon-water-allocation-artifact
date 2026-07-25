# Reproducibility sequence

`python run_all.py --quick` validates the frozen outputs, tests, source-data build, figure exports,
and export specifications. `python run_all.py --full` reruns all optimization blocks before the same
validation sequence. Both commands use repository-relative paths and require no network access.

The full scenario calculation is separated into three PUE blocks and then aggregated to make the
execution state inspectable and restartable.
