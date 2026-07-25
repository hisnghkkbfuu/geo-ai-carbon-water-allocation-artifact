from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(*parts: str) -> None:
    command = [PYTHON, *parts]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()

    if args.full:
        run("02_code/run_ema_environmental_frontier.py", "--root", ".", "--mode", "reference")
        for pue in ("hub_policy_compliant", "uniform_new_build", "legacy_stress"):
            run(
                "02_code/run_ema_environmental_frontier.py",
                "--root",
                ".",
                "--mode",
                "scenario",
                "--pue-scenario",
                pue,
            )
        run("02_code/run_ema_environmental_frontier.py", "--root", ".", "--mode", "aggregate")
        run("02_code/run_ema_environmental_frontier.py", "--root", ".", "--mode", "wue")
        run("02_code/run_ema_environmental_frontier.py", "--root", ".", "--mode", "energy")
        run("02_code/run_ema_reference_block_resampling.py", "--root", ".")

    run("02_code/verify_ema_environmental_frontier.py", "--root", ".")
    run("-m", "unittest", "discover", "-s", "03_tests", "-p", "test_*.py", "-v")
    run("02_code/build_ema_figure_source_data.py", "--root", ".")
    run("02_code/generate_ema_figures.py", "--root", ".")
    run("02_code/qa_ema_figure_exports.py", "--root", ".")


if __name__ == "__main__":
    main()
