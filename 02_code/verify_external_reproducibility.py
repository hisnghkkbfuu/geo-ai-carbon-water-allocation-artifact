from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


TRACKED_FILES = (
    "00_governance/external_confirmation_provenance_v1.json",
    "01_data_processed/external_vidur_execution_time_lookup_v1.csv",
    "01_data_processed/azure_code_hourly_service_demand_v1.csv",
    "01_data_processed/azure_code_aligned_regional_hourly_cef_v1.csv",
    "01_data_processed/azure_code_nonbinding_capacity_v1.csv",
    "01_data_processed/external_llm_energy_calibration_v1.csv",
    "01_data_processed/external_confirmation_input_qa_v1.json",
    "01_data_processed/azure_code_hourly_service_demand_triangulated_linear_exploratory_v1.csv",
    "01_data_processed/azure_code_nonbinding_capacity_triangulated_linear_exploratory_v1.csv",
    "01_data_processed/azure_code_hourly_service_demand_local_simplex_upper_exploratory_v1.csv",
    "01_data_processed/azure_code_nonbinding_capacity_local_simplex_upper_exploratory_v1.csv",
    "01_data_processed/external_mapping_sensitivity_input_qa_v1.json",
    "04_results/external_confirmation_policy_metrics_v1.csv",
    "04_results/external_confirmation_policy_dispatch_v1.csv",
    "04_results/external_confirmation_paired_block_intervals_v1.csv",
    "04_results/external_confirmation_experiment_qa_v1.json",
    "04_results/external_mapping_sensitivity_policy_metrics_v1.csv",
    "04_results/external_mapping_sensitivity_policy_dispatch_v1.csv",
    "04_results/external_mapping_sensitivity_intervals_v1.csv",
    "04_results/external_mapping_sensitivity_qa_v1.json",
    "04_results/external_daily_stratified_effects_v1.csv",
    "04_results/external_daily_stratified_audit_v1.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def hashes(root: Path) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in TRACKED_FILES}


def run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-trace", type=Path, required=True)
    parser.add_argument("--vidur-metrics", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    before = hashes(args.root)

    run(
        [
            sys.executable,
            "02_code/build_external_confirmation_inputs.py",
            "--raw-trace",
            str(args.raw_trace),
            "--vidur-metrics",
            str(args.vidur_metrics),
        ],
        args.root,
    )
    run(
        [
            sys.executable,
            "02_code/build_external_mapping_sensitivities.py",
            "--raw-trace",
            str(args.raw_trace),
        ],
        args.root,
    )
    run([sys.executable, "02_code/run_external_confirmation.py"], args.root)
    run([sys.executable, "02_code/run_external_mapping_sensitivities.py"], args.root)
    run([sys.executable, "02_code/build_external_stratified_audit.py"], args.root)
    test_run = run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "03_tests",
            "-p",
            "test_*.py",
            "-v",
        ],
        args.root,
    )

    after = hashes(args.root)
    exact = {relative: before[relative] == after[relative] for relative in TRACKED_FILES}
    manifest = {
        "status": "PASS" if all(exact.values()) else "FAIL",
        "tracked_files": len(TRACKED_FILES),
        "exact_sha256_matches": int(sum(exact.values())),
        "raw_trace_sha256": sha256_file(args.raw_trace),
        "vidur_metrics_sha256": sha256_file(args.vidur_metrics),
        "files": {
            relative: {
                "before_sha256": before[relative],
                "after_sha256": after[relative],
                "exact": exact[relative],
            }
            for relative in TRACKED_FILES
        },
        "tests": {
            "returncode": test_run.returncode,
            "summary": next(
                (
                    line.strip()
                    for line in reversed(test_run.stderr.splitlines())
                    if line.startswith("Ran ")
                ),
                "unittest completed",
            ),
        },
    }
    output = args.root / "04_results" / "external_reproducibility_manifest_v1.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "PASS":
        raise RuntimeError("External reconstruction outputs are not deterministic")


if __name__ == "__main__":
    main()
