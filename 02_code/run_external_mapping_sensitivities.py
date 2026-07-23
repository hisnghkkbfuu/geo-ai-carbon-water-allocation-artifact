from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from cws_lexicographic_model import load_case, solve_lexicographic
from run_external_confirmation import (
    BASELINE,
    CARBON_FIRST,
    ZERO_HIGH_STRESS,
    add_external_contrasts,
    evaluate_frozen_endpoints,
    external_paired_block_intervals,
)
from run_stage1_reconstruction_experiments import audit_solution


MAPPINGS = ("triangulated_linear", "local_simplex_upper")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    result_dir = args.root / "04_results"
    all_metrics: list[pd.DataFrame] = []
    all_dispatch: list[pd.DataFrame] = []
    all_intervals: list[pd.DataFrame] = []
    endpoint_diagnostics: dict[str, object] = {}
    audits: dict[str, object] = {}

    for mapping in MAPPINGS:
        case = load_case(
            args.root,
            carbon_mapping="equal_site_portfolio",
            pue_scenario="hub_policy_compliant",
            capacity_scenario="external_nonbinding",
            wue_profile="uniform_google_ai_2024",
            water_risk_mapping="portfolio_mean",
            energy_level="central",
            demand_filename=(
                f"azure_code_hourly_service_demand_{mapping}_exploratory_v1.csv"
            ),
            carbon_filename="azure_code_aligned_regional_hourly_cef_v1.csv",
            capacity_filename=(
                f"azure_code_nonbinding_capacity_{mapping}_exploratory_v1.csv"
            ),
            energy_filename="external_llm_energy_calibration_v1.csv",
            timestamp_column="trace_timestamp_utc",
            energy_workload_class="text_generation_llm_serving_proxy",
        )
        specs = [
            {"policy": BASELINE, "migration_share": 0.0, "max_latency_ms": 20.0},
            {"policy": CARBON_FIRST, "migration_share": 0.30, "max_latency_ms": 20.0},
            {
                "policy": ZERO_HIGH_STRESS,
                "migration_share": 0.30,
                "max_latency_ms": 20.0,
                "high_stress_water_cap_l": 0.0,
            },
        ]
        metrics_rows: list[dict[str, object]] = []
        dispatch_rows: list[pd.DataFrame] = []
        audits[mapping] = {}
        for spec in specs:
            result = solve_lexicographic(
                case, second_stage_objective="carbon", **spec
            )
            metrics_rows.append(result["metrics"])
            dispatch_rows.append(result["dispatch"])
            audits[mapping][str(spec["policy"])] = audit_solution(
                case,
                result,
                migration_limit=float(spec["migration_share"]),
                max_latency_ms=float(spec["max_latency_ms"]),
            )
        metrics = add_external_contrasts(pd.DataFrame(metrics_rows))
        metrics.insert(0, "service_mapping", mapping)
        dispatch = pd.concat(dispatch_rows, ignore_index=True)
        dispatch.insert(0, "service_mapping", mapping)
        intervals = external_paired_block_intervals(dispatch)
        intervals.insert(0, "service_mapping", mapping)
        diagnostic = evaluate_frozen_endpoints(metrics, intervals)
        diagnostic["status_label"] = "EXPLORATORY_DIRECTIONAL_DIAGNOSTIC"
        endpoint_diagnostics[mapping] = diagnostic
        all_metrics.append(metrics)
        all_dispatch.append(dispatch)
        all_intervals.append(intervals)

    metrics_path = result_dir / "external_mapping_sensitivity_policy_metrics_v1.csv"
    dispatch_path = result_dir / "external_mapping_sensitivity_policy_dispatch_v1.csv"
    intervals_path = result_dir / "external_mapping_sensitivity_intervals_v1.csv"
    pd.concat(all_metrics, ignore_index=True).to_csv(metrics_path, index=False)
    pd.concat(all_dispatch, ignore_index=True).to_csv(dispatch_path, index=False)
    pd.concat(all_intervals, ignore_index=True).to_csv(intervals_path, index=False)
    qa = {
        "status": "POST_CONFIRMATION_EXPLORATORY",
        "endpoint_diagnostics": endpoint_diagnostics,
        "solution_audits": audits,
        "output_sha256": {
            metrics_path.name: sha256_file(metrics_path),
            dispatch_path.name: sha256_file(dispatch_path),
            intervals_path.name: sha256_file(intervals_path),
        },
    }
    (result_dir / "external_mapping_sensitivity_qa_v1.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
