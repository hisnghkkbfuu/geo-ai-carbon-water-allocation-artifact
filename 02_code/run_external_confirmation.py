from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cws_lexicographic_model import load_case, solve_lexicographic
from run_stage1_reconstruction_experiments import audit_solution, circular_block_indices


BASELINE = "E0_true_no_migration_status_quo"
CARBON_FIRST = "E1_carbon_first"
ZERO_HIGH_STRESS = "E2_zero_high_stress_exposure"
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_BLOCK_HOURS = 24
BOOTSTRAP_SEED = 20260720


def percent_change(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0) if baseline else np.nan


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def add_external_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    baseline = result[result["policy"].eq(BASELINE)].iloc[0]
    for column in [
        "served_accelerator_hours",
        "carbon_kgco2e",
        "direct_water_l",
        "high_stress_direct_water_l",
        "high_stress_water_share",
    ]:
        result[f"{column}_change_pct_vs_e0"] = result[column].map(
            lambda value: percent_change(float(value), float(baseline[column]))
        )
    result["service_difference_accelerator_hours_vs_e0"] = (
        result["served_accelerator_hours"]
        - float(baseline["served_accelerator_hours"])
    )
    result["high_stress_water_change_l_vs_e0"] = (
        result["high_stress_direct_water_l"]
        - float(baseline["high_stress_direct_water_l"])
    )
    return result


def external_paired_block_intervals(dispatch: pd.DataFrame) -> pd.DataFrame:
    measures = [
        "assigned_accelerator_hours",
        "carbon_kgco2e",
        "direct_water_l",
        "high_stress_direct_water_l",
    ]
    hours = sorted(dispatch["hour_index"].unique())
    hourly: dict[str, pd.DataFrame] = {}
    for policy in [BASELINE, CARBON_FIRST, ZERO_HIGH_STRESS]:
        hourly[policy] = (
            dispatch[dispatch["policy"].eq(policy)]
            .groupby("hour_index")[measures]
            .sum()
            .reindex(hours, fill_value=0.0)
        )

    def contrasts(indices: np.ndarray) -> dict[str, float]:
        selected = {
            policy: frame.iloc[indices].sum() for policy, frame in hourly.items()
        }
        e0 = selected[BASELINE]
        e1 = selected[CARBON_FIRST]
        e2 = selected[ZERO_HIGH_STRESS]
        return {
            "e1_vs_e0_carbon_change_pct": percent_change(
                float(e1["carbon_kgco2e"]), float(e0["carbon_kgco2e"])
            ),
            "e1_vs_e0_high_stress_water_change_l": float(
                e1["high_stress_direct_water_l"]
                - e0["high_stress_direct_water_l"]
            ),
            "e2_vs_e0_carbon_change_pct": percent_change(
                float(e2["carbon_kgco2e"]), float(e0["carbon_kgco2e"])
            ),
            "e1_vs_e0_service_difference_accelerator_hours": float(
                e1["assigned_accelerator_hours"]
                - e0["assigned_accelerator_hours"]
            ),
            "e2_vs_e0_service_difference_accelerator_hours": float(
                e2["assigned_accelerator_hours"]
                - e0["assigned_accelerator_hours"]
            ),
        }

    point = contrasts(np.arange(len(hours)))
    samples = {metric: [] for metric in point}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = circular_block_indices(len(hours), BOOTSTRAP_BLOCK_HOURS, rng)
        values = contrasts(indices)
        for metric, value in values.items():
            samples[metric].append(value)

    rows: list[dict[str, object]] = []
    for metric, estimate in point.items():
        values = pd.Series(samples[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "block_interval_low_2_5pct": float(values.quantile(0.025)),
                "block_interval_high_97_5pct": float(values.quantile(0.975)),
                "share_positive": float(values.gt(0).mean()),
                "replicates": BOOTSTRAP_REPLICATES,
                "block_hours": BOOTSTRAP_BLOCK_HOURS,
                "seed": BOOTSTRAP_SEED,
            }
        )
    return pd.DataFrame(rows)


def evaluate_frozen_endpoints(
    metrics: pd.DataFrame, intervals: pd.DataFrame
) -> dict[str, object]:
    interval = intervals.set_index("metric")
    e0 = metrics[metrics["policy"].eq(BASELINE)].iloc[0]
    e1 = metrics[metrics["policy"].eq(CARBON_FIRST)].iloc[0]
    e2 = metrics[metrics["policy"].eq(ZERO_HIGH_STRESS)].iloc[0]
    service_tolerance = 1e-6 * float(e0["demand_accelerator_hours"])
    service_differences = {
        CARBON_FIRST: abs(
            float(e1["served_accelerator_hours"])
            - float(e0["served_accelerator_hours"])
        ),
        ZERO_HIGH_STRESS: abs(
            float(e2["served_accelerator_hours"])
            - float(e0["served_accelerator_hours"])
        ),
    }
    checks = {
        "endpoint_1_e1_carbon_direction_confirmed": bool(
            interval.loc["e1_vs_e0_carbon_change_pct", "estimate"] < 0
            and interval.loc[
                "e1_vs_e0_carbon_change_pct", "block_interval_high_97_5pct"
            ]
            < 0
        ),
        "endpoint_2_e1_high_stress_transfer_confirmed": bool(
            interval.loc[
                "e1_vs_e0_high_stress_water_change_l", "estimate"
            ]
            > 0
            and interval.loc[
                "e1_vs_e0_high_stress_water_change_l",
                "block_interval_low_2_5pct",
            ]
            > 0
        ),
        "endpoint_3_e2_carbon_direction_retained": bool(
            interval.loc["e2_vs_e0_carbon_change_pct", "estimate"] < 0
            and interval.loc[
                "e2_vs_e0_carbon_change_pct", "block_interval_high_97_5pct"
            ]
            < 0
        ),
        "endpoint_4_equal_service": bool(
            max(service_differences.values()) < service_tolerance
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "service_tolerance_accelerator_hours": service_tolerance,
        "absolute_service_differences_accelerator_hours": service_differences,
        "frozen_protocol": "00_governance/external_confirmation_protocol_v1.md",
        "mapping_addendum": (
            "00_governance/external_confirmation_mapping_addendum_v1.md"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    result_dir = args.root / "04_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    case = load_case(
        args.root,
        carbon_mapping="equal_site_portfolio",
        pue_scenario="hub_policy_compliant",
        capacity_scenario="external_nonbinding",
        wue_profile="uniform_google_ai_2024",
        water_risk_mapping="portfolio_mean",
        energy_level="central",
        demand_filename="azure_code_hourly_service_demand_v1.csv",
        carbon_filename="azure_code_aligned_regional_hourly_cef_v1.csv",
        capacity_filename="azure_code_nonbinding_capacity_v1.csv",
        energy_filename="external_llm_energy_calibration_v1.csv",
        timestamp_column="trace_timestamp_utc",
        energy_workload_class="text_generation_llm_serving_proxy",
    )
    run_specs = [
        {
            "policy": BASELINE,
            "migration_share": 0.0,
            "max_latency_ms": 20.0,
        },
        {
            "policy": CARBON_FIRST,
            "migration_share": 0.30,
            "max_latency_ms": 20.0,
        },
        {
            "policy": ZERO_HIGH_STRESS,
            "migration_share": 0.30,
            "max_latency_ms": 20.0,
            "high_stress_water_cap_l": 0.0,
        },
    ]
    runs: list[dict[str, object]] = []
    dispatches: list[pd.DataFrame] = []
    audits: dict[str, dict[str, float]] = {}
    for spec in run_specs:
        result = solve_lexicographic(
            case,
            second_stage_objective="carbon",
            **spec,
        )
        runs.append(result["metrics"])
        dispatches.append(result["dispatch"])
        audits[str(spec["policy"])] = audit_solution(
            case,
            result,
            migration_limit=float(spec["migration_share"]),
            max_latency_ms=float(spec["max_latency_ms"]),
        )

    metrics = add_external_contrasts(pd.DataFrame(runs))
    dispatch = pd.concat(dispatches, ignore_index=True)
    intervals = external_paired_block_intervals(dispatch)
    endpoint_result = evaluate_frozen_endpoints(metrics, intervals)

    metrics_path = result_dir / "external_confirmation_policy_metrics_v1.csv"
    dispatch_path = result_dir / "external_confirmation_policy_dispatch_v1.csv"
    intervals_path = result_dir / "external_confirmation_paired_block_intervals_v1.csv"
    metrics.to_csv(metrics_path, index=False)
    dispatch.to_csv(dispatch_path, index=False)
    intervals.to_csv(intervals_path, index=False)

    max_audit_value = max(
        value for audit in audits.values() for value in audit.values()
    )
    input_qa = json.loads(
        (args.root / "01_data_processed" / "external_confirmation_input_qa_v1.json").read_text(
            encoding="utf-8"
        )
    )
    constraint_checks = {
        "solution_audits_below_1e_7": bool(max_audit_value <= 1e-7),
        "baseline_migration_is_zero": bool(
            abs(
                float(
                    metrics.loc[
                        metrics["policy"].eq(BASELINE), "migration_share"
                    ].iloc[0]
                )
            )
            <= 1e-12
        ),
        "e2_high_stress_exposure_is_zero": bool(
            abs(
                float(
                    metrics.loc[
                        metrics["policy"].eq(ZERO_HIGH_STRESS),
                        "high_stress_direct_water_l",
                    ].iloc[0]
                )
            )
            <= 1e-7
        ),
    }
    endpoint_and_constraints_pass = bool(
        endpoint_result["status"] == "PASS" and all(constraint_checks.values())
    )
    boundary_service_share = float(
        input_qa["trace"]["boundary_fallback_service_share"]
    )
    qa = {
        "status": "ANALYZED",
        "validation_status": (
            "FROZEN_ENDPOINTS_PASS_WITH_MAJOR_CALIBRATION_LIMITATION"
            if endpoint_and_constraints_pass
            else "FROZEN_CONFIRMATION_FAIL"
        ),
        "evidence_grade": "DIRECTIONAL_MECHANISM_ONLY",
        "calibration_limitation": {
            "boundary_fallback_service_share": boundary_service_share,
            "interpretation": (
                "The frozen global upper-bound fallback dominates absolute service, carbon, "
                "and water magnitudes; use post-confirmation mapping sensitivities for "
                "directional robustness and do not report absolute footprint estimates."
            ),
        },
        "frozen_endpoint_result": endpoint_result,
        "constraint_checks": constraint_checks,
        "solution_audits": audits,
        "max_solution_audit_violation": max_audit_value,
        "baseline_migration_share": float(
            metrics.loc[metrics["policy"].eq(BASELINE), "migration_share"].iloc[0]
        ),
        "zero_high_stress_policy_exposure_l": float(
            metrics.loc[
                metrics["policy"].eq(ZERO_HIGH_STRESS),
                "high_stress_direct_water_l",
            ].iloc[0]
        ),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "block_hours": BOOTSTRAP_BLOCK_HOURS,
            "seed": BOOTSTRAP_SEED,
            "interpretation": (
                "paired circular-block interval over this trace segment; not a population p-value"
            ),
        },
        "output_sha256": {
            metrics_path.name: sha256_file(metrics_path),
            dispatch_path.name: sha256_file(dispatch_path),
            intervals_path.name: sha256_file(intervals_path),
        },
    }
    qa_path = result_dir / "external_confirmation_experiment_qa_v1.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
