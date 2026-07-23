from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cws_lexicographic_model import (
    load_case,
    revalue_water,
    reverse_regional_carbon,
    solve_lexicographic,
)


def percent_change(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0) if baseline else np.nan


def public_result_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep frozen result tables stable while retaining provenance internally."""
    provenance_columns = [
        "carbon_filename",
        "capacity_filename",
        "energy_filename",
        "timestamp_column",
        "energy_workload_class",
    ]
    return frame.drop(columns=provenance_columns, errors="ignore")


def add_contrasts(metrics: pd.DataFrame, baseline_policy: str) -> pd.DataFrame:
    result = metrics.copy()
    base = result[result["policy"].eq(baseline_policy)].iloc[0]
    for column in [
        "served_accelerator_hours",
        "carbon_kgco2e",
        "direct_water_l",
        "high_stress_direct_water_l",
        "high_stress_water_share",
    ]:
        result[f"{column}_change_pct_vs_status_quo"] = result[column].map(
            lambda x: percent_change(float(x), float(base[column]))
        )
    return result


def matched_hourly_carbon(
    baseline_dispatch: pd.DataFrame, treatment_dispatch: pd.DataFrame
) -> dict[str, float]:
    """Retrospective same-hour accounting for cases with unequal service.

    The common quantity is the minimum served quantity in each hour. Dispatch
    rows are proportionally reweighted inside that hour; this is accounting,
    not a deployable causal policy comparison.
    """
    baseline = baseline_dispatch.groupby("hour_index")[
        "assigned_accelerator_hours"
    ].sum()
    treatment = treatment_dispatch.groupby("hour_index")[
        "assigned_accelerator_hours"
    ].sum()
    common = pd.concat([baseline, treatment], axis=1).fillna(0.0).min(axis=1)

    def reweighted(frame: pd.DataFrame, served: pd.Series) -> float:
        work = frame.copy()
        work["hour_served"] = work["hour_index"].map(served).fillna(0.0)
        work["common"] = work["hour_index"].map(common).fillna(0.0)
        scale = np.where(work["hour_served"].gt(0), work["common"] / work["hour_served"], 0.0)
        return float((work["carbon_kgco2e"] * scale).sum())

    common_served = float(common.sum())
    baseline_carbon = reweighted(baseline_dispatch, baseline)
    treatment_carbon = reweighted(treatment_dispatch, treatment)
    return {
        "common_served_accelerator_hours": common_served,
        "matched_baseline_carbon_kgco2e": baseline_carbon,
        "matched_treatment_carbon_kgco2e": treatment_carbon,
        "matched_baseline_carbon_per_common_accelerator_hour": (
            baseline_carbon / common_served if common_served else np.nan
        ),
        "matched_treatment_carbon_per_common_accelerator_hour": (
            treatment_carbon / common_served if common_served else np.nan
        ),
        "matched_carbon_change_pct": percent_change(
            treatment_carbon, baseline_carbon
        ),
    }


def circular_block_indices(
    n_hours: int, block_hours: int, rng: np.random.Generator
) -> np.ndarray:
    blocks = int(np.ceil(n_hours / block_hours))
    starts = rng.integers(0, n_hours, size=blocks)
    indices = np.concatenate(
        [(start + np.arange(block_hours)) % n_hours for start in starts]
    )
    return indices[:n_hours]


def paired_block_intervals(
    dispatch: pd.DataFrame,
    *,
    replicates: int = 4000,
    block_hours: int = 24,
    seed: int = 20260719,
) -> pd.DataFrame:
    policies = ["B0_true_no_migration_status_quo", "B1_carbon_first"]
    measures = [
        "assigned_accelerator_hours",
        "carbon_kgco2e",
        "direct_water_l",
        "high_stress_direct_water_l",
    ]
    hourly: dict[str, pd.DataFrame] = {}
    hours = sorted(dispatch["hour_index"].unique())
    for policy in policies:
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
        base = selected[policies[0]]
        treatment = selected[policies[1]]
        return {
            "carbon_change_pct": percent_change(
                float(treatment["carbon_kgco2e"]), float(base["carbon_kgco2e"])
            ),
            "service_difference_accelerator_hours": float(
                treatment["assigned_accelerator_hours"]
                - base["assigned_accelerator_hours"]
            ),
            "direct_water_change_l": float(
                treatment["direct_water_l"] - base["direct_water_l"]
            ),
            "high_stress_water_change_l": float(
                treatment["high_stress_direct_water_l"]
                - base["high_stress_direct_water_l"]
            ),
        }

    full = np.arange(len(hours))
    point = contrasts(full)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in point}
    for _ in range(replicates):
        values = contrasts(circular_block_indices(len(hours), block_hours, rng))
        for metric, value in values.items():
            samples[metric].append(value)
    rows = []
    for metric, estimate in point.items():
        values = pd.Series(samples[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "block_interval_low_2_5pct": float(values.quantile(0.025)),
                "block_interval_high_97_5pct": float(values.quantile(0.975)),
                "share_positive": float(values.gt(0).mean()),
                "replicates": replicates,
                "block_hours": block_hours,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def audit_solution(
    case,
    result: dict[str, object],
    *,
    migration_limit: float,
    max_latency_ms: float,
) -> dict[str, float]:
    dispatch = result["dispatch"]
    unserved = result["unserved"].set_index("hour_index")
    assigned_hour = dispatch.groupby("hour_index")[
        "assigned_accelerator_hours"
    ].sum()
    demand = pd.Series(
        case.demand_accelerator_hours, index=case.hour_index, dtype=float
    )
    balance = (
        assigned_hour.reindex(case.hour_index, fill_value=0.0)
        + unserved["unserved_accelerator_hours"].reindex(case.hour_index, fill_value=0.0)
        - demand
    )
    capacity = dispatch.pivot(
        index="hour_index", columns="region", values="assigned_accelerator_hours"
    ).reindex(index=case.hour_index, columns=case.regions, fill_value=0.0)
    capacity_violation = np.maximum(
        capacity.to_numpy() - case.capacity_accelerator_hours, 0.0
    )
    migrated = (
        dispatch[dispatch["migrated"]]
        .groupby("hour_index")["assigned_accelerator_hours"]
        .sum()
        .reindex(case.hour_index, fill_value=0.0)
    )
    migration_violation = np.maximum(
        migrated.to_numpy() - migration_limit * case.demand_accelerator_hours,
        0.0,
    )
    latency_violation = dispatch[
        dispatch["migrated"]
        & dispatch["latency_from_east_ms"].gt(max_latency_ms)
    ]["assigned_accelerator_hours"].sum()
    return {
        "max_abs_demand_balance_error": float(balance.abs().max()),
        "max_capacity_violation": float(capacity_violation.max()),
        "max_migration_violation": float(migration_violation.max()),
        "ineligible_latency_assignment": float(latency_violation),
    }


def main_experiment(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    case = load_case(root)
    runs: list[dict[str, object]] = []
    dispatches: list[pd.DataFrame] = []

    status_quo = solve_lexicographic(
        case,
        policy="B0_true_no_migration_status_quo",
        migration_share=0.0,
        max_latency_ms=20.0,
        second_stage_objective="carbon",
    )
    carbon_first = solve_lexicographic(
        case,
        policy="B1_carbon_first",
        migration_share=0.30,
        max_latency_ms=20.0,
        second_stage_objective="carbon",
    )
    water_first = solve_lexicographic(
        case,
        policy="B2_direct_water_first",
        migration_share=0.30,
        max_latency_ms=20.0,
        second_stage_objective="direct_water",
    )
    for run in [status_quo, carbon_first, water_first]:
        runs.append(run["metrics"])
        dispatches.append(run["dispatch"])

    cf_high_water = float(carbon_first["metrics"]["high_stress_direct_water_l"])
    frontier_rows: list[dict[str, object]] = []
    for fraction in [0.0, 0.25, 0.50, 0.75, 1.00]:
        cap = fraction * cf_high_water
        run = solve_lexicographic(
            case,
            policy=f"P_high_stress_cap_{fraction:.2f}",
            migration_share=0.30,
            max_latency_ms=20.0,
            second_stage_objective="carbon",
            high_stress_water_cap_l=cap,
        )
        row = dict(run["metrics"])
        row["cap_fraction_of_carbon_first"] = fraction
        frontier_rows.append(row)
        dispatches.append(run["dispatch"])

    metrics = add_contrasts(
        pd.DataFrame(runs), "B0_true_no_migration_status_quo"
    )
    frontier = pd.DataFrame(frontier_rows)
    qa = {
        "status_quo_migration_share": float(status_quo["metrics"]["migration_share"]),
        "status_quo_service_rate": float(status_quo["metrics"]["service_rate"]),
        "carbon_first_service_rate": float(carbon_first["metrics"]["service_rate"]),
        "service_difference_accelerator_hours": float(
            carbon_first["metrics"]["served_accelerator_hours"]
            - status_quo["metrics"]["served_accelerator_hours"]
        ),
        "carbon_first_carbon_change_pct": percent_change(
            float(carbon_first["metrics"]["carbon_kgco2e"]),
            float(status_quo["metrics"]["carbon_kgco2e"]),
        ),
        "carbon_first_high_stress_water_l": cf_high_water,
        "frontier_points": len(frontier_rows),
        "status_quo_constraint_audit": audit_solution(
            case, status_quo, migration_limit=0.0, max_latency_ms=20.0
        ),
        "carbon_first_constraint_audit": audit_solution(
            case, carbon_first, migration_limit=0.30, max_latency_ms=20.0
        ),
    }
    return metrics, frontier, {
        "qa": qa,
        "dispatch": pd.concat(dispatches, ignore_index=True),
    }


def sensitivity_matrix(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_cache: dict[tuple[str, str], dict[str, object]] = {}
    for carbon_mapping in [
        "conservative_site_selection",
        "equal_site_portfolio",
        "optimistic_site_selection",
    ]:
        for capacity_scenario in ["tight", "central", "ample"]:
            case = load_case(
                root,
                carbon_mapping=carbon_mapping,
                capacity_scenario=capacity_scenario,
            )
            key = (carbon_mapping, capacity_scenario)
            baseline_cache[key] = solve_lexicographic(
                case,
                policy="status_quo",
                migration_share=0.0,
                max_latency_ms=20.0,
                second_stage_objective="carbon",
            )
            for migration_share in [0.15, 0.30, 0.50]:
                for max_latency in [15.0, 20.0, 35.0]:
                    result = solve_lexicographic(
                        case,
                        policy="carbon_first",
                        migration_share=migration_share,
                        max_latency_ms=max_latency,
                        second_stage_objective="carbon",
                    )
                    base = baseline_cache[key]["metrics"]
                    matched_reoptimized = solve_lexicographic(
                        case,
                        policy="carbon_first_fixed_to_status_quo_service",
                        migration_share=migration_share,
                        max_latency_ms=max_latency,
                        second_stage_objective="carbon",
                        fixed_total_unserved_accelerator_hours=float(
                            base["unserved_accelerator_hours"]
                        ),
                    )["metrics"]
                    matched = matched_hourly_carbon(
                        baseline_cache[key]["dispatch"], result["dispatch"]
                    )
                    rows.append(
                        {
                            **result["metrics"],
                            **matched,
                            "carbon_change_pct_vs_matched_scenario_status_quo": percent_change(
                                float(result["metrics"]["carbon_kgco2e"]),
                                float(base["carbon_kgco2e"]),
                            ),
                            "service_rate_difference_vs_status_quo": float(
                                result["metrics"]["service_rate"] - base["service_rate"]
                            ),
                            "status_quo_service_rate": float(base["service_rate"]),
                            "reoptimized_matched_service_rate": float(
                                matched_reoptimized["service_rate"]
                            ),
                            "reoptimized_matched_carbon_kgco2e": float(
                                matched_reoptimized["carbon_kgco2e"]
                            ),
                            "reoptimized_matched_carbon_change_pct": percent_change(
                                float(matched_reoptimized["carbon_kgco2e"]),
                                float(base["carbon_kgco2e"]),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def wue_factorial_accounting(root: Path, dispatch: pd.DataFrame) -> pd.DataFrame:
    wue = pd.read_csv(root / "01_data_processed" / "wue_profiles.csv")
    wue = wue[wue["wue_profile"].str.startswith("factorial_")]
    risk = pd.read_csv(root / "01_data_processed" / "regional_water_risk_screening.csv")
    records: list[dict[str, object]] = []
    for risk_mapping, risk_frame in risk.groupby("water_risk_mapping"):
        high_map = (
            risk_frame.set_index("region")["high_or_extremely_high"]
            .astype(str)
            .str.lower()
            .eq("true")
            .to_dict()
        )
        for profile, profile_frame in wue.groupby("wue_profile"):
            wue_map = profile_frame.set_index("region")["wue_l_per_kwh_it"].astype(float).to_dict()
            valued: dict[str, dict[str, float]] = {}
            for policy in ["B0_true_no_migration_status_quo", "B1_carbon_first"]:
                policy_dispatch = dispatch[dispatch["policy"].eq(policy)]
                valued[policy] = revalue_water(policy_dispatch, wue_map, high_map)
            base = valued["B0_true_no_migration_status_quo"]
            treatment = valued["B1_carbon_first"]
            records.append(
                {
                    "water_risk_mapping": risk_mapping,
                    "wue_profile": profile,
                    "status_quo_direct_water_l": base["direct_water_l"],
                    "carbon_first_direct_water_l": treatment["direct_water_l"],
                    "direct_water_change_pct": percent_change(
                        treatment["direct_water_l"], base["direct_water_l"]
                    ),
                    "status_quo_high_stress_water_l": base[
                        "high_stress_direct_water_l"
                    ],
                    "carbon_first_high_stress_water_l": treatment[
                        "high_stress_direct_water_l"
                    ],
                    "high_stress_water_share_change_points": 100.0
                    * (
                        treatment["high_stress_water_share"]
                        - base["high_stress_water_share"]
                    ),
                }
            )
    return pd.DataFrame(records)


def reversal_test(root: Path) -> dict[str, float]:
    case = load_case(root)
    original = solve_lexicographic(
        case,
        policy="original_carbon",
        migration_share=0.30,
        max_latency_ms=20.0,
        second_stage_objective="carbon",
    )
    reversed_case = reverse_regional_carbon(case)
    reversed_run = solve_lexicographic(
        reversed_case,
        policy="reversed_carbon",
        migration_share=0.30,
        max_latency_ms=20.0,
        second_stage_objective="carbon",
    )
    original_alloc = original["dispatch"].groupby("region")[
        "assigned_accelerator_hours"
    ].sum()
    reversed_alloc = reversed_run["dispatch"].groupby("region")[
        "assigned_accelerator_hours"
    ].sum()
    difference = float(
        original_alloc.sub(reversed_alloc, fill_value=0.0).abs().sum()
    )
    return {
        "allocation_l1_difference_accelerator_hours": difference,
        "original_migration_share": float(original["metrics"]["migration_share"]),
        "reversed_migration_share": float(reversed_run["metrics"]["migration_share"]),
        "responded_to_reversal": bool(difference > 1e-6),
    }


def execution_status_sensitivity(root: Path) -> dict[str, float]:
    case = load_case(
        root,
        demand_filename="gentd26_hourly_all_positive_execution_demand.csv",
    )
    baseline = solve_lexicographic(
        case,
        policy="all_executed_status_quo",
        migration_share=0.0,
        max_latency_ms=20.0,
    )["metrics"]
    treatment = solve_lexicographic(
        case,
        policy="all_executed_carbon_first",
        migration_share=0.30,
        max_latency_ms=20.0,
    )["metrics"]
    return {
        "all_positive_execution_accelerator_hours": float(
            baseline["demand_accelerator_hours"]
        ),
        "service_rate_status_quo": float(baseline["service_rate"]),
        "service_rate_carbon_first": float(treatment["service_rate"]),
        "carbon_change_pct": percent_change(
            float(treatment["carbon_kgco2e"]), float(baseline["carbon_kgco2e"])
        ),
        "interpretation": (
            "includes failed and processing attempts with positive execution time; "
            "not delivered-service quantity"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    out = args.root / "04_results"
    out.mkdir(parents=True, exist_ok=True)

    metrics, frontier, main_payload = main_experiment(args.root)
    public_result_schema(metrics).to_csv(out / "main_policy_metrics_v1.csv", index=False)
    public_result_schema(frontier).to_csv(out / "high_stress_water_pareto_v1.csv", index=False)
    main_payload["dispatch"].to_csv(out / "main_policy_dispatch_v1.csv", index=False)
    intervals = paired_block_intervals(main_payload["dispatch"])
    intervals.to_csv(out / "main_policy_paired_block_intervals_v1.csv", index=False)

    sensitivity = sensitivity_matrix(args.root)
    public_result_schema(sensitivity).to_csv(out / "carbon_service_sensitivity_matrix_v1.csv", index=False)
    wue = wue_factorial_accounting(args.root, main_payload["dispatch"])
    wue.to_csv(out / "wue_factorial_accounting_v1.csv", index=False)
    reversal = reversal_test(args.root)
    execution_status = execution_status_sensitivity(args.root)

    summary = {
        "status": "ANALYZED",
        "main": main_payload["qa"],
        "paired_block_intervals": intervals.to_dict(orient="records"),
        "sensitivity": {
            "runs": int(len(sensitivity)),
            "solver_failures": 0,
            "carbon_change_min_pct": float(
                sensitivity[
                    "carbon_change_pct_vs_matched_scenario_status_quo"
                ].min()
            ),
            "carbon_change_max_pct": float(
                sensitivity[
                    "carbon_change_pct_vs_matched_scenario_status_quo"
                ].max()
            ),
            "matched_carbon_change_min_pct": float(
                sensitivity["matched_carbon_change_pct"].min()
            ),
            "matched_carbon_change_max_pct": float(
                sensitivity["matched_carbon_change_pct"].max()
            ),
            "reoptimized_matched_carbon_change_min_pct": float(
                sensitivity["reoptimized_matched_carbon_change_pct"].min()
            ),
            "reoptimized_matched_carbon_change_max_pct": float(
                sensitivity["reoptimized_matched_carbon_change_pct"].max()
            ),
            "service_rate_difference_min": float(
                sensitivity["service_rate_difference_vs_status_quo"].min()
            ),
            "service_rate_difference_max": float(
                sensitivity["service_rate_difference_vs_status_quo"].max()
            ),
        },
        "wue_factorial": {
            "rows": int(len(wue)),
            "profiles": int(wue["wue_profile"].nunique()),
            "risk_mappings": int(wue["water_risk_mapping"].nunique()),
            "share_carbon_first_increases_total_direct_water": float(
                wue["direct_water_change_pct"].gt(1e-9).mean()
            ),
            "share_carbon_first_increases_high_stress_share": float(
                wue["high_stress_water_share_change_points"].gt(1e-9).mean()
            ),
        },
        "reversal_test": reversal,
        "execution_status_sensitivity": execution_status,
        "claim_boundary": (
            "Mechanism scenario using public traces, measured power calibration and projected "
            "hourly carbon factors; not observed EDWC operator outcomes."
        ),
    }
    (out / "stage1_experiment_qa.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
