from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cws_lexicographic_model import CaseData, load_case, solve_lexicographic


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-7
REFERENCE_CAP_FRACTIONS = tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
COARSE_CAP_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
CARBON_MAPPINGS = (
    "conservative_site_selection",
    "equal_site_portfolio",
    "optimistic_site_selection",
)
PUE_SCENARIOS = ("hub_policy_compliant", "uniform_new_build", "legacy_stress")
CAPACITY_SCENARIOS = ("tight", "central", "ample")
MIGRATION_LIMITS = (0.15, 0.30, 0.50)
LATENCY_LIMITS_MS = (15.0, 20.0, 35.0)
ENERGY_LEVELS = ("low", "central", "high")


def percent_change(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return float("nan")
    return 100.0 * (value / baseline - 1.0)


def _assert_optimal(result: dict[str, object]) -> None:
    metadata = result["solver_metadata"]
    if not metadata["stage_one_success"] or not metadata["stage_two_success"]:
        raise RuntimeError(f"Non-optimal solve: {metadata}")


def solve_status_quo(case: CaseData) -> dict[str, object]:
    result = solve_lexicographic(
        case,
        policy="status_quo_no_migration",
        migration_share=0.0,
        max_latency_ms=20.0,
        second_stage_objective="carbon",
    )
    _assert_optimal(result)
    return result


def solve_carbon_first_matched(
    case: CaseData,
    *,
    migration_limit: float,
    max_latency_ms: float,
    baseline_unserved: float,
    policy: str,
    high_stress_water_cap_l: float | None = None,
) -> dict[str, object]:
    result = solve_lexicographic(
        case,
        policy=policy,
        migration_share=migration_limit,
        max_latency_ms=max_latency_ms,
        second_stage_objective="carbon",
        high_stress_water_cap_l=high_stress_water_cap_l,
        fixed_total_unserved_accelerator_hours=baseline_unserved,
    )
    _assert_optimal(result)
    return result


def frontier_diagnostics(
    case: CaseData,
    result: dict[str, object],
    *,
    migration_limit: float,
    max_latency_ms: float,
    target_unserved: float,
    cap_l: float | None,
) -> dict[str, float | int | bool]:
    dispatch = result["dispatch"]
    unserved = result["unserved"].set_index("hour_index")[
        "unserved_accelerator_hours"
    ].reindex(case.hour_index, fill_value=0.0)
    assigned = (
        dispatch.pivot(index="hour_index", columns="region", values="assigned_accelerator_hours")
        .reindex(index=case.hour_index, columns=case.regions, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    demand_balance = assigned.sum(axis=1) + unserved.to_numpy() - case.demand_accelerator_hours
    capacity_slack = case.capacity_accelerator_hours - assigned

    frame = dispatch.copy()
    migrated_hourly = (
        frame.loc[frame["migrated"]]
        .groupby("hour_index")["assigned_accelerator_hours"]
        .sum()
        .reindex(case.hour_index, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    migration_slack = migration_limit * case.demand_accelerator_hours - migrated_hourly
    origin = case.regions.index("east")
    ineligible = np.array(
        [region != "east" and case.latency_ms[idx] > max_latency_ms for idx, region in enumerate(case.regions)],
        dtype=bool,
    )
    ineligible_assignment = float(assigned[:, ineligible].sum())
    metrics = result["metrics"]
    high_water = float(metrics["high_stress_direct_water_l"])
    cap_slack = float("nan") if cap_l is None else float(cap_l - high_water)
    allocation = assigned.sum(axis=0)
    output: dict[str, float | int | bool] = {
        "solver_stage_one_optimal": bool(result["solver_metadata"]["stage_one_success"]),
        "solver_stage_two_optimal": bool(result["solver_metadata"]["stage_two_success"]),
        "solver_stage_one_iterations": int(result["solver_metadata"]["stage_one_iterations"]),
        "solver_stage_two_iterations": int(result["solver_metadata"]["stage_two_iterations"]),
        "max_abs_demand_balance_error": float(np.abs(demand_balance).max()),
        "max_capacity_violation": float(np.maximum(-capacity_slack, 0.0).max()),
        "max_migration_violation": float(np.maximum(-migration_slack, 0.0).max()),
        "ineligible_latency_assignment": ineligible_assignment,
        "service_gap_vs_status_quo_unserved": float(abs(unserved.sum() - target_unserved)),
        "minimum_capacity_slack": float(capacity_slack.min()),
        "binding_capacity_region_hours": int(np.count_nonzero(np.abs(capacity_slack) <= TOLERANCE)),
        "minimum_migration_slack": float(migration_slack.min()),
        "binding_migration_hours": int(np.count_nonzero(np.abs(migration_slack) <= TOLERANCE)),
        "eligible_destination_count": int((~ineligible).sum() - 1),
        "high_stress_water_cap_slack_l": cap_slack,
        "high_stress_water_cap_violation_l": 0.0
        if cap_l is None
        else float(max(0.0, high_water - cap_l)),
        "assigned_east_accelerator_hours": float(allocation[origin]),
    }
    for idx, region in enumerate(case.regions):
        output[f"assigned_{region}_accelerator_hours"] = float(allocation[idx])
    return output


def frontier_record(
    case: CaseData,
    result: dict[str, object],
    *,
    status_quo: dict[str, object],
    carbon_first: dict[str, object],
    cap_fraction: float,
    cap_l: float | None,
    migration_limit: float,
    max_latency_ms: float,
    run_family: str,
) -> dict[str, object]:
    metrics = dict(result["metrics"])
    base = status_quo["metrics"]
    unbounded = carbon_first["metrics"]
    diagnostics = frontier_diagnostics(
        case,
        result,
        migration_limit=migration_limit,
        max_latency_ms=max_latency_ms,
        target_unserved=float(base["unserved_accelerator_hours"]),
        cap_l=cap_l,
    )
    high_water = float(metrics["high_stress_direct_water_l"])
    unbounded_high_water = float(unbounded["high_stress_direct_water_l"])
    record: dict[str, object] = {
        "run_family": run_family,
        "cap_fraction_of_carbon_first": cap_fraction,
        "high_stress_water_cap_l": cap_l,
        "carbon_first_high_stress_water_l": unbounded_high_water,
        "achieved_fraction_of_carbon_first_high_stress_water": (
            high_water / unbounded_high_water
            if unbounded_high_water > TOLERANCE
            else float("nan")
        ),
        "screened_direct_water_avoided_l_vs_carbon_first": (
            unbounded_high_water - high_water
        ),
        "additional_carbon_kgco2e_vs_carbon_first": (
            float(metrics["carbon_kgco2e"]) - float(unbounded["carbon_kgco2e"])
        ),
        "additional_carbon_pct_vs_carbon_first": percent_change(
            float(metrics["carbon_kgco2e"]), float(unbounded["carbon_kgco2e"])
        ),
        "carbon_change_pct_vs_status_quo": percent_change(
            float(metrics["carbon_kgco2e"]), float(base["carbon_kgco2e"])
        ),
        "service_difference_vs_status_quo_accelerator_hours": (
            float(metrics["served_accelerator_hours"])
            - float(base["served_accelerator_hours"])
        ),
        "status_quo_carbon_kgco2e": float(base["carbon_kgco2e"]),
        "status_quo_unserved_accelerator_hours": float(base["unserved_accelerator_hours"]),
        "carbon_first_carbon_kgco2e": float(unbounded["carbon_kgco2e"]),
        **metrics,
        **diagnostics,
    }
    return record


def solve_frontier(
    case: CaseData,
    *,
    migration_limit: float,
    max_latency_ms: float,
    cap_fractions: Iterable[float],
    run_family: str,
    status_quo: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status = status_quo or solve_status_quo(case)
    target_unserved = float(status["metrics"]["unserved_accelerator_hours"])
    carbon_first = solve_carbon_first_matched(
        case,
        migration_limit=migration_limit,
        max_latency_ms=max_latency_ms,
        baseline_unserved=target_unserved,
        policy="carbon_first_unconstrained",
    )
    carbon_first_high_water = float(
        carbon_first["metrics"]["high_stress_direct_water_l"]
    )
    records: list[dict[str, object]] = []
    dispatches: list[pd.DataFrame] = []
    for fraction in sorted({float(value) for value in cap_fractions}):
        cap_l = fraction * carbon_first_high_water
        result = solve_carbon_first_matched(
            case,
            migration_limit=migration_limit,
            max_latency_ms=max_latency_ms,
            baseline_unserved=target_unserved,
            policy=f"screened_water_cap_{fraction:.2f}",
            high_stress_water_cap_l=cap_l,
        )
        records.append(
            frontier_record(
                case,
                result,
                status_quo=status,
                carbon_first=carbon_first,
                cap_fraction=fraction,
                cap_l=cap_l,
                migration_limit=migration_limit,
                max_latency_ms=max_latency_ms,
                run_family=run_family,
            )
        )
        saved = result["dispatch"].copy()
        saved["run_family"] = run_family
        saved["cap_fraction_of_carbon_first"] = fraction
        saved["high_stress_water_cap_l"] = cap_l
        dispatches.append(saved)
    return pd.DataFrame(records), pd.concat(dispatches, ignore_index=True)


def add_segment_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        ordered = group.sort_values("cap_fraction_of_carbon_first", ascending=False)
        previous = None
        for _, row in ordered.iterrows():
            if previous is None:
                previous = row
                continue
            avoided_increment = (
                float(row["screened_direct_water_avoided_l_vs_carbon_first"])
                - float(previous["screened_direct_water_avoided_l_vs_carbon_first"])
            )
            carbon_increment = (
                float(row["additional_carbon_kgco2e_vs_carbon_first"])
                - float(previous["additional_carbon_kgco2e_vs_carbon_first"])
            )
            record = dict(zip(group_columns, key_values, strict=True))
            record.update(
                {
                    "from_cap_fraction": float(previous["cap_fraction_of_carbon_first"]),
                    "to_cap_fraction": float(row["cap_fraction_of_carbon_first"]),
                    "screened_direct_water_avoided_increment_l": avoided_increment,
                    "additional_carbon_increment_kgco2e": carbon_increment,
                    "marginal_additional_carbon_kgco2e_per_screened_l_avoided": (
                        carbon_increment / avoided_increment
                        if avoided_increment > TOLERANCE
                        else float("nan")
                    ),
                    "monotone_water_avoidance_pass": bool(avoided_increment >= -TOLERANCE),
                    "monotone_carbon_penalty_pass": bool(carbon_increment >= -TOLERANCE),
                }
            )
            records.append(record)
            previous = row
    return pd.DataFrame(records)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_reference(root: Path) -> None:
    case = load_case(root)
    frontier, dispatch = solve_frontier(
        case,
        migration_limit=0.30,
        max_latency_ms=20.0,
        cap_fractions=REFERENCE_CAP_FRACTIONS,
        run_family="reference_dense",
    )
    out = root / "03_results"
    frontier.to_csv(out / "ema_reference_frontier_v1.csv", index=False)
    dispatch.to_csv(out / "ema_reference_dispatch_v1.csv", index=False)
    segments = add_segment_metrics(frontier, ["run_family"])
    segments.to_csv(out / "ema_reference_frontier_segments_v1.csv", index=False)
    write_json(
        out / "ema_reference_summary_v1.json",
        {
            "status": "ANALYZED",
            "points": int(len(frontier)),
            "all_service_matched": bool(
                frontier["service_difference_vs_status_quo_accelerator_hours"].abs().max()
                <= TOLERANCE
            ),
            "maximum_feasibility_residual": float(
                frontier[
                    [
                        "max_abs_demand_balance_error",
                        "max_capacity_violation",
                        "max_migration_violation",
                        "ineligible_latency_assignment",
                        "service_gap_vs_status_quo_unserved",
                        "high_stress_water_cap_violation_l",
                    ]
                ].to_numpy(dtype=float).max()
            ),
        },
    )


def run_scenario_block(root: Path, pue_scenario: str) -> None:
    records: list[pd.DataFrame] = []
    for carbon_mapping in CARBON_MAPPINGS:
        for capacity_scenario in CAPACITY_SCENARIOS:
            case = load_case(
                root,
                carbon_mapping=carbon_mapping,
                pue_scenario=pue_scenario,
                capacity_scenario=capacity_scenario,
            )
            status = solve_status_quo(case)
            for migration_limit in MIGRATION_LIMITS:
                for max_latency_ms in LATENCY_LIMITS_MS:
                    frontier, _ = solve_frontier(
                        case,
                        migration_limit=migration_limit,
                        max_latency_ms=max_latency_ms,
                        cap_fractions=COARSE_CAP_FRACTIONS,
                        run_family="scenario_coarse",
                        status_quo=status,
                    )
                    records.append(frontier)
    result = pd.concat(records, ignore_index=True)
    out = root / "03_results"
    result.to_csv(out / f"ema_scenario_frontier_{pue_scenario}_v1.csv", index=False)
    write_json(
        out / f"ema_scenario_frontier_{pue_scenario}_summary_v1.json",
        {
            "status": "ANALYZED",
            "pue_scenario": pue_scenario,
            "points": int(len(result)),
            "expected_points": 405,
            "maximum_feasibility_residual": float(
                result[
                    [
                        "max_abs_demand_balance_error",
                        "max_capacity_violation",
                        "max_migration_violation",
                        "ineligible_latency_assignment",
                        "service_gap_vs_status_quo_unserved",
                        "high_stress_water_cap_violation_l",
                    ]
                ].to_numpy(dtype=float).max()
            ),
        },
    )


def run_wue(root: Path) -> None:
    wue_profiles = pd.read_csv(root / "01_data_processed" / "wue_profiles.csv")
    profiles = sorted(
        wue_profiles.loc[
            wue_profiles["wue_profile"].str.startswith("factorial_"), "wue_profile"
        ].unique()
    )
    records: list[pd.DataFrame] = []
    for profile in profiles:
        case = load_case(root, wue_profile=profile)
        frontier, _ = solve_frontier(
            case,
            migration_limit=0.30,
            max_latency_ms=20.0,
            cap_fractions=COARSE_CAP_FRACTIONS,
            run_family="spatial_wue_coarse",
        )
        records.append(frontier)
    result = pd.concat(records, ignore_index=True)
    out = root / "03_results"
    result.to_csv(out / "ema_spatial_wue_frontier_v1.csv", index=False)
    write_json(
        out / "ema_spatial_wue_frontier_summary_v1.json",
        {
            "status": "ANALYZED",
            "profiles": len(profiles),
            "points": int(len(result)),
            "expected_points": 405,
        },
    )


def run_energy(root: Path) -> None:
    records: list[pd.DataFrame] = []
    for energy_level in ENERGY_LEVELS:
        case = load_case(root, energy_level=energy_level)
        frontier, _ = solve_frontier(
            case,
            migration_limit=0.30,
            max_latency_ms=20.0,
            cap_fractions=COARSE_CAP_FRACTIONS,
            run_family="energy_scaling_check",
        )
        records.append(frontier)
    result = pd.concat(records, ignore_index=True)
    out = root / "03_results"
    result.to_csv(out / "ema_energy_scaling_frontier_v1.csv", index=False)


def aggregate(root: Path) -> None:
    out = root / "03_results"
    paths = [out / f"ema_scenario_frontier_{pue}_v1.csv" for pue in PUE_SCENARIOS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing scenario blocks: {missing}")
    scenario = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    scenario.to_csv(out / "ema_scenario_frontier_v1.csv", index=False)
    segments = add_segment_metrics(
        scenario,
        [
            "carbon_mapping",
            "pue_scenario",
            "capacity_scenario",
            "migration_limit",
            "max_latency_ms",
            "wue_profile",
            "water_risk_mapping",
            "energy_level",
        ],
    )
    segments.to_csv(out / "ema_scenario_frontier_segments_v1.csv", index=False)
    expected = 3 * 3 * 3 * 3 * 3 * len(COARSE_CAP_FRACTIONS)
    write_json(
        out / "ema_scenario_frontier_summary_v1.json",
        {
            "status": "ANALYZED",
            "points": int(len(scenario)),
            "expected_points": expected,
            "scenario_cells": int(len(scenario) / len(COARSE_CAP_FRACTIONS)),
            "all_solver_optimal": bool(
                scenario["solver_stage_one_optimal"].all()
                and scenario["solver_stage_two_optimal"].all()
            ),
            "all_service_matched": bool(
                scenario["service_difference_vs_status_quo_accelerator_hours"].abs().max()
                <= TOLERANCE
            ),
            "maximum_feasibility_residual": float(
                scenario[
                    [
                        "max_abs_demand_balance_error",
                        "max_capacity_violation",
                        "max_migration_violation",
                        "ineligible_latency_assignment",
                        "service_gap_vs_status_quo_unserved",
                        "high_stress_water_cap_violation_l",
                    ]
                ].to_numpy(dtype=float).max()
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("reference", "scenario", "wue", "energy", "aggregate"),
    )
    parser.add_argument("--pue-scenario", choices=PUE_SCENARIOS)
    args = parser.parse_args()
    if args.mode == "reference":
        run_reference(args.root)
    elif args.mode == "scenario":
        if args.pue_scenario is None:
            parser.error("--pue-scenario is required for scenario mode")
        run_scenario_block(args.root, args.pue_scenario)
    elif args.mode == "wue":
        run_wue(args.root)
    elif args.mode == "energy":
        run_energy(args.root)
    else:
        aggregate(args.root)


if __name__ == "__main__":
    main()
