from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_ema_environmental_frontier import (
    COARSE_CAP_FRACTIONS,
    PUE_SCENARIOS,
    REFERENCE_CAP_FRACTIONS,
    TOLERANCE,
    add_segment_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(results: list[dict[str, object]], name: str, passed: bool, detail: str) -> None:
    results.append({"check": name, "passed": bool(passed), "detail": detail})


def max_numeric(frame: pd.DataFrame, columns: list[str]) -> float:
    return float(np.nanmax(frame[columns].to_numpy(dtype=float)))


def verify_reference(root: Path, frozen_source: Path | None, results: list[dict[str, object]]) -> None:
    result_path = root / "03_results" / "ema_reference_frontier_v1.csv"
    frame = pd.read_csv(result_path)
    fractions = tuple(np.round(sorted(frame["cap_fraction_of_carbon_first"].unique()), 2))
    check(
        results,
        "reference_has_prespecified_21_point_grid",
        len(frame) == len(REFERENCE_CAP_FRACTIONS) and fractions == REFERENCE_CAP_FRACTIONS,
        f"points={len(frame)}, fractions={fractions}",
    )
    residual = max_numeric(
        frame,
        [
            "max_abs_demand_balance_error",
            "max_capacity_violation",
            "max_migration_violation",
            "ineligible_latency_assignment",
            "service_gap_vs_status_quo_unserved",
            "high_stress_water_cap_violation_l",
        ],
    )
    check(
        results,
        "reference_feasibility_residual",
        residual <= TOLERANCE * 2.0,
        f"maximum_residual={residual:.12g}",
    )
    check(
        results,
        "reference_solver_optimal",
        bool(frame["solver_stage_one_optimal"].all() and frame["solver_stage_two_optimal"].all()),
        f"stage_one={frame['solver_stage_one_optimal'].all()}, stage_two={frame['solver_stage_two_optimal'].all()}",
    )
    segments = add_segment_metrics(frame, ["run_family"])
    check(
        results,
        "reference_frontier_monotone",
        bool(
            segments["monotone_water_avoidance_pass"].all()
            and segments["monotone_carbon_penalty_pass"].all()
        ),
        f"segments={len(segments)}",
    )

    if frozen_source is not None:
        old_path = frozen_source / "04_results" / "high_stress_water_pareto_v1.csv"
        old = pd.read_csv(old_path)
        expected = old[[
            "cap_fraction_of_carbon_first",
            "carbon_kgco2e",
            "direct_water_l",
            "high_stress_direct_water_l",
        ]].copy()
        observed = frame[[
            "cap_fraction_of_carbon_first",
            "carbon_kgco2e",
            "direct_water_l",
            "high_stress_direct_water_l",
        ]].copy()
        merged = expected.merge(observed, on="cap_fraction_of_carbon_first", suffixes=("_frozen", "_new"))
        differences = []
        for column in ("carbon_kgco2e", "direct_water_l", "high_stress_direct_water_l"):
            differences.append(
                np.abs(merged[f"{column}_frozen"] - merged[f"{column}_new"]).max()
            )
        maximum_difference = float(max(differences))
        check(
            results,
            "reference_endpoints_match_frozen_five_point_results",
            len(merged) == 5 and maximum_difference <= 1e-6,
            f"matched_points={len(merged)}, maximum_absolute_difference={maximum_difference:.12g}",
        )


def verify_scenario(root: Path, results: list[dict[str, object]]) -> None:
    frame = pd.read_csv(root / "03_results" / "ema_scenario_frontier_v1.csv")
    expected_points = 3 * 3 * 3 * 3 * 3 * len(COARSE_CAP_FRACTIONS)
    group_columns = [
        "carbon_mapping",
        "pue_scenario",
        "capacity_scenario",
        "migration_limit",
        "max_latency_ms",
        "wue_profile",
        "water_risk_mapping",
        "energy_level",
    ]
    group_sizes = frame.groupby(group_columns, dropna=False).size()
    check(
        results,
        "scenario_frontier_complete",
        len(frame) == expected_points
        and len(group_sizes) == 243
        and bool((group_sizes == len(COARSE_CAP_FRACTIONS)).all()),
        f"points={len(frame)}, cells={len(group_sizes)}, expected_points={expected_points}",
    )
    check(
        results,
        "scenario_uses_all_three_pue_blocks",
        set(frame["pue_scenario"].unique()) == set(PUE_SCENARIOS),
        f"pue={sorted(frame['pue_scenario'].unique())}",
    )
    residual = max_numeric(
        frame,
        [
            "max_abs_demand_balance_error",
            "max_capacity_violation",
            "max_migration_violation",
            "ineligible_latency_assignment",
            "service_gap_vs_status_quo_unserved",
            "high_stress_water_cap_violation_l",
        ],
    )
    check(
        results,
        "scenario_feasibility_residual",
        residual <= TOLERANCE * 2.0,
        f"maximum_residual={residual:.12g}",
    )
    check(
        results,
        "scenario_solver_optimal",
        bool(frame["solver_stage_one_optimal"].all() and frame["solver_stage_two_optimal"].all()),
        f"stage_one={frame['solver_stage_one_optimal'].all()}, stage_two={frame['solver_stage_two_optimal'].all()}",
    )
    check(
        results,
        "scenario_carbon_not_above_status_quo",
        bool((frame["carbon_change_pct_vs_status_quo"] <= TOLERANCE).all()),
        f"maximum_change_pct={frame['carbon_change_pct_vs_status_quo'].max():.12g}",
    )
    segments = add_segment_metrics(frame, group_columns)
    check(
        results,
        "scenario_frontier_monotone",
        len(segments) == 243 * 4
        and bool(segments["monotone_water_avoidance_pass"].all())
        and bool(segments["monotone_carbon_penalty_pass"].all()),
        f"segments={len(segments)}",
    )


def verify_wue(root: Path, results: list[dict[str, object]]) -> None:
    frame = pd.read_csv(root / "03_results" / "ema_spatial_wue_frontier_v1.csv")
    group_sizes = frame.groupby("wue_profile").size()
    residual = max_numeric(
        frame,
        [
            "max_abs_demand_balance_error",
            "max_capacity_violation",
            "max_migration_violation",
            "ineligible_latency_assignment",
            "service_gap_vs_status_quo_unserved",
            "high_stress_water_cap_violation_l",
        ],
    )
    check(
        results,
        "spatial_wue_frontier_complete",
        len(frame) == 405
        and len(group_sizes) == 81
        and bool((group_sizes == len(COARSE_CAP_FRACTIONS)).all()),
        f"points={len(frame)}, profiles={len(group_sizes)}",
    )
    check(
        results,
        "spatial_wue_frontier_feasible_and_optimal",
        residual <= TOLERANCE * 2.0
        and bool(frame["solver_stage_one_optimal"].all())
        and bool(frame["solver_stage_two_optimal"].all()),
        f"maximum_residual={residual:.12g}",
    )


def verify_energy(root: Path, results: list[dict[str, object]]) -> None:
    frame = pd.read_csv(root / "03_results" / "ema_energy_scaling_frontier_v1.csv")
    check(
        results,
        "energy_scaling_frontier_complete",
        len(frame) == 15 and set(frame["energy_level"].unique()) == {"low", "central", "high"},
        f"points={len(frame)}, levels={sorted(frame['energy_level'].unique())}",
    )
    central = frame.loc[frame["energy_level"].eq("central")].set_index("cap_fraction_of_carbon_first")
    columns = [
        "assigned_east_accelerator_hours",
        "assigned_north_accelerator_hours",
        "assigned_northwest_accelerator_hours",
        "assigned_southwest_accelerator_hours",
        "carbon_change_pct_vs_status_quo",
        "achieved_fraction_of_carbon_first_high_stress_water",
    ]
    maximum_difference = 0.0
    for energy_level in ("low", "high"):
        compare = frame.loc[frame["energy_level"].eq(energy_level)].set_index("cap_fraction_of_carbon_first")
        maximum_difference = max(
            maximum_difference,
            float(np.nanmax(np.abs(compare[columns].to_numpy() - central[columns].to_numpy()))),
        )
    check(
        results,
        "energy_scaling_allocation_and_relative_results_invariant",
        maximum_difference <= 1e-6,
        f"maximum_absolute_difference={maximum_difference:.12g}",
    )


def verify_copied_inputs(root: Path, frozen_source: Path, results: list[dict[str, object]]) -> None:
    source_data = frozen_source / "01_data_processed"
    target_data = root / "01_data_processed"
    source_files = {path.relative_to(source_data).as_posix(): path for path in source_data.rglob("*") if path.is_file()}
    target_files = {path.relative_to(target_data).as_posix(): path for path in target_data.rglob("*") if path.is_file()}
    same_names = set(source_files) == set(target_files)
    mismatches = [
        relative
        for relative in sorted(set(source_files) & set(target_files))
        if sha256(source_files[relative]) != sha256(target_files[relative])
    ]
    check(
        results,
        "copied_processed_inputs_match_frozen_source",
        same_names and not mismatches,
        f"source_files={len(source_files)}, target_files={len(target_files)}, mismatches={mismatches}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--frozen-source",
        type=Path,
        help="Optional frozen predecessor used only for provenance/hash and five-point reproduction checks.",
    )
    args = parser.parse_args()
    results: list[dict[str, object]] = []
    if args.frozen_source is not None:
        verify_copied_inputs(args.root, args.frozen_source, results)
    verify_reference(args.root, args.frozen_source, results)
    verify_scenario(args.root, results)
    verify_wue(args.root, results)
    verify_energy(args.root, results)
    payload = {
        "status": "PASS" if all(row["passed"] for row in results) else "FAIL",
        "checks": results,
    }
    qa_dir = args.root / "06_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "ema_environmental_frontier_qa_v1.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(qa_dir / "ema_environmental_frontier_qa_v1.csv", index=False)
    if payload["status"] != "PASS":
        raise SystemExit("EMA environmental frontier QA failed")


if __name__ == "__main__":
    main()
