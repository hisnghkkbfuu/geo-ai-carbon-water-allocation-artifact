from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix, vstack


REGION_ORDER = ["east", "north", "northwest", "southwest"]
ORIGIN = "east"
TOLERANCE = 1e-8


@dataclass(frozen=True)
class CaseData:
    hour_index: np.ndarray
    timestamps: np.ndarray
    regions: tuple[str, ...]
    demand_accelerator_hours: np.ndarray
    capacity_accelerator_hours: np.ndarray
    latency_ms: np.ndarray
    pue: np.ndarray
    carbon_intensity_kgco2e_per_kwh: np.ndarray
    wue_l_per_kwh_it: np.ndarray
    high_water_stress: np.ndarray
    water_stress_category: tuple[str, ...]
    it_energy_kwh_per_accelerator_hour: float
    metadata: dict[str, str]


def load_case(
    root: Path,
    *,
    carbon_mapping: str = "equal_site_portfolio",
    pue_scenario: str = "hub_policy_compliant",
    capacity_scenario: str = "central",
    wue_profile: str = "uniform_google_ai_2024",
    water_risk_mapping: str = "portfolio_mean",
    energy_level: str = "central",
    demand_filename: str = "gentd26_hourly_service_demand.csv",
    carbon_filename: str = "gentd26_aligned_regional_hourly_cef.csv",
    capacity_filename: str = "capacity_scenarios.csv",
    energy_filename: str = "energy_calibration_bounds.csv",
    timestamp_column: str = "trace_timestamp_anonymized",
    energy_workload_class: str | None = None,
) -> CaseData:
    data = root / "01_data_processed"
    demand = pd.read_csv(data / demand_filename)
    demand = demand.sort_values("hour_index")
    hours = demand["hour_index"].astype(int).to_numpy()
    timestamps = pd.to_datetime(demand[timestamp_column]).to_numpy()

    carbon = pd.read_csv(data / carbon_filename)
    carbon = carbon[carbon["carbon_mapping_scenario"].eq(carbon_mapping)].copy()
    carbon_pivot = carbon.pivot(
        index="hour_index",
        columns="region",
        values="carbon_intensity_kgco2e_per_kwh",
    ).reindex(index=hours, columns=REGION_ORDER)
    if carbon_pivot.isna().any().any():
        raise ValueError(f"Incomplete carbon mapping: {carbon_mapping}")

    pue_table = pd.read_csv(data / "regional_pue_latency_scenarios.csv")
    pue_table = pue_table[pue_table["pue_scenario"].eq(pue_scenario)].set_index(
        "region"
    ).reindex(REGION_ORDER)
    if pue_table[["pue", "latency_from_east_ms"]].isna().any().any():
        raise ValueError(f"Incomplete PUE/latency scenario: {pue_scenario}")

    capacity = pd.read_csv(data / capacity_filename)
    capacity = capacity[capacity["capacity_scenario"].eq(capacity_scenario)]
    capacity_pivot = capacity.pivot(
        index="hour_index", columns="region", values="capacity_accelerator_hours"
    ).reindex(index=hours, columns=REGION_ORDER)
    if capacity_pivot.isna().any().any():
        raise ValueError(f"Incomplete capacity scenario: {capacity_scenario}")

    wue = pd.read_csv(data / "wue_profiles.csv")
    wue = wue[wue["wue_profile"].eq(wue_profile)].set_index("region").reindex(
        REGION_ORDER
    )
    if wue["wue_l_per_kwh_it"].isna().any():
        raise ValueError(f"Incomplete WUE profile: {wue_profile}")

    risk = pd.read_csv(data / "regional_water_risk_screening.csv")
    risk = risk[risk["water_risk_mapping"].eq(water_risk_mapping)].set_index(
        "region"
    ).reindex(REGION_ORDER)
    if risk["aqueduct_bws_category"].isna().any():
        raise ValueError(f"Incomplete water-risk mapping: {water_risk_mapping}")

    energy_table = pd.read_csv(data / energy_filename)
    if energy_workload_class is not None:
        energy_table = energy_table[
            energy_table["model_workload_class"].eq(energy_workload_class)
        ]
    if len(energy_table) != 1:
        raise ValueError(
            "Energy calibration selection must contain exactly one row; "
            f"found {len(energy_table)}"
        )
    energy = energy_table.iloc[0]
    energy_column = f"it_energy_{energy_level}_kwh_per_accelerator_hour"
    if energy_column not in energy:
        raise ValueError(f"Unknown energy level: {energy_level}")

    return CaseData(
        hour_index=hours,
        timestamps=timestamps,
        regions=tuple(REGION_ORDER),
        demand_accelerator_hours=demand["service_accelerator_hours"].astype(float).to_numpy(),
        capacity_accelerator_hours=capacity_pivot.astype(float).to_numpy(),
        latency_ms=pue_table["latency_from_east_ms"].astype(float).to_numpy(),
        pue=pue_table["pue"].astype(float).to_numpy(),
        carbon_intensity_kgco2e_per_kwh=carbon_pivot.astype(float).to_numpy(),
        wue_l_per_kwh_it=wue["wue_l_per_kwh_it"].astype(float).to_numpy(),
        high_water_stress=risk["high_or_extremely_high"].astype(str).str.lower().eq("true").to_numpy(),
        water_stress_category=tuple(risk["aqueduct_bws_category"].astype(str)),
        it_energy_kwh_per_accelerator_hour=float(energy[energy_column]),
        metadata={
            "carbon_mapping": carbon_mapping,
            "pue_scenario": pue_scenario,
            "capacity_scenario": capacity_scenario,
            "wue_profile": wue_profile,
            "water_risk_mapping": water_risk_mapping,
            "energy_level": energy_level,
            "demand_filename": demand_filename,
            "carbon_filename": carbon_filename,
            "capacity_filename": capacity_filename,
            "energy_filename": energy_filename,
            "timestamp_column": timestamp_column,
            "energy_workload_class": energy_workload_class or "single_row_default",
        },
    )


def physical_coefficients(case: CaseData) -> dict[str, np.ndarray]:
    n_hours = len(case.hour_index)
    it = np.full(
        (n_hours, len(case.regions)),
        case.it_energy_kwh_per_accelerator_hour,
        dtype=float,
    )
    facility = it * case.pue.reshape(1, -1)
    carbon = facility * case.carbon_intensity_kgco2e_per_kwh
    direct_water = it * case.wue_l_per_kwh_it.reshape(1, -1)
    high_stress_water = direct_water * case.high_water_stress.reshape(1, -1)
    return {
        "it_energy_kwh_per_accelerator_hour": it,
        "facility_energy_kwh_per_accelerator_hour": facility,
        "carbon_kgco2e_per_accelerator_hour": carbon,
        "direct_water_l_per_accelerator_hour": direct_water,
        "high_stress_water_l_per_accelerator_hour": high_stress_water,
    }


def _matrices(
    case: CaseData,
    migration_share: float,
) -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, list[tuple[float, float | None]]]:
    n_hours = len(case.hour_index)
    n_regions = len(case.regions)
    n_x = n_hours * n_regions
    n_variables = n_x + n_hours

    a_eq = lil_matrix((n_hours, n_variables), dtype=float)
    for t in range(n_hours):
        a_eq[t, t * n_regions : (t + 1) * n_regions] = 1.0
        a_eq[t, n_x + t] = 1.0
    b_eq = case.demand_accelerator_hours.astype(float).copy()

    origin_index = case.regions.index(ORIGIN)
    a_ub = lil_matrix((n_hours, n_variables), dtype=float)
    for t in range(n_hours):
        for r in range(n_regions):
            if r != origin_index:
                a_ub[t, t * n_regions + r] = 1.0
    b_ub = migration_share * case.demand_accelerator_hours.astype(float)

    bounds: list[tuple[float, float | None]] = []
    for t in range(n_hours):
        for r in range(n_regions):
            bounds.append((0.0, float(case.capacity_accelerator_hours[t, r])))
    bounds.extend([(0.0, None)] * n_hours)
    return a_eq.tocsr(), b_eq, a_ub.tocsr(), b_ub, bounds


def solve_lexicographic(
    case: CaseData,
    *,
    policy: str,
    migration_share: float,
    max_latency_ms: float,
    second_stage_objective: str = "carbon",
    direct_water_cap_l: float | None = None,
    high_stress_water_cap_l: float | None = None,
    carbon_cap_kgco2e: float | None = None,
    fixed_total_unserved_accelerator_hours: float | None = None,
    service_tolerance: float = TOLERANCE,
) -> dict[str, object]:
    if not 0.0 <= migration_share <= 1.0:
        raise ValueError("migration_share must be in [0, 1]")
    coefficients = physical_coefficients(case)
    n_hours = len(case.hour_index)
    n_regions = len(case.regions)
    n_x = n_hours * n_regions
    n_variables = n_x + n_hours
    a_eq, b_eq, a_ub, b_ub, bounds = _matrices(case, migration_share)

    origin_index = case.regions.index(ORIGIN)
    for t in range(n_hours):
        for r in range(n_regions):
            if r != origin_index and case.latency_ms[r] > max_latency_ms:
                bounds[t * n_regions + r] = (0.0, 0.0)

    stage_one_cost = np.zeros(n_variables, dtype=float)
    stage_one_cost[n_x:] = 1.0
    stage_one = linprog(
        stage_one_cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not stage_one.success:
        raise RuntimeError(f"{policy} stage one failed: {stage_one.message}")

    objective_map = {
        "carbon": coefficients["carbon_kgco2e_per_accelerator_hour"],
        "direct_water": coefficients["direct_water_l_per_accelerator_hour"],
        "high_stress_water": coefficients[
            "high_stress_water_l_per_accelerator_hour"
        ],
    }
    if second_stage_objective not in objective_map:
        raise ValueError(f"Unknown second-stage objective: {second_stage_objective}")
    stage_two_cost = np.zeros(n_variables, dtype=float)
    stage_two_cost[:n_x] = objective_map[second_stage_objective].reshape(-1)

    extra_rows: list[csr_matrix] = []
    extra_bounds: list[float] = []
    service_row = np.zeros(n_variables, dtype=float)
    service_row[n_x:] = 1.0
    if fixed_total_unserved_accelerator_hours is None:
        extra_rows.append(csr_matrix(service_row.reshape(1, -1)))
        extra_bounds.append(float(stage_one.fun) + service_tolerance)
    else:
        target = float(fixed_total_unserved_accelerator_hours)
        if target + service_tolerance < float(stage_one.fun):
            raise ValueError(
                "Fixed unserved target is below the maximum-service optimum"
            )
        extra_rows.append(csr_matrix(service_row.reshape(1, -1)))
        extra_bounds.append(target + service_tolerance)
        extra_rows.append(csr_matrix((-service_row).reshape(1, -1)))
        extra_bounds.append(-target + service_tolerance)

    caps = [
        (direct_water_cap_l, "direct_water_l_per_accelerator_hour"),
        (high_stress_water_cap_l, "high_stress_water_l_per_accelerator_hour"),
        (carbon_cap_kgco2e, "carbon_kgco2e_per_accelerator_hour"),
    ]
    for cap, key in caps:
        if cap is None:
            continue
        row = np.zeros(n_variables, dtype=float)
        row[:n_x] = coefficients[key].reshape(-1)
        extra_rows.append(csr_matrix(row.reshape(1, -1)))
        extra_bounds.append(float(cap))

    a_ub_stage_two = vstack([a_ub, *extra_rows], format="csr")
    b_ub_stage_two = np.concatenate([b_ub, np.asarray(extra_bounds)])
    stage_two = linprog(
        stage_two_cost,
        A_ub=a_ub_stage_two,
        b_ub=b_ub_stage_two,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not stage_two.success:
        raise RuntimeError(f"{policy} stage two failed: {stage_two.message}")

    x = stage_two.x[:n_x].reshape(n_hours, n_regions)
    unmet = stage_two.x[n_x:]
    rows: list[dict[str, object]] = []
    for t in range(n_hours):
        for r, region in enumerate(case.regions):
            assigned = float(x[t, r])
            it_energy = assigned * coefficients[
                "it_energy_kwh_per_accelerator_hour"
            ][t, r]
            facility_energy = assigned * coefficients[
                "facility_energy_kwh_per_accelerator_hour"
            ][t, r]
            carbon = assigned * coefficients[
                "carbon_kgco2e_per_accelerator_hour"
            ][t, r]
            water = assigned * coefficients[
                "direct_water_l_per_accelerator_hour"
            ][t, r]
            high_water = assigned * coefficients[
                "high_stress_water_l_per_accelerator_hour"
            ][t, r]
            rows.append(
                {
                    "policy": policy,
                    "hour_index": int(case.hour_index[t]),
                    "trace_timestamp_anonymized": pd.Timestamp(case.timestamps[t]),
                    "region": region,
                    "assigned_accelerator_hours": assigned,
                    "it_energy_kwh": float(it_energy),
                    "facility_energy_kwh": float(facility_energy),
                    "carbon_kgco2e": float(carbon),
                    "direct_water_l": float(water),
                    "high_stress_direct_water_l": float(high_water),
                    "water_stress_category": case.water_stress_category[r],
                    "latency_from_east_ms": float(case.latency_ms[r]),
                    "migrated": region != ORIGIN,
                }
            )
    dispatch = pd.DataFrame(rows)
    total_demand = float(case.demand_accelerator_hours.sum())
    total_served = float(x.sum())
    total_water = float(dispatch["direct_water_l"].sum())
    high_water = float(dispatch["high_stress_direct_water_l"].sum())
    migrated = float(
        dispatch.loc[dispatch["migrated"], "assigned_accelerator_hours"].sum()
    )
    metrics = {
        "policy": policy,
        **case.metadata,
        "migration_limit": migration_share,
        "max_latency_ms": max_latency_ms,
        "second_stage_objective": second_stage_objective,
        "demand_accelerator_hours": total_demand,
        "served_accelerator_hours": total_served,
        "unserved_accelerator_hours": float(unmet.sum()),
        "service_rate": total_served / total_demand if total_demand else np.nan,
        "migration_share": migrated / total_served if total_served else np.nan,
        "it_energy_kwh": float(dispatch["it_energy_kwh"].sum()),
        "facility_energy_kwh": float(dispatch["facility_energy_kwh"].sum()),
        "carbon_kgco2e": float(dispatch["carbon_kgco2e"].sum()),
        "direct_water_l": total_water,
        "high_stress_direct_water_l": high_water,
        "high_stress_water_share": high_water / total_water if total_water else np.nan,
        "stage_one_optimal_unserved_accelerator_hours": float(stage_one.fun),
        "direct_water_cap_l": direct_water_cap_l,
        "high_stress_water_cap_l": high_stress_water_cap_l,
        "carbon_cap_kgco2e": carbon_cap_kgco2e,
        "fixed_total_unserved_accelerator_hours": (
            fixed_total_unserved_accelerator_hours
        ),
    }
    solver_metadata = {
        "stage_one_success": bool(stage_one.success),
        "stage_one_status": int(stage_one.status),
        "stage_one_message": str(stage_one.message),
        "stage_one_objective": float(stage_one.fun),
        "stage_one_iterations": int(stage_one.nit),
        "stage_two_success": bool(stage_two.success),
        "stage_two_status": int(stage_two.status),
        "stage_two_message": str(stage_two.message),
        "stage_two_objective": float(stage_two.fun),
        "stage_two_iterations": int(stage_two.nit),
    }
    return {
        "dispatch": dispatch,
        "unserved": pd.DataFrame(
            {
                "policy": policy,
                "hour_index": case.hour_index,
                "unserved_accelerator_hours": unmet,
            }
        ),
        "metrics": metrics,
        "solver_metadata": solver_metadata,
        "raw_solution": stage_two.x,
    }


def reverse_regional_carbon(case: CaseData) -> CaseData:
    return replace(
        case,
        carbon_intensity_kgco2e_per_kwh=np.flip(
            case.carbon_intensity_kgco2e_per_kwh, axis=1
        ).copy(),
        metadata={**case.metadata, "carbon_mapping": "artificial_region_reversal"},
    )


def revalue_water(
    dispatch: pd.DataFrame,
    wue_by_region: dict[str, float],
    high_stress_by_region: dict[str, bool],
) -> dict[str, float]:
    frame = dispatch.copy()
    frame["wue"] = frame["region"].map(wue_by_region).astype(float)
    frame["high"] = frame["region"].map(high_stress_by_region).astype(bool)
    frame["water"] = frame["it_energy_kwh"] * frame["wue"]
    frame["high_water"] = frame["water"] * frame["high"].astype(float)
    total = float(frame["water"].sum())
    high = float(frame["high_water"].sum())
    return {
        "direct_water_l": total,
        "high_stress_direct_water_l": high,
        "high_stress_water_share": high / total if total else np.nan,
    }
