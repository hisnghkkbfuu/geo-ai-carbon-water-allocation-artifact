from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_code"))

from cws_lexicographic_model import (  # noqa: E402
    CaseData,
    load_case,
    physical_coefficients,
    reverse_regional_carbon,
    solve_lexicographic,
)
from run_stage1_reconstruction_experiments import matched_hourly_carbon  # noqa: E402


def synthetic_case() -> CaseData:
    return CaseData(
        hour_index=np.array([0, 1]),
        timestamps=pd.date_range("2025-01-01 00:00:00", periods=2, freq="h").to_numpy(),
        regions=("east", "north", "northwest", "southwest"),
        demand_accelerator_hours=np.array([10.0, 10.0]),
        capacity_accelerator_hours=np.array(
            [[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]]
        ),
        latency_ms=np.array([2.0, 10.0, 18.0, 17.0]),
        pue=np.array([1.25, 1.20, 1.20, 1.20]),
        carbon_intensity_kgco2e_per_kwh=np.array(
            [[0.8, 0.2, 0.4, 0.6], [0.8, 0.2, 0.4, 0.6]]
        ),
        wue_l_per_kwh_it=np.array([1.0, 2.0, 1.5, 0.5]),
        high_water_stress=np.array([False, True, True, False]),
        water_stress_category=("medium_high", "extremely_high", "high", "low_medium"),
        it_energy_kwh_per_accelerator_hour=0.5,
        metadata={"case": "synthetic"},
    )


class PhysicalAccountingTests(unittest.TestCase):
    def test_hand_calculated_unit_chain(self) -> None:
        case = synthetic_case()
        coeff = physical_coefficients(case)
        self.assertAlmostEqual(coeff["it_energy_kwh_per_accelerator_hour"][0, 0], 0.5)
        self.assertAlmostEqual(
            coeff["facility_energy_kwh_per_accelerator_hour"][0, 0], 0.625
        )
        self.assertAlmostEqual(
            coeff["carbon_kgco2e_per_accelerator_hour"][0, 0], 0.5
        )

    def test_wue_uses_it_energy_not_facility_energy(self) -> None:
        case = synthetic_case()
        coeff = physical_coefficients(case)
        self.assertAlmostEqual(
            coeff["direct_water_l_per_accelerator_hour"][0, 0], 0.5
        )
        changed_pue = replace(case, pue=np.array([2.0, 2.0, 2.0, 2.0]))
        changed = physical_coefficients(changed_pue)
        np.testing.assert_allclose(
            coeff["direct_water_l_per_accelerator_hour"],
            changed["direct_water_l_per_accelerator_hour"],
        )

    def test_water_stress_is_a_screen_not_multiplier(self) -> None:
        case = synthetic_case()
        coeff = physical_coefficients(case)
        self.assertAlmostEqual(
            coeff["high_stress_water_l_per_accelerator_hour"][0, 1], 1.0
        )
        self.assertAlmostEqual(
            coeff["high_stress_water_l_per_accelerator_hour"][0, 0], 0.0
        )


class OptimizationTests(unittest.TestCase):
    def test_status_quo_has_exactly_zero_migration(self) -> None:
        result = solve_lexicographic(
            synthetic_case(),
            policy="status_quo",
            migration_share=0.0,
            max_latency_ms=20.0,
        )
        self.assertAlmostEqual(result["metrics"]["migration_share"], 0.0)
        self.assertAlmostEqual(result["metrics"]["service_rate"], 1.0)

    def test_demand_conservation(self) -> None:
        case = synthetic_case()
        result = solve_lexicographic(
            case,
            policy="carbon",
            migration_share=0.3,
            max_latency_ms=20.0,
        )
        served = result["dispatch"].groupby("hour_index")[
            "assigned_accelerator_hours"
        ].sum()
        unmet = result["unserved"].set_index("hour_index")[
            "unserved_accelerator_hours"
        ]
        np.testing.assert_allclose(
            served.reindex(case.hour_index).to_numpy()
            + unmet.reindex(case.hour_index).to_numpy(),
            case.demand_accelerator_hours,
            atol=1e-7,
        )

    def test_migration_limit_is_enforced_per_hour(self) -> None:
        case = synthetic_case()
        result = solve_lexicographic(
            case,
            policy="carbon",
            migration_share=0.3,
            max_latency_ms=20.0,
        )
        migrated = (
            result["dispatch"]
            .query("migrated")
            .groupby("hour_index")["assigned_accelerator_hours"]
            .sum()
            .reindex(case.hour_index, fill_value=0.0)
        )
        np.testing.assert_array_less(
            migrated.to_numpy(), 0.3 * case.demand_accelerator_hours + 1e-7
        )

    def test_latency_ineligible_region_receives_zero(self) -> None:
        result = solve_lexicographic(
            synthetic_case(),
            policy="strict",
            migration_share=1.0,
            max_latency_ms=15.0,
        )
        ineligible = result["dispatch"][
            result["dispatch"]["region"].isin(["northwest", "southwest"])
        ]
        self.assertAlmostEqual(ineligible["assigned_accelerator_hours"].sum(), 0.0)

    def test_service_is_lexicographically_preserved(self) -> None:
        case = synthetic_case()
        expensive = replace(
            case,
            carbon_intensity_kgco2e_per_kwh=np.full((2, 4), 1e9),
        )
        result = solve_lexicographic(
            expensive,
            policy="expensive",
            migration_share=0.3,
            max_latency_ms=20.0,
        )
        self.assertAlmostEqual(result["metrics"]["service_rate"], 1.0, places=8)

    def test_high_stress_water_cap_is_enforced(self) -> None:
        result = solve_lexicographic(
            synthetic_case(),
            policy="risk_screened",
            migration_share=0.5,
            max_latency_ms=20.0,
            high_stress_water_cap_l=0.0,
        )
        self.assertLessEqual(
            result["metrics"]["high_stress_direct_water_l"], 1e-7
        )
        self.assertAlmostEqual(result["metrics"]["service_rate"], 1.0, places=8)

    def test_carbon_reversal_changes_allocation(self) -> None:
        case = synthetic_case()
        original = solve_lexicographic(
            case,
            policy="original",
            migration_share=0.5,
            max_latency_ms=20.0,
        )["dispatch"]
        reversed_dispatch = solve_lexicographic(
            reverse_regional_carbon(case),
            policy="reversed",
            migration_share=0.5,
            max_latency_ms=20.0,
        )["dispatch"]
        a = original.groupby("region")["assigned_accelerator_hours"].sum()
        b = reversed_dispatch.groupby("region")["assigned_accelerator_hours"].sum()
        self.assertGreater(a.sub(b, fill_value=0.0).abs().sum(), 1e-6)

    def test_fixed_service_counterfactual_matches_baseline_service(self) -> None:
        case = synthetic_case()
        capacity = case.capacity_accelerator_hours.copy()
        capacity[:, 0] = 5.0
        constrained = replace(case, capacity_accelerator_hours=capacity)
        baseline = solve_lexicographic(
            constrained,
            policy="baseline",
            migration_share=0.0,
            max_latency_ms=20.0,
        )["metrics"]
        matched = solve_lexicographic(
            constrained,
            policy="matched",
            migration_share=0.5,
            max_latency_ms=20.0,
            fixed_total_unserved_accelerator_hours=float(
                baseline["unserved_accelerator_hours"]
            ),
        )["metrics"]
        self.assertAlmostEqual(
            baseline["served_accelerator_hours"],
            matched["served_accelerator_hours"],
            places=7,
        )
        self.assertLessEqual(matched["carbon_kgco2e"], baseline["carbon_kgco2e"] + 1e-7)


class IntegrationInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_case(ROOT)

    def test_full_timestamp_not_collapsed_to_24_hours(self) -> None:
        self.assertEqual(len(self.case.hour_index), 554)
        span = pd.Timestamp(self.case.timestamps[-1]) - pd.Timestamp(self.case.timestamps[0])
        self.assertGreater(span.total_seconds(), 22 * 86400)

    def test_central_status_quo_serves_all_observed_equivalent_work(self) -> None:
        result = solve_lexicographic(
            self.case,
            policy="status_quo",
            migration_share=0.0,
            max_latency_ms=20.0,
        )
        self.assertAlmostEqual(result["metrics"]["service_rate"], 1.0, places=8)
        self.assertAlmostEqual(result["metrics"]["migration_share"], 0.0, places=12)

    def test_uniform_wue_keeps_total_water_constant_at_equal_service(self) -> None:
        base = solve_lexicographic(
            self.case,
            policy="status_quo",
            migration_share=0.0,
            max_latency_ms=20.0,
        )["metrics"]
        treatment = solve_lexicographic(
            self.case,
            policy="carbon",
            migration_share=0.3,
            max_latency_ms=20.0,
        )["metrics"]
        self.assertAlmostEqual(
            base["direct_water_l"], treatment["direct_water_l"], places=8
        )

    def test_all_positive_execution_sensitivity_is_not_smaller_than_successes(self) -> None:
        all_executed = load_case(
            ROOT,
            demand_filename="gentd26_hourly_all_positive_execution_demand.csv",
        )
        self.assertGreater(
            all_executed.demand_accelerator_hours.sum(),
            self.case.demand_accelerator_hours.sum(),
        )

    def test_matched_hourly_accounting_uses_common_service(self) -> None:
        baseline = pd.DataFrame(
            {
                "hour_index": [0, 1],
                "assigned_accelerator_hours": [10.0, 10.0],
                "carbon_kgco2e": [10.0, 10.0],
            }
        )
        treatment = pd.DataFrame(
            {
                "hour_index": [0, 1],
                "assigned_accelerator_hours": [5.0, 10.0],
                "carbon_kgco2e": [2.5, 5.0],
            }
        )
        result = matched_hourly_carbon(baseline, treatment)
        self.assertAlmostEqual(result["common_served_accelerator_hours"], 15.0)
        self.assertAlmostEqual(result["matched_baseline_carbon_kgco2e"], 15.0)
        self.assertAlmostEqual(result["matched_treatment_carbon_kgco2e"], 7.5)
        self.assertAlmostEqual(result["matched_carbon_change_pct"], -50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
