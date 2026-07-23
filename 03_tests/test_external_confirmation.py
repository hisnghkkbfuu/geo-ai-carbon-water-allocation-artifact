from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_code"))

from build_external_confirmation_inputs import (  # noqa: E402
    TP_SIZE,
    process_azure_trace,
    read_vidur_lookup,
)
from generate_vidur_calibration_grid import (  # noqa: E402
    ARRIVAL_GAP_SECONDS,
    MAX_TOKENS,
    TOKEN_GRID,
    build_grid,
)
from build_external_mapping_sensitivities import map_execution_times  # noqa: E402
from cws_lexicographic_model import load_case  # noqa: E402


class VidurGridTests(unittest.TestCase):
    def test_formal_grid_is_complete_ordered_and_isolated(self) -> None:
        grid = build_grid()
        expected_pairs = [
            (prefill, decode)
            for prefill in TOKEN_GRID
            for decode in TOKEN_GRID
            if prefill + decode <= MAX_TOKENS
        ]
        actual_pairs = list(
            grid[["num_prefill_tokens", "num_decode_tokens"]].itertuples(
                index=False, name=None
            )
        )
        self.assertEqual(actual_pairs, expected_pairs)
        np.testing.assert_array_equal(
            grid["grid_request_id"].to_numpy(), np.arange(len(grid))
        )
        np.testing.assert_array_equal(
            grid["arrived_at"].to_numpy(),
            np.arange(len(grid)) * ARRIVAL_GAP_SECONDS,
        )

    def test_vidur_lookup_rejects_nonzero_queueing(self) -> None:
        manifest = build_grid()
        metrics = pd.DataFrame(
            {
                "Request Id": manifest["grid_request_id"],
                "request_execution_time": (
                    manifest["num_decode_tokens"]
                    + manifest["num_prefill_tokens"] / 10000.0
                ),
                "request_model_execution_time": (
                    manifest["num_decode_tokens"]
                    + manifest["num_prefill_tokens"] / 10000.0
                ),
                "request_scheduling_delay": 0.0,
                "request_preemption_time": 0.0,
            }
        )
        metrics.loc[0, "request_scheduling_delay"] = 1e-6
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest.to_csv(folder / "manifest.csv", index=False)
            metrics.to_csv(folder / "metrics.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "nonzero"):
                read_vidur_lookup(folder / "manifest.csv", folder / "metrics.csv")

    def test_triangulated_mapping_reproduces_grid_vertices(self) -> None:
        grid = build_grid()
        points = grid[["num_prefill_tokens", "num_decode_tokens"]].to_numpy(
            dtype=float
        )
        values = points[:, 0] / 1000.0 + points[:, 1] / 100.0
        linear, local_upper = map_execution_times(
            Delaunay(points),
            values,
            points[:, 0].astype(int),
            points[:, 1].astype(int),
        )
        np.testing.assert_allclose(linear, values, rtol=0.0, atol=1e-10)
        self.assertTrue(np.all(local_upper + 1e-12 >= linear))


class AzureMappingTests(unittest.TestCase):
    @staticmethod
    def synthetic_time_matrix() -> np.ndarray:
        grid = np.asarray(TOKEN_GRID, dtype=int)
        matrix = np.full((len(grid), len(grid)), np.nan, dtype=float)
        for i, prefill in enumerate(grid):
            for j, decode in enumerate(grid):
                if prefill + decode <= MAX_TOKENS:
                    matrix[i, j] = prefill / 1000.0 + decode / 100.0
        return matrix

    def test_upper_grid_boundary_fallback_and_tp_conversion(self) -> None:
        raw = pd.DataFrame(
            {
                "TIMESTAMP": [
                    "2024-05-10 00:10:00",
                    "2024-05-10 00:20:00",
                    "2024-05-18 00:10:00",
                ],
                "ContextTokens": [3, 3000, 0],
                "GeneratedTokens": [3, 1096, 0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            raw.to_csv(path, index=False)
            hourly, qa = process_azure_trace(
                path, self.synthetic_time_matrix(), boundary_execution_time_s=100.0
            )
        regular_execution = 4 / 1000.0 + 4 / 100.0
        zero_clipped_execution = 1 / 1000.0 + 1 / 100.0
        expected_service = (
            regular_execution + 100.0 + zero_clipped_execution
        ) * TP_SIZE / 3600.0
        self.assertAlmostEqual(
            hourly["service_accelerator_hours"].sum(), expected_service
        )
        self.assertEqual(qa["boundary_fallback_requests"], 1)
        self.assertEqual(qa["processed_context_token_quantiles"]["0.0"], 1)
        self.assertEqual(qa["processed_generated_token_quantiles"]["0.0"], 1)
        self.assertEqual(len(hourly), 193)

    def test_exactly_168_hourly_bins_count_as_seven_days(self) -> None:
        raw = pd.DataFrame(
            {
                "TIMESTAMP": pd.date_range("2024-05-10", periods=168, freq="h"),
                "ContextTokens": [10] * 168,
                "GeneratedTokens": [10] * 168,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            raw.to_csv(path, index=False)
            hourly, qa = process_azure_trace(
                path, self.synthetic_time_matrix(), boundary_execution_time_s=100.0
            )
        self.assertEqual(len(hourly), 168)
        self.assertEqual(qa["coverage_hours_between_endpoints"], 167.0)

    def test_more_than_one_percent_invalid_rows_fail(self) -> None:
        raw = pd.DataFrame(
            {
                "TIMESTAMP": pd.date_range("2024-05-10", periods=100, freq="2h"),
                "ContextTokens": [10] * 99 + [-1],
                "GeneratedTokens": [10] * 100,
            }
        )
        raw.loc[98, "ContextTokens"] = -1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "exceeds 1%"):
                process_azure_trace(
                    path, self.synthetic_time_matrix(), boundary_execution_time_s=100.0
                )


class ExternalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_case(
            ROOT,
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

    def test_external_trace_has_exactly_seven_days(self) -> None:
        self.assertEqual(len(self.case.hour_index), 168)
        span = pd.Timestamp(self.case.timestamps[-1]) - pd.Timestamp(
            self.case.timestamps[0]
        )
        self.assertEqual(span.total_seconds(), 167 * 3600)

    def test_external_capacity_is_nonbinding_in_every_region(self) -> None:
        demand = self.case.demand_accelerator_hours.reshape(-1, 1)
        self.assertTrue(np.all(self.case.capacity_accelerator_hours >= demand))

    def test_external_energy_row_is_llm_proxy_not_diffusion(self) -> None:
        self.assertAlmostEqual(
            self.case.it_energy_kwh_per_accelerator_hour,
            0.6599981359479854,
        )
        self.assertEqual(
            self.case.metadata["energy_workload_class"],
            "text_generation_llm_serving_proxy",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
