from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_code"))

from run_ema_environmental_frontier import add_segment_metrics  # noqa: E402


class SegmentMetricTests(unittest.TestCase):
    def test_tightening_cap_reports_positive_water_avoidance_and_penalty(self) -> None:
        frame = pd.DataFrame(
            {
                "family": ["reference", "reference", "reference"],
                "cap_fraction_of_carbon_first": [1.0, 0.5, 0.0],
                "screened_direct_water_avoided_l_vs_carbon_first": [0.0, 2.0, 4.0],
                "additional_carbon_kgco2e_vs_carbon_first": [0.0, 1.0, 3.0],
            }
        )
        result = add_segment_metrics(frame, ["family"])
        self.assertEqual(len(result), 2)
        self.assertTrue(result["monotone_water_avoidance_pass"].all())
        self.assertTrue(result["monotone_carbon_penalty_pass"].all())
        self.assertAlmostEqual(
            result.loc[0, "marginal_additional_carbon_kgco2e_per_screened_l_avoided"],
            0.5,
        )

    def test_zero_water_increment_leaves_marginal_ratio_missing(self) -> None:
        frame = pd.DataFrame(
            {
                "family": ["flat", "flat"],
                "cap_fraction_of_carbon_first": [1.0, 0.0],
                "screened_direct_water_avoided_l_vs_carbon_first": [0.0, 0.0],
                "additional_carbon_kgco2e_vs_carbon_first": [0.0, 0.0],
            }
        )
        result = add_segment_metrics(frame, ["family"])
        self.assertTrue(pd.isna(result.loc[0, "marginal_additional_carbon_kgco2e_per_screened_l_avoided"]))
        self.assertTrue(result.loc[0, "monotone_water_avoidance_pass"])
        self.assertTrue(result.loc[0, "monotone_carbon_penalty_pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
