from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cws_lexicographic_model import load_case
from run_ema_environmental_frontier import solve_status_quo


ROOT = Path(__file__).resolve().parents[1]


def circular_block_indices(
    n_hours: int, block_hours: int, rng: np.random.Generator
) -> np.ndarray:
    block_count = int(np.ceil(n_hours / block_hours))
    starts = rng.integers(0, n_hours, size=block_count)
    indices = np.concatenate(
        [(start + np.arange(block_hours)) % n_hours for start in starts]
    )
    return indices[:n_hours]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--replicates", type=int, default=4000)
    parser.add_argument("--block-hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    reference = pd.read_csv(args.root / "03_results" / "ema_reference_frontier_v1.csv")
    dispatch = pd.read_csv(args.root / "03_results" / "ema_reference_dispatch_v1.csv")
    fractions = sorted(reference["cap_fraction_of_carbon_first"].unique())
    hours = sorted(dispatch["hour_index"].unique())
    case = load_case(args.root)
    status = solve_status_quo(case)

    status_hourly = (
        status["dispatch"].groupby("hour_index")["carbon_kgco2e"].sum().reindex(hours, fill_value=0.0).to_numpy()
    )
    carbon = []
    high_water = []
    for fraction in fractions:
        subset = dispatch.loc[dispatch["cap_fraction_of_carbon_first"].eq(fraction)]
        carbon.append(
            subset.groupby("hour_index")["carbon_kgco2e"].sum().reindex(hours, fill_value=0.0).to_numpy()
        )
        high_water.append(
            subset.groupby("hour_index")["high_stress_direct_water_l"].sum().reindex(hours, fill_value=0.0).to_numpy()
        )
    carbon_matrix = np.asarray(carbon)
    high_water_matrix = np.asarray(high_water)
    carbon_first_index = fractions.index(1.0)

    rng = np.random.default_rng(args.seed)
    carbon_change = np.empty((args.replicates, len(fractions)), dtype=float)
    added_carbon = np.empty_like(carbon_change)
    water_avoided = np.empty_like(carbon_change)
    for replicate in range(args.replicates):
        indices = circular_block_indices(len(hours), args.block_hours, rng)
        status_total = status_hourly[indices].sum()
        carbon_totals = carbon_matrix[:, indices].sum(axis=1)
        water_totals = high_water_matrix[:, indices].sum(axis=1)
        carbon_first_total = carbon_totals[carbon_first_index]
        carbon_first_water = water_totals[carbon_first_index]
        carbon_change[replicate] = 100.0 * (carbon_totals / status_total - 1.0)
        added_carbon[replicate] = carbon_totals - carbon_first_total
        water_avoided[replicate] = carbon_first_water - water_totals

    records: list[dict[str, object]] = []
    for index, fraction in enumerate(fractions):
        row = reference.loc[reference["cap_fraction_of_carbon_first"].eq(fraction)].iloc[0]
        records.append(
            {
                "cap_fraction_of_carbon_first": fraction,
                "replicates": args.replicates,
                "block_hours": args.block_hours,
                "seed": args.seed,
                "carbon_change_pct_vs_status_quo_estimate": float(row["carbon_change_pct_vs_status_quo"]),
                "carbon_change_pct_vs_status_quo_q025": float(np.quantile(carbon_change[:, index], 0.025)),
                "carbon_change_pct_vs_status_quo_q975": float(np.quantile(carbon_change[:, index], 0.975)),
                "additional_carbon_kgco2e_vs_carbon_first_estimate": float(row["additional_carbon_kgco2e_vs_carbon_first"]),
                "additional_carbon_kgco2e_vs_carbon_first_q025": float(np.quantile(added_carbon[:, index], 0.025)),
                "additional_carbon_kgco2e_vs_carbon_first_q975": float(np.quantile(added_carbon[:, index], 0.975)),
                "screened_direct_water_avoided_l_vs_carbon_first_estimate": float(row["screened_direct_water_avoided_l_vs_carbon_first"]),
                "screened_direct_water_avoided_l_vs_carbon_first_q025": float(np.quantile(water_avoided[:, index], 0.025)),
                "screened_direct_water_avoided_l_vs_carbon_first_q975": float(np.quantile(water_avoided[:, index], 0.975)),
                "interpretation": "paired circular-block quantiles for the supplied fixed solved trace segment; not population confidence intervals or parameter uncertainty",
            }
        )

    output = pd.DataFrame(records)
    output.to_csv(args.root / "03_results" / "ema_reference_block_intervals_v1.csv", index=False)
    summary = {
        "status": "ANALYZED",
        "points": len(output),
        "replicates": args.replicates,
        "block_hours": args.block_hours,
        "seed": args.seed,
        "all_carbon_interval_upper_bounds_nonpositive": bool(
            (output["carbon_change_pct_vs_status_quo_q975"] <= 0.0).all()
        ),
        "interpretation": "conditional temporal-composition stability only",
    }
    (args.root / "03_results" / "ema_reference_block_intervals_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
