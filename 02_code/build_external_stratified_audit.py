from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_external_confirmation import BASELINE, CARBON_FIRST, ZERO_HIGH_STRESS


def percent_change(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0) if baseline else np.nan


def daily_contrasts(dispatch: pd.DataFrame) -> pd.DataFrame:
    work = dispatch.copy()
    work["date_utc"] = pd.to_datetime(
        work["trace_timestamp_anonymized"], utc=True
    ).dt.date.astype(str)
    measures = [
        "assigned_accelerator_hours",
        "carbon_kgco2e",
        "high_stress_direct_water_l",
    ]
    grouped = (
        work.groupby(["service_mapping", "date_utc", "policy"], as_index=False)[
            measures
        ]
        .sum()
        .set_index(["service_mapping", "date_utc", "policy"])
    )
    records: list[dict[str, object]] = []
    for (mapping, date), frame in grouped.groupby(level=[0, 1]):
        policy = frame.droplevel([0, 1])
        e0 = policy.loc[BASELINE]
        e1 = policy.loc[CARBON_FIRST]
        e2 = policy.loc[ZERO_HIGH_STRESS]
        records.append(
            {
                "service_mapping": mapping,
                "date_utc": date,
                "e1_vs_e0_carbon_change_pct": percent_change(
                    float(e1["carbon_kgco2e"]), float(e0["carbon_kgco2e"])
                ),
                "e2_vs_e0_carbon_change_pct": percent_change(
                    float(e2["carbon_kgco2e"]), float(e0["carbon_kgco2e"])
                ),
                "e2_vs_e1_carbon_penalty_pct": percent_change(
                    float(e2["carbon_kgco2e"]), float(e1["carbon_kgco2e"])
                ),
                "e1_vs_e0_high_stress_water_change_l": float(
                    e1["high_stress_direct_water_l"]
                    - e0["high_stress_direct_water_l"]
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
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    results = args.root / "04_results"
    frozen = pd.read_csv(results / "external_confirmation_policy_dispatch_v1.csv")
    frozen.insert(0, "service_mapping", "frozen_global_upper")
    exploratory = pd.read_csv(
        results / "external_mapping_sensitivity_policy_dispatch_v1.csv"
    )
    daily = daily_contrasts(pd.concat([frozen, exploratory], ignore_index=True))
    daily.to_csv(results / "external_daily_stratified_effects_v1.csv", index=False)

    summary: dict[str, object] = {
        "status": "ANALYZED",
        "purpose": "daily direction audit; not an additional confirmatory endpoint",
        "mappings": {},
    }
    for mapping, frame in daily.groupby("service_mapping"):
        summary["mappings"][mapping] = {
            "days": int(len(frame)),
            "days_e1_carbon_lower_than_e0": int(
                frame["e1_vs_e0_carbon_change_pct"].lt(0).sum()
            ),
            "days_e2_carbon_lower_than_e0": int(
                frame["e2_vs_e0_carbon_change_pct"].lt(0).sum()
            ),
            "days_e1_high_stress_water_above_e0": int(
                frame["e1_vs_e0_high_stress_water_change_l"].gt(0).sum()
            ),
            "e1_carbon_change_min_pct": float(
                frame["e1_vs_e0_carbon_change_pct"].min()
            ),
            "e1_carbon_change_max_pct": float(
                frame["e1_vs_e0_carbon_change_pct"].max()
            ),
            "e2_vs_e1_carbon_penalty_min_pct": float(
                frame["e2_vs_e1_carbon_penalty_pct"].min()
            ),
            "e2_vs_e1_carbon_penalty_max_pct": float(
                frame["e2_vs_e1_carbon_penalty_pct"].max()
            ),
            "max_abs_service_difference_accelerator_hours": float(
                frame[
                    [
                        "e1_vs_e0_service_difference_accelerator_hours",
                        "e2_vs_e0_service_difference_accelerator_hours",
                    ]
                ]
                .abs()
                .to_numpy()
                .max()
            ),
        }
    (results / "external_daily_stratified_audit_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
