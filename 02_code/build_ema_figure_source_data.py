from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    frame.to_csv(path, index=False)
    return {"file": path.name, "rows": len(frame), "sha256": sha256(path)}


def deterministic_summary(frame: pd.DataFrame, group: str, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, subset in frame.groupby(group, sort=True):
        row: dict[str, object] = {group: value, "scenario_count": len(subset)}
        for metric in metrics:
            values = subset[metric]
            row[f"{metric}_minimum"] = float(values.min())
            row[f"{metric}_q25"] = float(values.quantile(0.25))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q75"] = float(values.quantile(0.75))
            row[f"{metric}_maximum"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    source_dir = root / "04_figures" / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root / "03_results"
    manifest: list[dict[str, object]] = []

    model_definition = pd.DataFrame(
        [
            {"component": "Demand", "role": "Trace-derived hourly service proxy", "evidence_class": "Observed trace field"},
            {"component": "Regional inputs", "role": "Carbon, PUE, WUE, capacity, latency, water-stress scenario layers", "evidence_class": "Proxy or scenario"},
            {"component": "Stage 1", "role": "Minimize unserved service", "evidence_class": "Optimization definition"},
            {"component": "Stage 2", "role": "Minimize carbon at fixed maximum feasible service", "evidence_class": "Optimization definition"},
            {"component": "Screen", "role": "Bound physical direct water in High or Extremely high water-stress regions", "evidence_class": "Scenario constraint"},
            {"component": "Boundary", "role": "No facility measurement, causal estimate, hydrologic damage model, or cross-hour queueing", "evidence_class": "Interpretation limit"},
        ]
    )
    payload = write_csv(model_definition, source_dir / "fig01_model_definition_source_v1.csv")
    manifest.append({"figure": "Fig. 1", "description": "Model components and interpretation boundary", **payload})

    reference = pd.read_csv(results_dir / "ema_reference_frontier_v1.csv")
    reference_columns = [
        "cap_fraction_of_carbon_first",
        "carbon_kgco2e",
        "carbon_change_pct_vs_status_quo",
        "high_stress_direct_water_l",
        "screened_direct_water_avoided_l_vs_carbon_first",
        "additional_carbon_kgco2e_vs_carbon_first",
        "additional_carbon_pct_vs_carbon_first",
        "migration_share",
        "assigned_east_accelerator_hours",
        "assigned_north_accelerator_hours",
        "assigned_northwest_accelerator_hours",
        "assigned_southwest_accelerator_hours",
        "max_abs_demand_balance_error",
        "max_capacity_violation",
        "max_migration_violation",
        "ineligible_latency_assignment",
        "service_gap_vs_status_quo_unserved",
        "high_stress_water_cap_violation_l",
    ]
    payload = write_csv(reference[reference_columns], source_dir / "fig02_reference_frontier_source_v1.csv")
    manifest.append({"figure": "Fig. 2", "description": "Reference epsilon-constraint frontier", **payload})

    segments = pd.read_csv(results_dir / "ema_reference_frontier_segments_v1.csv")
    payload = write_csv(segments, source_dir / "fig02_reference_segment_source_v1.csv")
    manifest.append({"figure": "Fig. 2", "description": "Reference marginal trade-off segments", **payload})

    block_intervals = pd.read_csv(results_dir / "ema_reference_block_intervals_v1.csv")
    payload = write_csv(block_intervals, source_dir / "fig02_reference_block_intervals_source_v1.csv")
    manifest.append({"figure": "Fig. 2", "description": "Conditional paired circular-block intervals for reference frontier", **payload})

    selected = reference.loc[
        reference["cap_fraction_of_carbon_first"].isin([0.0, 0.5, 1.0]),
        [
            "cap_fraction_of_carbon_first",
            "assigned_east_accelerator_hours",
            "assigned_north_accelerator_hours",
            "assigned_northwest_accelerator_hours",
            "assigned_southwest_accelerator_hours",
            "high_stress_direct_water_l",
            "carbon_kgco2e",
        ],
    ].copy()
    payload = write_csv(selected, source_dir / "fig02_selected_allocations_source_v1.csv")
    manifest.append({"figure": "Fig. 2", "description": "Reference regional allocation at three cap points", **payload})

    scenario = pd.read_csv(results_dir / "ema_scenario_frontier_v1.csv")
    scenario_metrics = [
        "carbon_change_pct_vs_status_quo",
        "additional_carbon_kgco2e_vs_carbon_first",
        "screened_direct_water_avoided_l_vs_carbon_first",
    ]
    scenario_summary = deterministic_summary(
        scenario, "cap_fraction_of_carbon_first", scenario_metrics
    )
    payload = write_csv(scenario_summary, source_dir / "fig03_scenario_cap_summary_source_v1.csv")
    manifest.append({"figure": "Fig. 3", "description": "Deterministic 243-cell scenario envelope by cap", **payload})

    zero_cap = scenario.loc[scenario["cap_fraction_of_carbon_first"].eq(0.0)].copy()
    payload = write_csv(zero_cap, source_dir / "fig03_zero_cap_scenarios_source_v1.csv")
    manifest.append({"figure": "Fig. 3", "description": "All zero-cap scenario cells", **payload})

    pue_summary = deterministic_summary(
        zero_cap, "pue_scenario", ["carbon_change_pct_vs_status_quo"]
    )
    payload = write_csv(pue_summary, source_dir / "fig03_zero_cap_pue_summary_source_v1.csv")
    manifest.append({"figure": "Fig. 3", "description": "Zero-cap deterministic ranges by PUE scenario", **payload})

    scenario_segments = pd.read_csv(results_dir / "ema_scenario_frontier_segments_v1.csv")
    payload = write_csv(scenario_segments, source_dir / "fig03_scenario_segment_source_v1.csv")
    manifest.append({"figure": "Fig. 3", "description": "Scenario adjacent-cap marginal trade-offs", **payload})

    qa = pd.read_csv(root / "06_qa" / "ema_environmental_frontier_qa_v1.csv")
    payload = write_csv(qa, source_dir / "fig03_solver_qa_source_v1.csv")
    manifest.append({"figure": "Fig. 3", "description": "Machine-readable scenario QA checks", **payload})

    wue = pd.read_csv(results_dir / "ema_spatial_wue_frontier_v1.csv")
    payload = write_csv(wue, source_dir / "fig04_spatial_wue_frontier_source_v1.csv")
    manifest.append({"figure": "Fig. 4", "description": "81-profile reoptimized spatial WUE frontier", **payload})

    wue_summary = deterministic_summary(
        wue,
        "cap_fraction_of_carbon_first",
        ["carbon_change_pct_vs_status_quo", "additional_carbon_kgco2e_vs_carbon_first"],
    )
    payload = write_csv(wue_summary, source_dir / "fig04_spatial_wue_cap_summary_source_v1.csv")
    manifest.append({"figure": "Fig. 4", "description": "Deterministic WUE-profile envelope by cap", **payload})

    wue_inputs = pd.read_csv(root / "01_data_processed" / "wue_profiles.csv")
    wue_inputs = wue_inputs.loc[wue_inputs["wue_profile"].str.startswith("factorial_")]
    wue_attributes = wue_inputs.pivot(
        index="wue_profile", columns="region", values="wue_l_per_kwh_it"
    ).reset_index()
    wue_attributes["mean_screened_region_wue_l_per_kwh_it"] = (
        wue_attributes["north"] + wue_attributes["northwest"]
    ) / 2.0
    wue_attributes["mean_unscreened_region_wue_l_per_kwh_it"] = (
        wue_attributes["east"] + wue_attributes["southwest"]
    ) / 2.0
    payload = write_csv(wue_attributes, source_dir / "fig04_wue_profile_attributes_source_v1.csv")
    manifest.append({"figure": "Fig. 4", "description": "Spatial WUE profile attributes", **payload})

    energy = pd.read_csv(results_dir / "ema_energy_scaling_frontier_v1.csv")
    payload = write_csv(energy, source_dir / "fig04_energy_scaling_source_v1.csv")
    manifest.append({"figure": "Fig. 4", "description": "Energy-intensity scaling invariance check", **payload})

    source_manifest = pd.DataFrame(manifest)
    source_manifest.to_csv(source_dir / "figure_source_data_manifest_v1.csv", index=False)


if __name__ == "__main__":
    main()
