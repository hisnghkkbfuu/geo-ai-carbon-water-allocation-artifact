from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from build_external_confirmation_inputs import (
    EXPECTED_COLUMNS,
    MAX_TOKENS,
    TP_SIZE,
    build_nonbinding_capacity,
)


MAPPINGS = ("triangulated_linear", "local_simplex_upper")


def build_triangulation(lookup_path: Path) -> tuple[Delaunay, np.ndarray]:
    lookup = pd.read_csv(lookup_path)
    points = lookup[["num_prefill_tokens", "num_decode_tokens"]].to_numpy(dtype=float)
    values = lookup["request_execution_time"].to_numpy(dtype=float)
    if len(points) != 168 or (~np.isfinite(values)).any() or (values <= 0).any():
        raise RuntimeError("Formal Vidur lookup is incomplete or invalid")
    return Delaunay(points), values


def map_execution_times(
    triangulation: Delaunay,
    vertex_values: np.ndarray,
    prefill: np.ndarray,
    decode: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.column_stack([prefill, decode]).astype(float, copy=False)
    simplex = triangulation.find_simplex(query, tol=1e-10)
    if (simplex < 0).any():
        raise RuntimeError("A clipped token pair lies outside the Vidur grid convex hull")
    transform = triangulation.transform[simplex]
    first_weights = np.einsum(
        "nij,nj->ni", transform[:, :2, :], query - transform[:, 2, :]
    )
    weights = np.column_stack([first_weights, 1.0 - first_weights.sum(axis=1)])
    vertices = triangulation.simplices[simplex]
    values = vertex_values[vertices]
    linear = np.sum(values * weights, axis=1)
    local_upper = values.max(axis=1)
    if (
        (~np.isfinite(linear)).any()
        or (~np.isfinite(local_upper)).any()
        or (linear <= 0).any()
        or (local_upper + 1e-12 < linear).any()
    ):
        raise RuntimeError("Triangulated execution-time mapping is invalid")
    return linear, local_upper


def process_mapping_sensitivities(
    raw_path: Path,
    lookup_path: Path,
    *,
    chunksize: int = 1_000_000,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    triangulation, values = build_triangulation(lookup_path)
    hourly_parts = {mapping: [] for mapping in MAPPINGS}
    rows_total = 0
    rows_valid = 0
    rows_invalid = 0
    total_service = {mapping: 0.0 for mapping in MAPPINGS}

    reader = pd.read_csv(raw_path, chunksize=chunksize)
    if tuple(reader._engine.names) != EXPECTED_COLUMNS:
        raise RuntimeError("Unexpected Azure trace schema")
    for chunk in reader:
        rows_total += len(chunk)
        timestamp = pd.to_datetime(chunk["TIMESTAMP"], errors="coerce", utc=True)
        context_numeric = pd.to_numeric(chunk["ContextTokens"], errors="coerce")
        generated_numeric = pd.to_numeric(chunk["GeneratedTokens"], errors="coerce")
        finite = np.isfinite(context_numeric) & np.isfinite(generated_numeric)
        integer = (
            np.isclose(context_numeric, np.round(context_numeric), rtol=0.0, atol=1e-9)
            & np.isclose(generated_numeric, np.round(generated_numeric), rtol=0.0, atol=1e-9)
        )
        valid = timestamp.notna() & finite & integer & context_numeric.ge(0) & generated_numeric.ge(0)
        rows_invalid += int((~valid).sum())
        if not valid.any():
            continue

        context = np.round(context_numeric[valid]).astype(np.int64).to_numpy()
        generated = np.round(generated_numeric[valid]).astype(np.int64).to_numpy()
        decode = np.clip(generated, 1, MAX_TOKENS - 1)
        prefill = np.clip(context, 1, MAX_TOKENS - decode)
        linear_s, local_upper_s = map_execution_times(
            triangulation, values, prefill, decode
        )
        timestamp_hour = timestamp[valid].dt.floor("h").to_numpy()
        for mapping, execution_s in zip(MAPPINGS, [linear_s, local_upper_s]):
            service = execution_s * TP_SIZE / 3600.0
            work = pd.DataFrame(
                {
                    "trace_timestamp_utc": timestamp_hour,
                    "service_accelerator_hours": service,
                }
            )
            hourly_parts[mapping].append(
                work.groupby("trace_timestamp_utc", as_index=False).agg(
                    request_count=("service_accelerator_hours", "size"),
                    service_accelerator_hours=("service_accelerator_hours", "sum"),
                )
            )
            total_service[mapping] += float(service.sum())
        rows_valid += len(prefill)

    if rows_total == 0 or rows_invalid / rows_total > 0.01:
        raise RuntimeError("Azure mapping sensitivity failed input validity checks")

    outputs: dict[str, pd.DataFrame] = {}
    mapping_qa: dict[str, object] = {}
    for mapping in MAPPINGS:
        hourly = (
            pd.concat(hourly_parts[mapping], ignore_index=True)
            .groupby("trace_timestamp_utc", as_index=False)
            .sum(numeric_only=True)
            .sort_values("trace_timestamp_utc")
        )
        full_hours = pd.DataFrame(
            {
                "trace_timestamp_utc": pd.date_range(
                    hourly["trace_timestamp_utc"].min(),
                    hourly["trace_timestamp_utc"].max(),
                    freq="h",
                    tz="UTC",
                )
            }
        )
        hourly = full_hours.merge(hourly, on="trace_timestamp_utc", how="left").fillna(0.0)
        hourly.insert(0, "hour_index", np.arange(len(hourly), dtype=int))
        hourly["request_count"] = hourly["request_count"].astype(np.int64)
        hourly["service_mapping"] = mapping
        outputs[mapping] = hourly
        mapping_qa[mapping] = {
            "total_service_accelerator_hours": float(total_service[mapping]),
            "peak_hour_service_accelerator_hours": float(
                hourly["service_accelerator_hours"].max()
            ),
            "hourly_rows": int(len(hourly)),
            "nonzero_hours": int(hourly["service_accelerator_hours"].gt(0).sum()),
        }

    qa = {
        "status": "POST_CONFIRMATION_EXPLORATORY",
        "raw_rows": int(rows_total),
        "valid_rows": int(rows_valid),
        "excluded_invalid_rows": int(rows_invalid),
        "mappings": mapping_qa,
        "interpretation": {
            "triangulated_linear": (
                "piecewise-linear interpolation on the formal Vidur token grid convex hull"
            ),
            "local_simplex_upper": (
                "maximum execution time among the three vertices of the containing simplex"
            ),
        },
    }
    return outputs, qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-trace", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    data = args.root / "01_data_processed"
    lookup_path = data / "external_vidur_execution_time_lookup_v1.csv"
    outputs, qa = process_mapping_sensitivities(args.raw_trace, lookup_path)
    for mapping, hourly in outputs.items():
        hourly.to_csv(
            data / f"azure_code_hourly_service_demand_{mapping}_exploratory_v1.csv",
            index=False,
        )
        build_nonbinding_capacity(hourly).to_csv(
            data / f"azure_code_nonbinding_capacity_{mapping}_exploratory_v1.csv",
            index=False,
        )
    (data / "external_mapping_sensitivity_input_qa_v1.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
