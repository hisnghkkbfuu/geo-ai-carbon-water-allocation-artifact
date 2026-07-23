from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_vidur_calibration_grid import MAX_TOKENS, TOKEN_GRID


OFFICIAL_AZURE_CODE_SIZE = 691_989_454
EXPECTED_COLUMNS = ("TIMESTAMP", "ContextTokens", "GeneratedTokens")
TP_SIZE = 8
VIDUR_COMMIT = "8383d2935bc62723a212090baa9f98ada206fc14"
ENERGY_SOURCE_CLASS = "distributed_training_like"
ENERGY_EXTERNAL_CLASS = "text_generation_llm_serving_proxy"
REGIONS = ("east", "north", "northwest", "southwest")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def exact_hist_quantile(histogram: np.ndarray, quantile: float) -> int:
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("Cannot compute a token quantile from an empty histogram")
    target = quantile * (total - 1)
    return int(np.searchsorted(np.cumsum(histogram), target + 1, side="left"))


def read_vidur_lookup(
    manifest_path: Path, metrics_path: Path
) -> tuple[pd.DataFrame, np.ndarray, float, dict[str, object]]:
    manifest = pd.read_csv(manifest_path).sort_values("grid_request_id")
    metrics = pd.read_csv(metrics_path).rename(columns={"Request Id": "grid_request_id"})
    metrics["grid_request_id"] = metrics["grid_request_id"].astype(int)
    lookup = manifest.merge(metrics, on="grid_request_id", how="outer", validate="one_to_one")
    if len(lookup) != len(manifest) or lookup["request_execution_time"].isna().any():
        raise RuntimeError("Vidur manifest and request metrics do not map one-to-one")
    expected_ids = np.arange(len(lookup), dtype=int)
    if not np.array_equal(lookup["grid_request_id"].to_numpy(), expected_ids):
        raise RuntimeError("Vidur request ids are not contiguous and ordered")
    for column in ["request_scheduling_delay", "request_preemption_time"]:
        if float(lookup[column].abs().max()) > 1e-9:
            raise RuntimeError(f"Isolated Vidur grid has nonzero {column}")
    execution = lookup["request_execution_time"].astype(float)
    if (~np.isfinite(execution)).any() or execution.le(0).any():
        raise RuntimeError("Vidur execution-time lookup contains invalid values")
    slowest = lookup.loc[execution.idxmax()]
    if (int(slowest["num_prefill_tokens"]), int(slowest["num_decode_tokens"])) != (
        1,
        4095,
    ):
        raise RuntimeError("The slowest formal Vidur grid point is not (1, 4095)")

    grid = np.asarray(TOKEN_GRID, dtype=int)
    matrix = np.full((len(grid), len(grid)), np.nan, dtype=float)
    index = {int(value): position for position, value in enumerate(grid)}
    for row in lookup.itertuples(index=False):
        matrix[index[int(row.num_prefill_tokens)], index[int(row.num_decode_tokens)]] = float(
            row.request_execution_time
        )
    qa = {
        "grid_requests": int(len(lookup)),
        "max_scheduling_delay_s": float(lookup["request_scheduling_delay"].abs().max()),
        "max_preemption_time_s": float(lookup["request_preemption_time"].abs().max()),
        "slowest_prefill_tokens": int(slowest["num_prefill_tokens"]),
        "slowest_decode_tokens": int(slowest["num_decode_tokens"]),
        "slowest_execution_time_s": float(slowest["request_execution_time"]),
        "vidur_commit": VIDUR_COMMIT,
        "configuration": {
            "model": "meta-llama/Llama-2-70b-hf",
            "device": "h100",
            "network_device": "h100_dgx",
            "tensor_parallel_size": TP_SIZE,
            "pipeline_stages": 1,
            "replicas": 1,
            "scheduler": "sarathi",
            "max_tokens": MAX_TOKENS,
        },
    }
    keep = [
        "grid_request_id",
        "num_prefill_tokens",
        "num_decode_tokens",
        "request_execution_time",
        "request_model_execution_time",
        "request_scheduling_delay",
        "request_preemption_time",
    ]
    return lookup[keep].copy(), matrix, float(slowest["request_execution_time"]), qa


def process_azure_trace(
    raw_path: Path,
    time_matrix: np.ndarray,
    boundary_execution_time_s: float,
    *,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    grid = np.asarray(TOKEN_GRID, dtype=int)
    hourly_parts: list[pd.DataFrame] = []
    context_hist = np.zeros(MAX_TOKENS + 1, dtype=np.int64)
    generated_hist = np.zeros(MAX_TOKENS + 1, dtype=np.int64)
    rows_total = 0
    rows_valid = 0
    rows_invalid = 0
    boundary_rows = 0
    boundary_service = 0.0
    total_service = 0.0

    reader = pd.read_csv(raw_path, chunksize=chunksize)
    if tuple(reader._engine.names) != EXPECTED_COLUMNS:
        raise RuntimeError(
            f"Unexpected Azure columns: {tuple(reader._engine.names)}; expected {EXPECTED_COLUMNS}"
        )

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

        p_raw = np.round(context_numeric[valid]).astype(np.int64).to_numpy()
        d_raw = np.round(generated_numeric[valid]).astype(np.int64).to_numpy()
        d = np.clip(d_raw, 1, MAX_TOKENS - 1)
        p = np.clip(p_raw, 1, MAX_TOKENS - d)
        p_index = np.searchsorted(grid, p, side="left")
        d_index = np.searchsorted(grid, d, side="left")
        p_up = grid[p_index]
        d_up = grid[d_index]
        boundary = p_up + d_up > MAX_TOKENS
        execution_s = np.full(len(p), boundary_execution_time_s, dtype=float)
        regular = ~boundary
        execution_s[regular] = time_matrix[p_index[regular], d_index[regular]]
        if (~np.isfinite(execution_s)).any() or (execution_s <= 0).any():
            raise RuntimeError("At least one Azure row did not map to the frozen Vidur lookup")

        service = execution_s * TP_SIZE / 3600.0
        valid_timestamp = timestamp[valid]
        work = pd.DataFrame(
            {
                "trace_timestamp_utc": valid_timestamp.dt.floor("h").to_numpy(),
                "service_accelerator_hours": service,
                "boundary_fallback_request": boundary.astype(np.int64),
                "boundary_fallback_service_accelerator_hours": service * boundary,
                "processed_context_tokens": p,
                "processed_generated_tokens": d,
            }
        )
        hourly_parts.append(
            work.groupby("trace_timestamp_utc", as_index=False).agg(
                request_count=("service_accelerator_hours", "size"),
                service_accelerator_hours=("service_accelerator_hours", "sum"),
                boundary_fallback_request_count=("boundary_fallback_request", "sum"),
                boundary_fallback_service_accelerator_hours=(
                    "boundary_fallback_service_accelerator_hours",
                    "sum",
                ),
                processed_context_tokens=("processed_context_tokens", "sum"),
                processed_generated_tokens=("processed_generated_tokens", "sum"),
            )
        )
        context_hist += np.bincount(p, minlength=MAX_TOKENS + 1)
        generated_hist += np.bincount(d, minlength=MAX_TOKENS + 1)
        rows_valid += len(p)
        boundary_rows += int(boundary.sum())
        boundary_service += float(service[boundary].sum())
        total_service += float(service.sum())

    if rows_total == 0 or rows_valid == 0:
        raise RuntimeError("Azure trace contains no valid requests")
    invalid_share = rows_invalid / rows_total
    if invalid_share > 0.01:
        raise RuntimeError(f"Invalid Azure token/timestamp row share exceeds 1%: {invalid_share:.6%}")

    hourly = (
        pd.concat(hourly_parts, ignore_index=True)
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
    for column in ["request_count", "boundary_fallback_request_count"]:
        hourly[column] = hourly[column].astype(np.int64)

    span_hours = (
        hourly["trace_timestamp_utc"].iloc[-1] - hourly["trace_timestamp_utc"].iloc[0]
    ).total_seconds() / 3600.0
    if len(hourly) < 7 * 24:
        raise RuntimeError(
            "Azure trace covers fewer than 168 consecutive hourly bins: "
            f"{len(hourly)} bins ({span_hours:.3f} h between endpoints)"
        )

    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    qa = {
        "raw_rows": int(rows_total),
        "valid_rows": int(rows_valid),
        "excluded_invalid_rows": int(rows_invalid),
        "invalid_row_share": float(invalid_share),
        "first_timestamp_utc": str(hourly["trace_timestamp_utc"].iloc[0]),
        "last_timestamp_utc": str(hourly["trace_timestamp_utc"].iloc[-1]),
        "coverage_hours_between_endpoints": float(span_hours),
        "hourly_rows": int(len(hourly)),
        "nonzero_hours": int(hourly["service_accelerator_hours"].gt(0).sum()),
        "total_service_accelerator_hours": float(total_service),
        "peak_hour_service_accelerator_hours": float(
            hourly["service_accelerator_hours"].max()
        ),
        "boundary_fallback_requests": int(boundary_rows),
        "boundary_fallback_request_share": float(boundary_rows / rows_valid),
        "boundary_fallback_service_accelerator_hours": float(boundary_service),
        "boundary_fallback_service_share": float(boundary_service / total_service),
        "processed_context_token_quantiles": {
            str(q): exact_hist_quantile(context_hist, q) for q in quantiles
        },
        "processed_generated_token_quantiles": {
            str(q): exact_hist_quantile(generated_hist, q) for q in quantiles
        },
        "service_definition": (
            "Vidur H100/Llama-2-70B request_execution_time x TP8 / 3600; "
            "model-calibrated H100-equivalent accelerator-hours, not Azure-measured GPU occupancy"
        ),
    }
    return hourly, qa


def build_aligned_carbon(root: Path, hourly: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(root / "01_data_processed" / "gentd26_aligned_regional_hourly_cef.csv")
    source = source[source["carbon_mapping_scenario"].eq("equal_site_portfolio")].copy()
    source_hours = sorted(source["hour_index"].unique())
    if len(source_hours) == 0:
        raise RuntimeError("No equal-site carbon sequence is available")
    records: list[pd.DataFrame] = []
    for hour_index, timestamp in hourly[["hour_index", "trace_timestamp_utc"]].itertuples(
        index=False
    ):
        source_hour = source_hours[int(hour_index) % len(source_hours)]
        frame = source[source["hour_index"].eq(source_hour)].copy()
        frame["hour_index"] = int(hour_index)
        frame["trace_timestamp_utc"] = timestamp
        frame["external_alignment_source_hour_index"] = int(source_hour)
        frame["alignment_status"] = (
            "cyclic ordinal alignment to projected 2025 S1 CEF; not contemporaneous Azure carbon"
        )
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def build_nonbinding_capacity(hourly: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in hourly.itertuples(index=False):
        capacity = float(row.service_accelerator_hours) * (1.0 + 1e-9) + 1e-9
        for region in REGIONS:
            records.append(
                {
                    "capacity_scenario": "external_nonbinding",
                    "hour_index": int(row.hour_index),
                    "region": region,
                    "capacity_accelerator_hours": capacity,
                    "capacity_status": (
                        "constructed nonbinding capacity for fixed-service external confirmation"
                    ),
                }
            )
    return pd.DataFrame(records)


def build_external_energy(energy_calibration: Path) -> pd.DataFrame:
    source_path = energy_calibration
    energy = pd.read_csv(source_path)
    energy = energy[energy["model_workload_class"].eq(ENERGY_SOURCE_CLASS)].copy()
    if len(energy) != 1:
        raise RuntimeError("Expected one Text Generation LLM energy calibration row")
    energy["original_model_workload_class"] = ENERGY_SOURCE_CLASS
    energy["model_workload_class"] = ENERGY_EXTERNAL_CLASS
    energy["external_use_boundary"] = (
        "serving energy-intensity proxy; not measured Azure or Vidur electrical power"
    )
    return energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-trace", type=Path, required=True)
    parser.add_argument("--vidur-metrics", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--energy-calibration", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "01_data_processed"
    data.mkdir(parents=True, exist_ok=True)

    actual_size = args.raw_trace.stat().st_size
    if actual_size != OFFICIAL_AZURE_CODE_SIZE:
        raise RuntimeError(
            f"Azure code trace size mismatch: {actual_size} != {OFFICIAL_AZURE_CODE_SIZE}"
        )

    manifest_path = data / "external_vidur_grid_manifest_v1.csv"
    lookup, matrix, boundary_time, vidur_qa = read_vidur_lookup(
        manifest_path, args.vidur_metrics
    )
    lookup.to_csv(data / "external_vidur_execution_time_lookup_v1.csv", index=False)
    hourly, trace_qa = process_azure_trace(args.raw_trace, matrix, boundary_time)
    hourly.to_csv(data / "azure_code_hourly_service_demand_v1.csv", index=False)
    build_aligned_carbon(args.root, hourly).to_csv(
        data / "azure_code_aligned_regional_hourly_cef_v1.csv", index=False
    )
    build_nonbinding_capacity(hourly).to_csv(
        data / "azure_code_nonbinding_capacity_v1.csv", index=False
    )
    external_energy = build_external_energy(args.energy_calibration)
    external_energy.to_csv(data / "external_llm_energy_calibration_v1.csv", index=False)

    provenance = {
        "status": "ANALYZED",
        "dataset": "Azure LLM Inference Trace 2024 code-service one-week trace",
        "official_release_url": (
            "https://github.com/Azure/AzurePublicDataset/releases/tag/dataset-llm-2024"
        ),
        "official_documentation_url": (
            "https://github.com/Azure/AzurePublicDataset/blob/master/"
            "AzureLLMInferenceDataset2024.md"
        ),
        "license": "CC BY 4.0",
        "raw_file_name": args.raw_trace.name,
        "raw_file_size_bytes": actual_size,
        "raw_file_sha256": sha256_file(args.raw_trace),
        "vidur_repository": "https://github.com/microsoft/vidur",
        "vidur_commit": VIDUR_COMMIT,
        "vidur_metrics_sha256": sha256_file(args.vidur_metrics),
        "vidur_grid_manifest_sha256": sha256_file(manifest_path),
        "energy_calibration_source": args.energy_calibration.name,
        "energy_calibration_source_sha256": sha256_file(args.energy_calibration),
        "claim_boundaries": [
            "Azure publishes token lengths and timestamps, not GPU execution time or power.",
            "Vidur supplies a model-based execution-time calibration, not Azure-observed latency.",
            "The LLM energy intensity is an external measured-workload proxy, not Azure site telemetry.",
            "Carbon is cyclic ordinal scenario alignment and is not contemporaneous with Azure timestamps.",
            "Nonbinding capacity is constructed solely to preserve equal service across policies.",
        ],
    }
    qa = {"status": "ANALYZED", "trace": trace_qa, "vidur": vidur_qa}
    (data / "external_confirmation_input_qa_v1.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.root / "00_governance" / "external_confirmation_provenance_v1.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
