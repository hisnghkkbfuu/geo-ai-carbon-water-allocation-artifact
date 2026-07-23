from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TOKEN_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4095)
MAX_TOKENS = 4096
ARRIVAL_GAP_SECONDS = 1000


def build_grid() -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    request_id = 0
    for prefill in TOKEN_GRID:
        for decode in TOKEN_GRID:
            if prefill + decode > MAX_TOKENS:
                continue
            rows.append(
                {
                    "grid_request_id": request_id,
                    "arrived_at": request_id * ARRIVAL_GAP_SECONDS,
                    "num_prefill_tokens": prefill,
                    "num_decode_tokens": decode,
                }
            )
            request_id += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()

    grid = build_grid()
    args.trace_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    grid[["arrived_at", "num_prefill_tokens", "num_decode_tokens"]].to_csv(
        args.trace_out, index=False
    )
    grid.to_csv(args.manifest_out, index=False)
    print(f"Wrote {len(grid)} isolated Vidur calibration requests")


if __name__ == "__main__":
    main()
