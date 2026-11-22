"""CLI boundary for versioned, non-overwriting conservative extraction products."""
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from conservative import SCHEMA, extract_conservative
from local_range import LocalRangeClient
from range_client import RangeClient
from tiff_header import TiffHeaderReader


def configure_parser(parser):
    parser.add_argument("url")
    parser.add_argument("--local-file", type=Path,
                        help="read an existing downloaded TIFF; URL remains its archive provenance")
    parser.add_argument("--source-version", required=True, help="immutable archive revision/bitstream UUID")
    parser.add_argument("--citation", required=True, help="publisher/archive dataset record URL")
    parser.add_argument("--units", default="nematodes/m^2")
    parser.add_argument("--sampling-depth", default="unknown")
    parser.add_argument("--source-date", default="unknown")
    parser.add_argument("--licence", default="unknown")
    parser.add_argument("--out-width", type=int, default=1024)
    parser.add_argument("--out-height", type=int, default=512)
    parser.add_argument("--west", type=float, default=-180)
    parser.add_argument("--east", type=float, default=180)
    parser.add_argument("--north", type=float, default=88)
    parser.add_argument("--south", type=float, default=-88)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-budget-mib", type=float, default=80)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--block-rows", type=int, default=32)
    parser.add_argument("--pilot-start-row", type=int, default=0)
    parser.add_argument("--pilot-rows", type=int, help="bounded non-publishable partial run")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--request-interval", type=float, default=1,
                        help="minimum seconds between HTTP request starts, including headers/retries")
    parser.add_argument("--rate-limit-cooldown", type=float, default=60,
                        help="minimum cooldown after HTTP429; longer Retry-After always wins")


def run_cli(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("tag must be a simple filename component")
    if args.out_width < 1 or args.out_height < 1:
        raise ValueError("Output dimensions must be positive")
    if args.pilot_start_row and args.pilot_rows is None:
        raise ValueError("--pilot-start-row requires --pilot-rows")
    if args.max_retries < 1 or args.timeout <= 0:
        raise ValueError("Retry count and timeout must be positive")
    grid_path = args.out_dir / f"{args.tag}_grid.npz"
    meta_path = args.out_dir / f"{args.tag}_meta.json"
    if grid_path.exists() or meta_path.exists():
        raise FileExistsError("Final output already exists; choose a new tag")
    start = time.monotonic()
    local_path = getattr(args, "local_file", None)
    if local_path is not None:
        print(json.dumps(dict(event="hashing-local-source", file=local_path.name)), flush=True)
        client = LocalRangeClient(local_path)
    else:
        client = RangeClient(args.url, max_retries=args.max_retries, timeout=args.timeout,
                             min_request_interval=args.request_interval,
                             rate_limit_cooldown=args.rate_limit_cooldown)
    # Two bounded requests usually contain the whole ETH IFD + strip tables;
    # avoid a burst of tiny requests triggering the archive's rate limiter.
    client.fetch(0, 15)
    prefix = client.fetch(0, min(client.source_identity[0], 512 * 1024) - 1)
    def header_fetch(a, b):
        return prefix[a:b + 1] if b < len(prefix) else client.fetch(a, b)
    info = TiffHeaderReader(args.url, header_fetch, source_identity=client.source_identity).parse()
    print(json.dumps(dict(event="header", width=info.width, height=info.height,
                          raster_type=info.raster_type, geographic_crs=info.geographic_crs,
                          source_identity=client.source_identity)), flush=True)
    header_bytes = client.bytes_fetched
    result = extract_conservative(
        args.url, info, np.linspace(args.west, args.east, args.out_width + 1),
        np.linspace(args.north, args.south, args.out_height + 1),
        cache_dir=args.cache_dir, source_version=args.source_version, client=client,
        block_rows=args.block_rows, row_start=args.pilot_start_row,
        row_stop=None if args.pilot_rows is None else args.pilot_start_row + args.pilot_rows,
        cache_budget_bytes=int(args.cache_budget_mib * 1024**2), progress=True)
    if local_path is not None:
        client.verify_unchanged()  # required even when all reduced blocks were cached
    if args.pilot_rows is None and not result["complete"]:
        raise RuntimeError("Incomplete extraction cannot be published")
    # A deliberately bounded command is labelled a pilot even on a tiny fixture
    # whose selected row count happens to cover the whole file.
    result["publishable"] = result["complete"] and args.pilot_rows is None
    arrays = {k: result.pop(k) for k in
              ("density", "valid_model_fraction", "processed_fraction", "lon_edges", "lat_edges")}
    result.update(schema=SCHEMA, elapsed_s=time.monotonic() - start,
                  header_bytes_fetched=header_bytes, requests_made=client.requests_made,
                  retries_used=client.retries_used,
                  source_http_identity=client.source_identity if local_path is None else None,
                  request_interval_s=args.request_interval, rate_limit_cooldown_s=args.rate_limit_cooldown,
                  source_payload_bytes=info.width * info.height * info.bits_per_sample // 8,
                  provenance=dict(citation=args.citation, source_url=args.url,
                                  units=args.units, sampling_depth=args.sampling_depth,
                                  source_date=args.source_date, licence=args.licence,
                                  extracted_at=datetime.now(timezone.utc).isoformat(),
                                  transformation="spherical overlap-area mean; invalid source contributes no known soil mass; validity separate",
                                  point_support="midpoint cells, clipped to destination domain"))
    result["provenance"]["transport"] = "local-file" if local_path is not None else "http-range"
    if local_path is not None:
        result["local_reads_made"] = client.reads_made
        result["provenance"].update(local_file_name=local_path.name,
                                     local_file_sha256=client.sha256,
                                     local_file_bytes=client.source_identity[0])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_grid = grid_path.with_suffix(".npz.part")
    with temp_grid.open("wb") as f:
        np.savez_compressed(f, **arrays)
    result["grid_sha256"] = hashlib.sha256(temp_grid.read_bytes()).hexdigest()
    result["grid_file"] = grid_path.name
    temp_meta = meta_path.with_suffix(".json.part")
    temp_meta.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp_grid.replace(grid_path)
    temp_meta.replace(meta_path)  # final marker, only after the whole extraction
    print(json.dumps({k: v for k, v in result.items() if k != "source"}, indent=2, allow_nan=False))
