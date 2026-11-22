#!/usr/bin/env python3
"""Extract a decimated global grid from the ETH Research Collection nematode
abundance GeoTIFFs via HTTP range requests, without downloading the full
multi-GB raster.

Usage:
    uv run fetch/eth_raster.py verify <bitstream-url>
    uv run fetch/eth_raster.py extract <bitstream-url> --out-width 512 --out-height 256 \
        --row-coverage 0.025 --out-dir out/

Network reachability of research-collection.ethz.ch is INCONSISTENT across
environments -- confirmed by direct comparison in the same session: one
environment reached it successfully (used to produce the real results in
pipeline-report.md, with a ~7% per-request retry rate and per-connection
throughput anywhere from ~70 KB/s to outright resets), while another got
`HTTP 000` / connection-refused on every attempt to the same host. Both
observations are real; treat this endpoint as unreliable rather than either
"up" or "down". `fetch/test_synthetic.py` exercises this whole module
against a local fixture server so its correctness does not depend on ETH
being reachable at all. See .superpowers/session2/pipeline-report.md for
the full evidence (header bytes, throughput, validation ratio) from the run
that did succeed, and exactly what is still unverified as a result.

Correctness rule (per the design spec / task brief): decimation must be done
by averaging CONTIGUOUS BLOCKS of native pixels, never by point-sampling
every Nth row/column -- point-decimating a high-frequency random-forest
prediction surface aliases badly and breaks the mass-conservation the
downstream pipeline depends on.

Column averaging here is always full-coverage (every one of the ~43201
native columns in a fetched row contributes to its output column) because
once a row is fetched, averaging extra columns is free -- no extra bytes
over the wire. Row averaging is the one axis that costs bandwidth per
native row touched, and is controlled by --row-block / --row-coverage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from range_client import RangeClient
from tiff_header import TiffHeaderReader, TiffInfo

EARTH_RADIUS_M = 6_371_000.0  # IUGG mean radius; adequate for a sanity-check integral
PUBLISHED_TOTAL = 4.4e20


# ---------------------------------------------------------------------------
# Header verification
# ---------------------------------------------------------------------------


def verify_header(url: str) -> TiffInfo:
    client = RangeClient(url)
    reader = TiffHeaderReader(url, client.fetch)
    info = reader.parse()

    print(f"url: {url}")
    print(f"byte_order: {'little' if info.byte_order == '<' else 'big'}-endian")
    print(f"big_tiff: {info.big_tiff}")
    print(f"ifd_offset: {info.ifd_offset}")
    print(f"width x height: {info.width} x {info.height}")
    print(
        f"bits_per_sample: {info.bits_per_sample}  sample_format: {info.sample_format} "
        f"({info.dtype_str})"
    )
    print(f"samples_per_pixel: {info.samples_per_pixel}")
    print(f"compression: {info.compression} (1 = uncompressed)")
    print(f"rows_per_strip: {info.rows_per_strip}")
    print(f"num_strips: {len(info.strip_offsets)}")
    print(f"strip_offsets[0:3]: {info.strip_offsets[:3]}")
    print(f"strip_byte_counts[0:3]: {info.strip_byte_counts[:3]}")
    print(f"model_pixel_scale: {info.model_pixel_scale}")
    print(f"model_tiepoint: {info.model_tiepoint}")

    # Verify the striping is what makes range-addressing valid: RowsPerStrip==1
    # and constant stride between consecutive strip offsets.
    if info.rows_per_strip != 1:
        print(
            f"WARNING: RowsPerStrip={info.rows_per_strip}, not 1 -- "
            f"per-row addressing assumption does not hold as expected."
        )
    if len(info.strip_offsets) >= 3:
        stride_a = info.strip_offsets[1] - info.strip_offsets[0]
        stride_b = info.strip_offsets[2] - info.strip_offsets[1]
        last_stride = info.strip_offsets[-1] - info.strip_offsets[-2]
        print(f"row_stride (from offsets[0],[1]): {stride_a} bytes")
        print(f"row_stride (from offsets[1],[2]): {stride_b} bytes")
        print(f"row_stride (from last two offsets): {last_stride} bytes")
        expected = info.width * info.bits_per_sample // 8
        print(f"expected stride (width * bytes_per_sample): {expected} bytes")
        if not (stride_a == stride_b == last_stride == expected):
            print(
                "WARNING: strip stride is not constant / does not match "
                "width*bytes_per_sample -- contiguous-row assumption may not hold "
                "for arbitrary row ranges."
            )
    return info


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _row_band_edges(n_in: int, n_out: int) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) index bands partitioning n_in rows/cols
    into n_out bands as evenly as possible (used for full-coverage runs and
    for picking each output row's home span even under partial coverage)."""
    edges = np.linspace(0, n_in, n_out + 1)
    bands = []
    for i in range(n_out):
        a = int(round(edges[i]))
        b = int(round(edges[i + 1]))
        if b <= a:
            b = a + 1
        bands.append((a, min(b, n_in)))
    return bands


def _fetch_row_block(
    client: RangeClient, info: TiffInfo, row_start: int, row_count: int
) -> np.ndarray:
    """Fetch `row_count` contiguous full-width rows starting at `row_start`,
    return as a 2D float64 array [row_count, width] (NaN preserved)."""
    stride = info.width * info.bits_per_sample // 8
    if (info.compression != 1 or info.samples_per_pixel != 1 or info.rows_per_strip != 1
            or info.sample_format != 3 or info.bits_per_sample not in (32, 64)
            or len(info.strip_offsets) != info.height or len(info.strip_byte_counts) != info.height
            or any(n != stride for n in info.strip_byte_counts)
            or any(b - a != stride for a, b in zip(info.strip_offsets, info.strip_offsets[1:]))):
        raise ValueError("Unsupported legacy strip layout: expected contiguous single-channel float rows")
    byte_start = info.strip_offsets[row_start]
    dt = np.dtype(f"{info.byte_order}{info.dtype_str}")
    # Preserve the same native samples and legacy averaging, but never ask the
    # bounded HTTP client for an entire >32MiB output-row band at once. Decode
    # each sample-aligned subrequest straight into the final float64 array.
    samples_per_request = client.max_response_bytes // dt.itemsize
    if samples_per_request < 1:
        raise ValueError("Response bound is smaller than one native sample")
    out = np.empty(row_count * info.width, dtype=np.float64)
    for offset in range(0, out.size, samples_per_request):
        count = min(samples_per_request, out.size - offset)
        start = byte_start + offset * dt.itemsize
        raw = client.fetch(start, start + count * dt.itemsize - 1)
        out[offset:offset + count] = np.frombuffer(raw, dtype=dt)
    return out.reshape(row_count, info.width)


def extract_grid(
    url: str,
    info: TiffInfo,
    out_width: int,
    out_height: int,
    row_coverage: float = 1.0,
    max_workers: int = 4,
    progress: bool = True,
) -> dict:
    """Box-average-decimate the raster to out_width x out_height.

    row_coverage in (0, 1]: fraction of each output row's native-row span
    that is actually fetched and averaged, as a contiguous block centered in
    that span. 1.0 = true full-coverage decimation (every native row
    contributes to exactly one output row -- the mass-conserving, correct
    thing to do, but reads the entire file). Values < 1.0 trade rigor for
    bandwidth: still block-averaged (never single-row point sampling), but a
    systematic/stratified sample rather than full coverage. This run must
    report which mode it used.

    Column averaging is ALWAYS full-coverage regardless of row_coverage: once
    a row is on the wire, every column in it is used.
    """
    assert 0 < row_coverage <= 1.0
    row_bands = _row_band_edges(info.height, out_height)

    density = np.zeros((out_height, out_width), dtype=np.float64)
    land_frac = np.zeros((out_height, out_width), dtype=np.float64)
    rows_touched = 0

    col_edges = np.linspace(0, info.width, out_width + 1).astype(np.int64)

    def col_average(row_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """row_block: [r, width] float64 with NaN for ocean.
        Returns (density_row[out_width], land_frac_row[out_width])."""
        r = row_block.shape[0]
        d_out = np.empty(out_width, dtype=np.float64)
        l_out = np.empty(out_width, dtype=np.float64)
        valid = ~np.isnan(row_block)
        filled = np.where(valid, row_block, 0.0)
        # column-wise cumulative sums for O(width) binning instead of O(out_width*avg_bin)
        csum_d = np.concatenate([[0.0], np.cumsum(filled.sum(axis=0))])
        csum_v = np.concatenate([[0.0], np.cumsum(valid.sum(axis=0))])
        for c in range(out_width):
            c0, c1 = col_edges[c], col_edges[c + 1]
            n_valid = csum_v[c1] - csum_v[c0]
            total_cells = r * (c1 - c0)
            d_sum = csum_d[c1] - csum_d[c0]
            # Mass-conserving: average over ALL cells (ocean counted as 0), not
            # just land cells. `_perSquareMeter` already bakes land-fraction
            # dilution into each native pixel; averaging only the land cells
            # here would UN-dilute mixed coastal boxes and inflate the global
            # integral. See pipeline-report.md for the reasoning.
            d_out[c] = (d_sum / total_cells) if total_cells > 0 else 0.0
            l_out[c] = (n_valid / total_cells) if total_cells > 0 else 0.0
        return d_out, l_out

    tls = {}

    def _thread_local_client():
        import threading

        tid = threading.get_ident()
        if tid not in tls:
            tls[tid] = RangeClient(url)
        return tls[tid]

    def work(i: int):
        band_start, band_end = row_bands[i]
        span = band_end - band_start
        block_rows = max(1, min(span, round(span * row_coverage)))
        # center the block within the band
        center = band_start + span // 2
        rs = max(band_start, center - block_rows // 2)
        rs = min(rs, band_end - block_rows)
        client = _thread_local_client()
        block = _fetch_row_block(client, info, rs, block_rows)
        d_row, l_row = col_average(block)
        return i, d_row, l_row, block_rows

    failed_rows: list[int] = []

    def run_pass(indices, workers):
        nonlocal rows_touched
        still_failed = []
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(work, i): i for i in indices}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    i, d_row, l_row, block_rows = fut.result()
                except Exception as exc:  # noqa: BLE001 -- report and continue
                    print(f"  WARNING: row {i} failed: {exc}", file=sys.stderr)
                    still_failed.append(i)
                    continue
                density[i, :] = d_row
                land_frac[i, :] = l_row
                rows_touched += block_rows
                done += 1
                if progress and (
                    done % max(1, len(indices) // 20 or 1) == 0 or done == len(indices)
                ):
                    elapsed = time.time() - t0
                    total_bytes = sum(c.bytes_fetched for c in tls.values())
                    print(
                        f"  [{done}/{len(indices)}] rows_touched={rows_touched} "
                        f"bytes={total_bytes / 1e6:.1f}MB elapsed={elapsed:.0f}s",
                        file=sys.stderr,
                    )
        return still_failed

    t_start = time.time()
    pending = list(range(out_height))
    # First pass at full concurrency, then a couple of low-concurrency retry
    # passes for whatever rows hit a connection that never recovered -- the
    # ETH server resets connections routinely; a fresh, less-contended retry
    # clears most of these (see pipeline-report.md).
    for pass_num, workers in enumerate([max_workers, 2, 1]):
        if not pending:
            break
        if pass_num > 0:
            print(
                f"  retry pass {pass_num}: {len(pending)} row(s) outstanding",
                file=sys.stderr,
            )
        pending = run_pass(pending, workers)

    failed_rows = pending
    if failed_rows:
        print(
            f"  WARNING: {len(failed_rows)} row(s) never succeeded after all "
            f"retry passes: {failed_rows}",
            file=sys.stderr,
        )
        for i in failed_rows:
            density[i, :] = np.nan
            land_frac[i, :] = np.nan

    total_bytes = sum(c.bytes_fetched for c in tls.values())
    total_requests = sum(c.requests_made for c in tls.values())
    total_retries = sum(c.retries_used for c in tls.values())
    elapsed = time.time() - t_start

    return {
        "density": density,
        "land_fraction": land_frac,
        "rows_touched": rows_touched,
        "row_coverage_actual": rows_touched / info.height,
        "bytes_fetched": total_bytes,
        "requests_made": total_requests,
        "retries_used": total_retries,
        "elapsed_s": elapsed,
        "failed_rows": failed_rows,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def area_weighted_integral(
    density: np.ndarray, info: TiffInfo, out_height: int, out_width: int
) -> dict:
    """Sum density[r,c] * pixel_area(r) over the grid, using exact spherical
    lat-band area (not a small-angle approximation): a band from lat_top to
    lat_bottom spanning the full 360 deg of longitude has area
    R^2 * (2*pi) * (sin(lat_top) - sin(lat_bottom)); divide by out_width for
    the per-pixel area in that row (equirectangular -> equal-area columns).

    Rows that failed to fetch (all-NaN, see extract_grid's failed_rows) are
    excluded from the sum rather than poisoning it with NaN; the excluded
    area fraction is returned alongside the integral so the caller can judge
    whether the result is trustworthy."""
    if info.model_tiepoint is None or info.model_pixel_scale is None:
        raise ValueError("Missing georeferencing tags; cannot compute pixel area")

    # ModelTiepointTag: (I, J, K, X, Y, Z) -- pixel (I,J) maps to (X,Y,Z)
    i0, j0, _k0, lon0, lat0, _z0 = info.model_tiepoint
    dlon, dlat, _dz = (
        info.model_pixel_scale
    )  # degrees per pixel; dlat is positive magnitude

    row_bands = _row_band_edges(info.height, out_height)
    total = 0.0
    total_area = 0.0
    excluded_area = 0.0
    for r, (a, b) in enumerate(row_bands):
        lat_top = lat0 - (a - j0) * dlat
        lat_bot = lat0 - (b - j0) * dlat
        band_area = (
            EARTH_RADIUS_M**2
            * math.radians(360.0)
            * (math.sin(math.radians(lat_top)) - math.sin(math.radians(lat_bot)))
        )
        pixel_area = band_area / out_width
        row_area = pixel_area * out_width
        total_area += row_area
        row = density[r, :]
        if np.isnan(row).any():
            excluded_area += row_area
            continue
        total += float(row.sum()) * pixel_area
    return {
        "integral": total,
        "total_area_m2": total_area,
        "excluded_area_m2": excluded_area,
        "excluded_area_fraction": (excluded_area / total_area) if total_area else 0.0,
    }


def percentiles(density: np.ndarray, land_frac: np.ndarray) -> dict:
    """Percentiles over LAND pixels only (land_fraction > 0), since ocean is
    a defined 0 by construction, not a sampled value -- including it would
    just report 'what fraction of the grid is ocean' at every percentile."""
    land_vals = density[land_frac > 0]
    if land_vals.size == 0:
        return {}
    qs = [0, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(land_vals, qs)
    return dict(
        zip(["min", "p05", "p25", "p50", "p75", "p95", "p99", "max"], vals.tolist())
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="parse and print TIFF header only")
    p_verify.add_argument("url")

    p_extract = sub.add_parser("extract", help="extract a decimated grid")
    p_extract.add_argument("url")
    p_extract.add_argument("--out-width", type=int, default=512)
    p_extract.add_argument("--out-height", type=int, default=256)
    p_extract.add_argument(
        "--row-coverage",
        type=float,
        default=1.0,
        help="fraction of each output row's native span to "
        "actually fetch (1.0 = full/correct decimation)",
    )
    p_extract.add_argument("--max-workers", type=int, default=4)
    p_extract.add_argument("--out-dir", type=Path, default=Path("out"))
    p_extract.add_argument(
        "--tag", type=str, default="grid", help="filename tag, e.g. total_persqm"
    )

    # Legacy extraction remains untouched; this path has a different, explicit
    # coverage/geographic contract and never overwrites legacy outputs.
    from conservative_cli import configure_parser, run_cli
    configure_parser(sub.add_parser("conservative", help="full-row conservative extraction with reduced checkpoints"))

    args = ap.parse_args()

    if args.cmd == "conservative":
        run_cli(args)
        return

    if args.cmd == "verify":
        verify_header(args.url)
        return

    if args.cmd == "extract":
        info = verify_header(args.url)
        print(
            f"\nExtracting {args.out_width}x{args.out_height} "
            f"(row_coverage={args.row_coverage}) ...",
            file=sys.stderr,
        )
        result = extract_grid(
            args.url,
            info,
            args.out_width,
            args.out_height,
            row_coverage=args.row_coverage,
            max_workers=args.max_workers,
        )
        density = result["density"]
        land_frac = result["land_fraction"]
        if result["failed_rows"] or not (np.isfinite(density).all() and np.isfinite(land_frac).all()):
            raise RuntimeError(f"Legacy extraction incomplete; refusing final products (failed rows: {result['failed_rows']})")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            args.out_dir / f"{args.tag}_density_{args.out_width}x{args.out_height}.npy",
            density.astype(np.float32),
        )
        np.save(
            args.out_dir
            / f"{args.tag}_landfrac_{args.out_width}x{args.out_height}.npy",
            land_frac.astype(np.float32),
        )

        integral_info = area_weighted_integral(
            density, info, args.out_height, args.out_width
        )
        integral = integral_info["integral"]
        pct = percentiles(density, land_frac)

        meta = {
            "source_url": args.url,
            "tiff": {
                "width": info.width,
                "height": info.height,
                "dtype": info.dtype_str,
                "big_tiff": info.big_tiff,
                "rows_per_strip": info.rows_per_strip,
                "model_pixel_scale": info.model_pixel_scale,
                "model_tiepoint": info.model_tiepoint,
            },
            "out_width": args.out_width,
            "out_height": args.out_height,
            "row_coverage_requested": args.row_coverage,
            "row_coverage_actual": result["row_coverage_actual"],
            "column_coverage": 1.0,
            "rows_touched": result["rows_touched"],
            "failed_rows": result["failed_rows"],
            "bytes_fetched": result["bytes_fetched"],
            "requests_made": result["requests_made"],
            "retries_used": result["retries_used"],
            "elapsed_s": result["elapsed_s"],
            "area_weighted_integral": integral,
            "integral_excluded_area_fraction": integral_info["excluded_area_fraction"],
            "published_total": PUBLISHED_TOTAL,
            "ratio_to_published": integral / PUBLISHED_TOTAL,
            "percentiles": pct,
        }
        with open(
            args.out_dir / f"{args.tag}_meta_{args.out_width}x{args.out_height}.json",
            "w",
        ) as f:
            json.dump(meta, f, indent=2)

        print("\n=== extraction summary ===")
        print(json.dumps(meta, indent=2))
        return


if __name__ == "__main__":
    main()
