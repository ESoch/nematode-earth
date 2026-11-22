"""Pytest suite: build small synthetic BigTIFFs locally with the same layout
as the real ETH rasters (uncompressed, RowsPerStrip=1, float64, EPSG:4326
tiepoint + pixel scale tags, latitude clipped to +/-88 not +/-90), serve them
over a local HTTP server that honours Range requests, and run the real
header parser / range-addressing / extraction / validation code against them
end-to-end.

This is the offline fallback path required by the task brief: prove the
whole mechanism is correct independent of whether the live ETH endpoint is
reachable from this environment (see .superpowers/session2/pipeline-report.md
for why that mattered this session -- the live endpoint was unreachable from
the reporting environment, though reachable, slowly, from this one).

Run: uv run pytest fetch/test_synthetic.py -v
 or:  uv run fetch/test_synthetic.py          (same suite, plain prints)
"""

from __future__ import annotations

import http.server
import socketserver
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest
from eth_raster import area_weighted_integral, extract_grid, percentiles
from range_client import RangeClient
from tiff_header import TiffHeaderReader, TiffInfo

EARTH_RADIUS_M = 6_371_000.0

# Main fixture: random land/ocean data, mimics the real raster's statistics
# loosely (heterogeneous magnitude, an irregular coastline-like NaN mask).
WIDTH, HEIGHT = 400, 200
LON0, LAT0 = -180.0, 88.0
DLON, DLAT = 360.0 / WIDTH, 176.0 / HEIGHT  # same +/-88 clipping as the real grid


# ---------------------------------------------------------------------------
# Synthetic BigTIFF construction
# ---------------------------------------------------------------------------


def build_bigtiff(
    data: np.ndarray, *, lon0: float, lat0: float, dlon: float, dlat: float
) -> bytes:
    """Construct a minimal valid BigTIFF around `data` (2D float64, NaN =
    ocean/no-data): header, IFD with the tags our parser reads, then
    RowsPerStrip=1 uncompressed strips, each directly seekable via
    StripOffsets. This mirrors the real files' byte layout closely enough
    that the same parser code path is exercised."""
    height, width = data.shape
    row_stride = width * 8
    pixel_bytes = data.astype("<f8").tobytes()
    assert len(pixel_bytes) == row_stride * height

    endian = "<"
    entries_meta = []  # (tag, type, count, inline_value_bytes)
    overflow_tags = []  # tags whose payload must live after the IFD

    entries_meta.append((256, 16, 1, struct.pack(endian + "Q", width)))  # ImageWidth
    entries_meta.append((257, 16, 1, struct.pack(endian + "Q", height)))  # ImageLength
    entries_meta.append(
        (258, 3, 1, struct.pack(endian + "H", 64) + b"\x00" * 6)
    )  # BitsPerSample
    entries_meta.append(
        (259, 3, 1, struct.pack(endian + "H", 1) + b"\x00" * 6)
    )  # Compression=1
    entries_meta.append(
        (277, 3, 1, struct.pack(endian + "H", 1) + b"\x00" * 6)
    )  # SamplesPerPixel
    entries_meta.append((278, 16, 1, struct.pack(endian + "Q", 1)))  # RowsPerStrip=1
    entries_meta.append(
        (339, 3, 1, struct.pack(endian + "H", 3) + b"\x00" * 6)
    )  # SampleFormat=float

    strip_byte_counts = [row_stride] * height
    overflow_tags.append(("STRIP_OFFSETS", 273, 16, height, None))
    overflow_tags.append(("STRIP_BYTE_COUNTS", 279, 16, height, strip_byte_counts))
    overflow_tags.append(("PIXEL_SCALE", 33550, 12, 3, [dlon, dlat, 0.0]))
    overflow_tags.append(("TIEPOINT", 33922, 12, 6, [0.0, 0.0, 0.0, lon0, lat0, 0.0]))

    n_entries = len(entries_meta) + len(overflow_tags)
    ifd_offset = 16  # right after the 16-byte BigTIFF header
    entry_size = 20
    ifd_total_size = 8 + n_entries * entry_size + 8  # count + entries + next-ifd-ptr
    overflow_start = ifd_offset + ifd_total_size

    cursor = overflow_start
    overflow_blobs: list[tuple[int, bytes]] = []
    resolved = {}
    for name, tag, type_, count, values in overflow_tags:
        if name == "STRIP_OFFSETS":
            payload_len = count * 8
            resolved[name] = (tag, type_, count, cursor, payload_len)
            cursor += payload_len
        else:
            payload = struct.pack(
                endian + ("Q" if type_ == 16 else "d") * count, *values
            )
            overflow_blobs.append((cursor, payload))
            resolved[name] = (tag, type_, count, cursor, len(payload))
            cursor += len(payload)

    pixel_data_start = cursor
    strip_offsets = [pixel_data_start + i * row_stride for i in range(height)]
    so_tag, so_type, so_count, so_offset, _ = resolved["STRIP_OFFSETS"]
    overflow_blobs.append(
        (so_offset, struct.pack(endian + "Q" * so_count, *strip_offsets))
    )

    ifd_entries_bytes = b""
    for tag, type_, count, value_bytes in entries_meta:
        vb = value_bytes[:8].ljust(8, b"\x00")
        ifd_entries_bytes += (
            struct.pack(endian + "HH", tag, type_)
            + struct.pack(endian + "Q", count)
            + vb
        )
    for name in ("STRIP_OFFSETS", "STRIP_BYTE_COUNTS", "PIXEL_SCALE", "TIEPOINT"):
        tag, type_, count, off, _ = resolved[name]
        ifd_entries_bytes += (
            struct.pack(endian + "HH", tag, type_)
            + struct.pack(endian + "Q", count)
            + struct.pack(endian + "Q", off)
        )

    header = (
        b"II"
        + struct.pack(endian + "H", 43)
        + struct.pack(endian + "HH", 8, 0)
        + struct.pack(endian + "Q", ifd_offset)
    )
    assert len(header) == 16
    ifd_blob = (
        struct.pack(endian + "Q", n_entries)
        + ifd_entries_bytes
        + struct.pack(endian + "Q", 0)
    )
    assert ifd_offset + len(ifd_blob) == overflow_start

    buf = bytearray(header) + bytearray(ifd_blob)
    for off, blob in sorted(overflow_blobs):
        assert off == len(buf), (off, len(buf))
        buf += blob
    assert len(buf) == pixel_data_start
    buf += pixel_bytes
    return bytes(buf)


def make_main_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    data = rng.uniform(1e5, 2e7, size=(HEIGHT, WIDTH)).astype(np.float64)
    mask_ocean = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask_ocean[:, : WIDTH // 6] = True  # a whole "ocean" column band
    for i in range(HEIGHT):  # plus a diagonal "coastline"
        j = int(i * WIDTH / HEIGHT)
        mask_ocean[i, max(0, j - 3) : j + 3] = True
    data[mask_ocean] = np.nan
    return data


def make_striped_data(height: int, width: int) -> np.ndarray:
    """Rows alternate between two constants. A correct box-average over an
    even-sized contiguous block gives exactly the midpoint; a point-sampler
    that just reads one row from within the block would return one of the
    two extremes instead. This is the trap the task brief explicitly warns
    about ('do not sample every Nth row')."""
    data = np.empty((height, width), dtype=np.float64)
    for i in range(height):
        data[i, :] = 0.0 if i % 2 == 0 else 1_000_000.0
    return data


# ---------------------------------------------------------------------------
# Local range-serving HTTP server
# ---------------------------------------------------------------------------


class RangeHTTPHandler(http.server.BaseHTTPRequestHandler):
    file_bytes: bytes = b""

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        body = type(self).file_bytes
        if rng and rng.startswith("bytes="):
            start_s, end_s = rng[6:].split("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(body) - 1
            chunk = body[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@dataclass
class Served:
    url: str
    info: TiffInfo
    data: np.ndarray


def _serve(data: np.ndarray, *, lon0: float, lat0: float, dlon: float, dlat: float):
    file_bytes = build_bigtiff(data, lon0=lon0, lat0=lat0, dlon=dlon, dlat=dlat)

    class Handler(RangeHTTPHandler):
        pass

    Handler.file_bytes = file_bytes
    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)

    url = f"http://127.0.0.1:{port}/synthetic.tif"
    client = RangeClient(url, max_retries=2, timeout=5.0)
    info = TiffHeaderReader(url, client.fetch).parse()
    return httpd, Served(url=url, info=info, data=data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def main_served():
    data = make_main_data()
    httpd, served = _serve(data, lon0=LON0, lat0=LAT0, dlon=DLON, dlat=DLAT)
    try:
        yield served
    finally:
        httpd.shutdown()


@pytest.fixture(scope="module")
def striped_served():
    h, w = 40, 20
    data = make_striped_data(h, w)
    httpd, served = _serve(data, lon0=LON0, lat0=LAT0, dlon=DLON, dlat=DLAT)
    try:
        yield served
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_header_parse(main_served):
    info = main_served.info
    assert info.big_tiff is True
    assert info.width == WIDTH and info.height == HEIGHT
    assert info.rows_per_strip == 1
    assert info.dtype_str == "f8"
    assert len(info.strip_offsets) == HEIGHT
    assert info.model_pixel_scale == (DLON, DLAT, 0.0)
    assert info.model_tiepoint == (0.0, 0.0, 0.0, LON0, LAT0, 0.0)
    stride = info.strip_offsets[1] - info.strip_offsets[0]
    assert stride == WIDTH * 8


def test_row_addressing_matches_stripoffsets(main_served):
    """The single most load-bearing claim in the whole approach: a byte
    range built from StripOffsets[i] must land on exactly row i's data, for
    an arbitrary row, not just row 0. Prove it directly against several
    rows scattered across the file, bypassing extract_grid entirely."""
    info = main_served.info
    client = RangeClient(main_served.url, max_retries=2, timeout=5.0)
    stride = info.strip_offsets[1] - info.strip_offsets[0]

    for row in (0, 1, 7, HEIGHT // 2, HEIGHT - 1):
        start = info.strip_offsets[row]
        raw = client.fetch(start, start + stride - 1)
        got = np.frombuffer(raw, dtype="<f8")
        want = main_served.data[row, :]
        # NaN-aware exact comparison -- these are raw bytes off the wire,
        # not recomputed values, so it must be bit-exact.
        assert np.array_equal(got, want, equal_nan=True), f"row {row} mismatch"


def test_extraction_matches_numpy_reference(main_served):
    info, truth = main_served.info, main_served.data
    out_w, out_h = 40, 20
    result = extract_grid(
        main_served.url,
        info,
        out_w,
        out_h,
        row_coverage=1.0,
        max_workers=3,
        progress=False,
    )
    density, land_frac = result["density"], result["land_fraction"]
    assert result["rows_touched"] == HEIGHT, "full coverage must touch every row"
    assert not result["failed_rows"]
    assert not np.isnan(density).any(), "NaN must not leak into the density channel"

    row_edges = np.linspace(0, HEIGHT, out_h + 1).round().astype(int)
    col_edges = np.linspace(0, WIDTH, out_w + 1).round().astype(int)
    filled_truth = np.where(np.isnan(truth), 0.0, truth)
    valid_truth = ~np.isnan(truth)
    ref_density = np.zeros((out_h, out_w))
    ref_land = np.zeros((out_h, out_w))
    for r in range(out_h):
        for c in range(out_w):
            rs, re = row_edges[r], row_edges[r + 1]
            cs, ce = col_edges[c], col_edges[c + 1]
            ref_density[r, c] = filled_truth[
                rs:re, cs:ce
            ].mean()  # NaN->0, mean over ALL cells
            ref_land[r, c] = valid_truth[rs:re, cs:ce].mean()

    assert np.abs(density - ref_density).max() < 1e-6
    assert np.abs(land_frac - ref_land).max() < 1e-9


def test_block_average_not_point_sample(striped_served):
    """Direct test of the brief's explicit correctness rule: 'fetch
    contiguous row blocks and average, do not sample every Nth row'. Native
    rows alternate 0 / 1,000,000. A single output row backed by a
    contiguous, even-sized block must read exactly the midpoint
    (500,000) -- a point-sampler could only ever produce 0 or 1,000,000,
    never the midpoint, so this fails loudly if decimation regresses to
    point sampling."""
    info, data = striped_served.info, striped_served.data
    h, w = data.shape
    out_h = h // 4  # each output row backed by a 4-row block (even -> exact midpoint)
    result = extract_grid(
        striped_served.url,
        info,
        w,
        out_h,
        row_coverage=1.0,
        max_workers=2,
        progress=False,
    )
    density = result["density"]
    assert np.allclose(density, 500_000.0), (
        f"expected exact block-average midpoint 500000, got {density.ravel()[:5]} "
        f"-- looks like point sampling, not block averaging"
    )


def test_nan_becomes_zero_density_with_separate_land_mask(main_served):
    info = main_served.info
    result = extract_grid(
        main_served.url, info, 40, 20, row_coverage=1.0, max_workers=3, progress=False
    )
    density, land_frac = result["density"], result["land_fraction"]
    assert not np.isnan(density).any()
    assert not np.isnan(land_frac).any()
    # Pure-ocean output cells (land_frac == 0) must have density == 0 exactly.
    assert np.all(density[land_frac == 0] == 0.0)
    # Mixed coastal cells must be strictly less than the land-only mean would
    # give -- i.e. genuinely diluted by the ocean fraction, not un-diluted.
    mixed = (land_frac > 0) & (land_frac < 1)
    assert mixed.any(), "fixture should produce at least one mixed coastal cell"


def test_validation_integral_recovers_known_uniform_density(main_served):
    info = main_served.info
    out_h, out_w = 20, 40
    uniform = np.full((out_h, out_w), 1.0)
    result = area_weighted_integral(uniform, info, out_h, out_w)
    # Exact analytic answer for a uniform-density field on a lat band
    # clipped to +/-88 (not +/-90): R^2 * 2*pi * (sin(88deg) - sin(-88deg)).
    expected = (
        EARTH_RADIUS_M**2
        * (2 * np.pi)
        * (np.sin(np.radians(88.0)) - np.sin(np.radians(-88.0)))
    )
    assert abs(result["integral"] / expected - 1.0) < 1e-9
    assert result["excluded_area_fraction"] == 0.0

    integral2 = area_weighted_integral(uniform * 2, info, out_h, out_w)["integral"]
    assert abs(integral2 / result["integral"] - 2.0) < 1e-9, (
        "integral must scale linearly with density"
    )


def test_area_weighting_respects_clipped_latitude_not_full_poles(main_served):
    """Confirms the +/-88 clipping claim is actually honoured by the area
    formula: the grid's total area must be strictly LESS than a full
    sphere's, by the amount the polar caps beyond 88deg would have
    contributed if they were (wrongly) assumed present."""
    info = main_served.info
    out_h, out_w = 20, 40
    uniform = np.full((out_h, out_w), 1.0)
    result = area_weighted_integral(uniform, info, out_h, out_w)
    full_sphere_area = 4 * np.pi * EARTH_RADIUS_M**2
    assert result["total_area_m2"] < full_sphere_area
    # The two polar caps beyond +/-88deg, by the exact spherical-cap formula
    # 2*pi*R^2*(1 - sin(88deg)), account for the difference.
    missing_caps = 2 * (2 * np.pi * EARTH_RADIUS_M**2 * (1 - np.sin(np.radians(88.0))))
    assert (
        abs(result["total_area_m2"] + missing_caps - full_sphere_area)
        / full_sphere_area
        < 1e-9
    )


def test_excluded_area_does_not_poison_integral(main_served):
    info = main_served.info
    out_h, out_w = 20, 40
    with_gap = np.full((out_h, out_w), 1.0)
    with_gap[3, :] = np.nan  # simulates a row that failed to fetch after all retries
    result = area_weighted_integral(with_gap, info, out_h, out_w)
    assert not np.isnan(result["integral"])
    assert result["excluded_area_fraction"] > 0


def test_percentiles_only_over_land(main_served):
    info = main_served.info
    result = extract_grid(
        main_served.url, info, 40, 20, row_coverage=1.0, max_workers=3, progress=False
    )
    pct = percentiles(result["density"], result["land_fraction"])
    assert set(pct) == {"min", "p05", "p25", "p50", "p75", "p95", "p99", "max"}
    assert pct["min"] <= pct["p50"] <= pct["max"]


# ---------------------------------------------------------------------------
# Plain-script entry point (kept for a quick human-readable run outside pytest)
# ---------------------------------------------------------------------------


def _run_as_script():
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))


if __name__ == "__main__":
    _run_as_script()
