"""Conservative spherical geographic reduction; only reduced blocks persist.

Density is an all-destination-cell area mean of the observed soil contribution.
Zero with validity=0 is UNKNOWN, never a measured absence or an ocean classifier.
PixelIsPoint samples use explicitly documented midpoint cell support, clipped to
the requested geographic domain. This reconstruction is not source subpixel data.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from range_client import RangeClient

RADIUS_M = 6_371_000.0
SCHEMA = "soil-conservative-v1"


def source_edges(info):
    if info.raster_type not in (1, 2):
        raise ValueError("Explicit GeoTIFF RasterTypeGeoKey (area or point) required")
    if info.model_type != 2 or info.geographic_crs != 4326:
        raise ValueError("Only explicit geographic EPSG:4326 rasters are supported")
    if 34264 in info.raw_entries:
        raise ValueError("Rotated/transformed rasters are unsupported")
    if not info.model_pixel_scale or not info.model_tiepoint:
        raise ValueError("Missing source georeferencing")
    dx, dy, _ = info.model_pixel_scale
    i, j, _, x, y, _ = info.model_tiepoint
    if not (dx > 0 and dy > 0):
        raise ValueError("Positive pixel scales required")
    shift = 0.5 if info.raster_type == 2 else 0.0
    return (x + (np.arange(info.width + 1) - i - shift) * dx,
            y - (np.arange(info.height + 1) - j - shift) * dy)


def _longitude_integrals(values, source_x, dest_x):
    """Exact piecewise-constant integration, including split native pixels."""
    widths = np.diff(source_x)
    prefix = np.pad(np.cumsum(values * widths, axis=1), ((0, 0), (1, 0)))
    q = np.clip(dest_x, source_x[0], source_x[-1])
    index = np.minimum(np.searchsorted(source_x, q, side="right") - 1, values.shape[1] - 1)
    at_edges = prefix[:, index] + values[:, index] * (q - source_x[index])
    return np.diff(at_edges, axis=1) * (np.pi / 180)


def _destination_rows(source_y, start, stop, dest_y):
    rows = np.flatnonzero((dest_y[:-1] > source_y[stop]) & (dest_y[1:] < source_y[start]))
    return (int(rows[0]), int(rows[-1]) + 1) if rows.size else (0, 0)


def _reduce(block, sx, sy, dx, dy, start, first, last, nodata):
    valid = np.isfinite(block)
    if nodata is not None and np.isfinite(nodata):
        valid &= block != nodata
    if np.any(block[valid] < 0):
        raise ValueError("Negative source values require an explicit no-data convention")
    filled = np.where(valid, block, 0)
    top = np.minimum(dy[first:last, None], sy[None, start:start + len(block)])
    bottom = np.maximum(dy[first + 1:last + 1, None], sy[None, start + 1:start + len(block) + 1])
    weights = np.where(top > bottom, np.sin(np.deg2rad(top)) - np.sin(np.deg2rad(bottom)), 0)
    reduced = np.stack([weights @ _longitude_integrals(v, sx, dx)
                        for v in (filled, valid.astype(float), np.ones_like(block))])
    # Independent native-cell integral over the requested domain (not derived
    # from the reduced output), retained for conservation accounting on resume.
    widths = np.maximum(0, np.minimum(sx[1:], dx[-1]) - np.maximum(sx[:-1], dx[0])) * np.pi / 180
    t = np.minimum(sy[start:start + len(block)], dy[0])
    b = np.maximum(sy[start + 1:start + len(block) + 1], dy[-1])
    lat_weights = np.where(t > b, np.sin(np.deg2rad(t)) - np.sin(np.deg2rad(b)), 0)
    native_mass = float(lat_weights @ (filled @ widths))
    return reduced, native_mass


def _read_checkpoint(path, shape):
    try:
        marker = json.loads(path.with_suffix(".json").read_text())
        # Bound even corrupted input before loading/decompressing it.
        if path.stat().st_size > np.prod(shape) * 8 + 8192:
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() != marker["sha256"]:
            return None
        with np.load(path, allow_pickle=False) as saved:
            reduced, native = saved["reduced"], float(saved["native_mass"])
            if reduced.shape != shape or not np.isfinite(reduced).all() or not np.isfinite(native):
                return None
            return reduced, native
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _save_checkpoint(path, reduced, native_mass):
    # Completion marker is published last; interrupted files are never resumed.
    temp = path.with_suffix(".npz.part")
    with temp.open("wb") as f:
        np.savez_compressed(f, reduced=reduced, native_mass=np.array(native_mass))
    temp.replace(path)
    marker = path.with_suffix(".json")
    pending = marker.with_suffix(".json.part")
    pending.write_text(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    pending.replace(marker)


def extract_conservative(url, info, lon_edges, lat_edges, *, cache_dir,
                         source_version, client=None, block_rows=32,
                         row_start=0, row_stop=None, cache_budget_bytes=80 * 1024**2,
                         progress=False):
    """Read every selected row, fail on unresolved errors, resume verified blocks.

    Omit row_start/row_stop for full coverage. Partial results carry complete=False
    and processed_fraction; they cannot be mistaken for publishable full products.
    source_version must identify an immutable source revision (e.g. bitstream UUID).
    Changing source/layout/grid/block configuration produces a separate namespace.
    """
    sx, sy = source_edges(info)
    dx, dy = np.asarray(lon_edges, dtype=float), np.asarray(lat_edges, dtype=float)
    if (dx.ndim != 1 or dy.ndim != 1 or len(dx) < 2 or len(dy) < 2
            or not np.isfinite(dx).all() or not np.isfinite(dy).all()
            or not np.all(np.diff(dx) > 0) or not np.all(np.diff(dy) < 0)
            or dx[0] < -180 or dx[-1] > 180 or dy[0] > 90 or dy[-1] < -90):
        raise ValueError("Destination edges must be ordered geographic cell boundaries")
    if not source_version:
        raise ValueError("Immutable source version required for resumable extraction")
    stop = info.height if row_stop is None else row_stop
    if not (0 <= row_start < stop <= info.height and block_rows > 0):
        raise ValueError("Invalid row selection")
    stride = info.width * info.bits_per_sample // 8
    if (info.compression != 1 or info.samples_per_pixel != 1 or info.rows_per_strip != 1
            or info.sample_format != 3 or info.bits_per_sample not in (32, 64)
            or len(info.strip_offsets) != info.height or len(info.strip_byte_counts) != info.height
            or any(n != stride for n in info.strip_byte_counts)
            or any(b - a != stride for a, b in zip(info.strip_offsets, info.strip_offsets[1:]))):
        raise ValueError("Expected contiguous uncompressed single-channel float rows")
    if block_rows * stride > 32 * 1024**2:
        raise ValueError("Native block exceeds 32 MiB memory/request bound")
    blocks = []
    for a in range(row_start, stop, block_rows):
        b = min(a + block_rows, stop)
        first, last = _destination_rows(sy, a, b, dy)
        blocks.append((a, b, first, last))
    estimate = sum(3 * (last - first) * (len(dx) - 1) * 8 + 8192 for _, _, first, last in blocks)
    if estimate > cache_budget_bytes:
        raise ValueError(f"Reduced checkpoint budget exceeded: estimate={estimate}, budget={cache_budget_bytes}")
    client = client or RangeClient(url)
    before_probe = client.bytes_fetched
    if getattr(client, "source_identity", None) is None:
        # The standalone API may receive a parsed TiffInfo but no primed client.
        # Establish validated HTTP identity even when every block is cached.
        client.fetch(0, 15)
    if getattr(client, "source_identity", None) is None:
        raise ValueError("Verified source identity required before checkpoint selection")
    if info.source_identity is None or tuple(info.source_identity) != tuple(client.source_identity):
        raise ValueError("TIFF header source identity is missing or stale; reparse the header with the current client")
    identity_probe_bytes = client.bytes_fetched - before_probe
    identity = dict(schema=SCHEMA, url=url, source_version=source_version,
                    http_identity=client.source_identity,
                    width=info.width, height=info.height, dtype=info.byte_order + info.dtype_str,
                    raster_type=info.raster_type, geographic_crs=info.geographic_crs,
                    scale=info.model_pixel_scale, tiepoint=info.model_tiepoint,
                    nodata=str(info.nodata), block_rows=block_rows,
                    lon_edges=dx.tolist(), lat_edges=dy.tolist(),
                    layout_sha256=hashlib.sha256(np.array(info.strip_offsets, dtype="<u8").tobytes()).hexdigest())
    serialized = json.dumps(identity, sort_keys=True)
    key = hashlib.sha256(serialized.encode()).hexdigest()
    folder = Path(cache_dir) / key
    # Shared budget across source/config namespaces. Existing user data stays;
    # budget exhaustion is an explicit error, never automatic eviction.
    existing = sum(p.stat().st_size for p in Path(cache_dir).rglob("*") if p.is_file())
    replaced = 0
    for a, b, _, _ in blocks:
        for suffix in (".npz", ".json", ".npz.part", ".json.part"):
            p = folder / f"block-{a:06d}-{b:06d}{suffix}"
            if p.exists():
                replaced += p.stat().st_size
    manifest = folder / "identity.json"
    if manifest.exists():
        replaced += manifest.stat().st_size
    if existing - replaced + estimate + len(serialized.encode()) > cache_budget_bytes:
        raise ValueError("Shared reduced checkpoint budget exceeded; retain existing data and choose a smaller product")
    folder.mkdir(parents=True, exist_ok=True)
    manifest.write_text(serialized)
    initial_bytes = client.bytes_fetched
    sums = np.zeros((3, len(dy) - 1, len(dx) - 1))
    native_mass = 0.0
    hits = 0
    for a, b, first, last in blocks:
        path = folder / f"block-{a:06d}-{b:06d}.npz"
        shape = (3, last - first, len(dx) - 1)
        cached = _read_checkpoint(path, shape)
        if cached is None:
            raw = client.fetch(info.strip_offsets[a], info.strip_offsets[b - 1] + stride - 1)
            block = np.frombuffer(raw, dtype=info.byte_order + info.dtype_str).reshape(b - a, info.width)
            reduced, native = _reduce(block, sx, sy, dx, dy, a, first, last, info.nodata)
            _save_checkpoint(path, reduced, native)
        else:
            reduced, native = cached
            hits += 1
        sums[:, first:last] += reduced
        native_mass += native
        if progress:
            print(json.dumps(dict(event="block", row_start=a, row_stop=b,
                                  cached=cached is not None,
                                  bytes_fetched=client.bytes_fetched - initial_bytes)),
                  file=sys.stderr, flush=True)
    area = (np.sin(np.deg2rad(dy[:-1])) - np.sin(np.deg2rad(dy[1:])))[:, None] * np.deg2rad(np.diff(dx))[None, :]
    output_mass = float(sums[0].sum())
    return dict(density=sums[0] / area, valid_model_fraction=np.clip(sums[1] / area, 0, 1),
                processed_fraction=np.clip(sums[2] / area, 0, 1), lon_edges=dx, lat_edges=dy,
                complete=row_start == 0 and stop == info.height, rows_touched=stop - row_start,
                row_coverage_actual=(stop - row_start) / info.height,
                row_start=row_start, row_stop=stop, cache_hits=hits,
                bytes_fetched=client.bytes_fetched - initial_bytes,
                identity_probe_bytes=identity_probe_bytes,
                native_integral=native_mass * RADIUS_M**2,
                output_integral=output_mass * RADIUS_M**2,
                conservation_relative_error=abs(output_mass - native_mass) / max(abs(native_mass), 1e-300),
                cache_identity=key, cache_estimated_max_bytes=estimate, source=identity)
