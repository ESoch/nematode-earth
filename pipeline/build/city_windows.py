#!/usr/bin/env python3
"""Extract native-resolution (30 arc-second, ~1 km) nematode density
windows around ~20 cities from the real ETH `TotalNumber_perSquareMeter.tif`,
paired with matching GHSL built-up-fraction windows (already cached
locally from the elevation/urban-mask task, no new network needed for
that half).

Why per-row COLUMN-SLICED range fetches, not full-width row blocks:
`eth_raster.py`'s global extraction fetches full 345,608-byte rows because
it needs every column anyway. A city window needs only ~1 deg of
longitude (~100-150 native columns out of 43201) per row, so fetching
full rows here would pull ~300x more data than needed (a 1deg x 1deg,
120-row window is ~115 KB column-sliced vs ~41 MB full-row). Column
slicing keeps this genuinely "cheap": ~20 cities x ~115 KB is a couple MB
total, at the cost of one small HTTP range request PER ROW (not per
window) -- more requests, far less bandwidth, and the per-connection
throttle observed against this server (~70-120 KB/s) makes bandwidth, not
request count, the thing worth minimizing.

Window size: 1.0 x 1.0 degree (~111 km N-S, less E-W at higher
latitudes) centered on each city -- several times the extent of most
individual cities' built-up cores, so it captures genuine urban-to-rural
contrast within the SAME window (the immediate-surroundings comparison
the task brief asked for, not a comparison against a global mean).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "fetch"))
from range_client import RangeClient
from tiff_header import TiffHeaderReader

DENSITY_URL = (
    "https://www.research-collection.ethz.ch/server/api/core/bitstreams/"
    "670c47cd-b642-4932-8f50-769bf71f19c6/content"
)

# ~20 cities spanning climate, development level, and continent, per the
# task brief: dense European, North American sprawl, megacities, and a
# few high-density-surroundings cases so the rural/urban contrast is
# measurable at all. (lat, lon) in degrees.
CITIES = {
    "Amsterdam": (52.37, 4.90),
    "Paris": (48.86, 2.35),
    "London": (51.51, -0.13),
    "Stockholm": (59.33, 18.07),
    "Moscow": (55.76, 37.62),
    "Phoenix": (33.45, -112.07),
    "Atlanta": (33.75, -84.39),
    "Toronto": (43.65, -79.38),
    "Mexico City": (19.43, -99.13),
    "Tokyo": (35.68, 139.65),
    "Cairo": (30.04, 31.24),
    "Lagos": (6.52, 3.38),
    "Sao Paulo": (-23.55, -46.63),
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Beijing": (39.90, 116.41),
    "Istanbul": (41.01, 28.98),
    "Nairobi": (-1.29, 36.82),
    "Buenos Aires": (-34.60, -58.38),
    "Jakarta": (-6.21, 106.85),
}

WINDOW_DEG = 1.0  # total width/height of each city window, degrees


def fetch_density_header() -> object:
    client = RangeClient(DENSITY_URL, max_retries=12)
    info = TiffHeaderReader(DENSITY_URL, client.fetch).parse()
    return client, info


def latlon_to_rowcol(lat: float, lon: float, info) -> tuple[int, int]:
    """Native pixel (row, col) for a lat/lon, using the TIFF's own
    tiepoint/pixel-scale (origin -180, 88; 0.0083333 deg/px)."""
    i0, j0, _k0, lon0, lat0, _z0 = info.model_tiepoint
    dlon, dlat, _dz = info.model_pixel_scale
    col = int(round((lon - lon0) / dlon + i0))
    row = int(round((lat0 - lat) / dlat + j0))
    return row, col


def fetch_density_window(
    client: RangeClient,
    info,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    max_workers: int = 5,
) -> np.ndarray:
    """Fetch native rows [row0,row1) x columns [col0,col1) via one small
    column-sliced range GET per row, in parallel. Returns float64 array
    with NaN for ocean, shape (row1-row0, col1-col0)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stride = info.row_stride
    dt = np.dtype(f"{info.byte_order}{info.dtype_str}")
    itemsize = dt.itemsize
    n_rows = row1 - row0
    n_cols = col1 - col0
    out = np.empty((n_rows, n_cols), dtype=np.float64)

    def fetch_row(i: int):
        r = row0 + i
        base = info.strip_offsets[r]
        start = base + col0 * itemsize
        end = base + col1 * itemsize - 1
        raw = client.fetch(start, end)
        return i, np.frombuffer(raw, dtype=dt).astype(np.float64)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_row, i) for i in range(n_rows)]
        for fut in as_completed(futures):
            i, row = fut.result()
            out[i, :] = row
    return out


def city_bbox(lat: float, lon: float, info) -> tuple[int, int, int, int]:
    half = WINDOW_DEG / 2
    row_n, col_w = latlon_to_rowcol(lat + half, lon - half, info)
    row_s, col_e = latlon_to_rowcol(lat - half, lon + half, info)
    row0, row1 = max(0, row_n), min(info.height, row_s)
    col0, col1 = max(0, col_w), min(info.width, col_e)
    return row0, row1, col0, col1


if __name__ == "__main__":
    client, info = fetch_density_header()
    print(f"header ok: {info.width}x{info.height}, row_stride={info.row_stride}")
    name, (lat, lon) = next(iter(CITIES.items()))
    row0, row1, col0, col1 = city_bbox(lat, lon, info)
    print(
        f"{name}: rows [{row0},{row1}) cols [{col0},{col1}) -> "
        f"{row1 - row0}x{col1 - col0} pixels"
    )
    window = fetch_density_window(client, info, row0, row1, col0, col1)
    print(
        f"fetched window shape={window.shape}, "
        f"n_nan={np.isnan(window).sum()}, "
        f"land_min/max={np.nanmin(window):.0f}/{np.nanmax(window):.0f}"
    )
