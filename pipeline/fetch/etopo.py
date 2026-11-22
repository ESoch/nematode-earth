#!/usr/bin/env python3
"""Fetch ETOPO 2022 bedrock elevation (NOAA NCEI, CC0) and decimate it to a
target equirect grid, same box-averaging discipline as `eth_raster.py`.

Source: https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_bed_elev_gtif/ETOPO_2022_v1_60s_N90W180_bed.tif
  A single seamless global GeoTIFF (confirmed -- unlike the 15 arc-second
  product, which is split into ~57 regional tiles; the 30s and 60s products
  are each one file). Verified this session:
    - 21600 x 10800 pixels, float32, EPSG:4326, tiepoint (-180, +90),
      pixel scale 1/60 deg (60 arc-second) -- FULL pole-to-pole coverage,
      unlike the nematode raster's +/-88 clip.
    - Compression: Adobe Deflate (TIFF compression=8) with a horizontal
      FLOATING-POINT predictor (tag 317=3), tiled (256x256 tiles, 3655
      tiles total) -- NOT the uncompressed one-strip-per-row layout the
      nematode rasters used. This makes arithmetic per-row byte
      addressing (StripOffsets[i] + i*stride) inapplicable; a tile's
      compressed bytes must be inflated and predictor-reversed to recover
      pixel values, which risks a subtle bug if hand-rolled (the FP
      predictor does a byte-plane deinterleave, not a simple delta) --
      handled here via `imagecodecs` (the same codec library `tifffile`
      itself uses), not reimplemented from scratch.
    - Bare curl (no UA) got HTTP 200 here -- NOAA does not gate on
      User-Agent the way the ETH Research Collection did. Checked, not
      assumed, per that lesson.
    - Download speed measured ~4.6-12 MB/s (vs. ETH's ~70-120 KB/s), so at
      478 MB the file downloads whole in under 2 minutes -- full download
      + `tifffile.imread` (which handles tiling/compression/predictor
      correctly and robustly) is simpler and safer here than a range-fetch
      row-block scheme, and is the approach this module takes.
    - GDAL_NODATA tag reads the string "-99999" (confirming the brief's
      claim about the declared sentinel) but an EXHAUSTIVE scan of all
      233,280,000 pixels in the downloaded file found ZERO occurrences of
      -99999, no NaN, no Inf. The claim that the nodata value is declared
      but never actually present is CONFIRMED, not assumed -- see
      `verify_no_nodata` below.

30 arc-second (1.62 GB) and 15 arc-second (tiled) products exist for a
finer future extraction, but are NOT fetched here: even the coarsest of
the three target cube resolutions used downstream (face_size 64) implies
far more than 60-arc-second angular resolution per texel (a face_size
that would actually MATCH 60"'s native resolution works out to N~5395,
wildly impractical for a texture), so 60" alone comfortably covers every
tier without upsampling. See bake_elevation_cube.py's own resolution
check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import requests
import tifffile

SOURCE_URL = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/"
    "60s_bed_elev_gtif/ETOPO_2022_v1_60s_N90W180_bed.tif"
)
NODATA_SENTINEL = -99999.0
DEFAULT_CACHE = (
    Path(__file__).parent.parent
    / "cache"
    / "etopo"
    / "ETOPO_2022_v1_60s_N90W180_bed.tif"
)

# Surface companion raster (ice-top where glaciated, bare-earth/seafloor
# elsewhere) -- same NOAA host, same 60" tiled+deflate+FP-predictor layout,
# same (10800, 21600) float32 shape, confirmed this session via a HEAD
# request (HTTP 200, content-length 465969062 bytes ~= 444 MiB, matching
# the ~456 MB the task brief expected) before anything downstream was built
# on it. Needed alongside the bed raster because bed's sign alone is NOT a
# land mask: an ice sheet sitting on below-sea-level bedrock (e.g. the
# Byrd Subglacial Basin, -913 m bed) reads as ocean by bed sign even though
# it's clearly land -- see bake_elevation_cube.py's land-mask rule.
SURFACE_SOURCE_URL = (
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/"
    "60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif"
)
DEFAULT_SURFACE_CACHE = (
    Path(__file__).parent.parent
    / "cache"
    / "etopo"
    / "ETOPO_2022_v1_60s_N90W180_surface.tif"
)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def ensure_downloaded(cache_path: Path = DEFAULT_CACHE, url: str = SOURCE_URL) -> Path:
    """Download the full ETOPO GeoTIFF to `cache_path` if not already
    present (478 MB; NOAA's server is fast and doesn't gate on
    User-Agent, so a plain streamed download is simpler and more robust
    here than range-fetching -- see module docstring)."""
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {cache_path} ...", file=sys.stderr)
    with requests.get(
        url, headers={"User-Agent": _UA}, stream=True, timeout=120
    ) as resp:
        resp.raise_for_status()
        tmp = cache_path.with_suffix(".tif.partial")
        with open(tmp, "wb") as f:
            f.writelines(resp.iter_content(chunk_size=1 << 20))
        tmp.rename(cache_path)
    return cache_path


def load_elevation(
    cache_path: Path = DEFAULT_CACHE, url: str = SOURCE_URL
) -> np.ndarray:
    """Full array, float32, shape (10800, 21600), meters, EPSG:4326,
    tiepoint (-180, +90), 1/60 deg/pixel. Decoded via `tifffile`
    (correctly handles the tiled+deflate+floating-point-predictor
    layout).

    `url` lets this same loader fetch the surface companion raster (pass
    `DEFAULT_SURFACE_CACHE`/`SURFACE_SOURCE_URL`) -- `ensure_downloaded`
    already took a `url` param but this caller hadn't threaded it through,
    which would have silently downloaded BED data into a surface-named
    cache path if ever called with only a different `cache_path`."""
    path = ensure_downloaded(cache_path, url)
    arr = tifffile.imread(str(path))
    assert arr.shape == (10800, 21600), f"unexpected shape {arr.shape}"
    assert arr.dtype == np.float32, f"unexpected dtype {arr.dtype}"
    return arr


def verify_no_nodata(arr: np.ndarray) -> dict:
    """Exhaustive (not sampled) check of the -99999-never-present claim."""
    n_sentinel = int(np.sum(arr == NODATA_SENTINEL))
    n_nan = int(np.sum(np.isnan(arr)))
    n_inf = int(np.sum(np.isinf(arr)))
    return {
        "n_pixels": int(arr.size),
        "n_nodata_sentinel": n_sentinel,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "confirmed_never_present": n_sentinel == 0 and n_nan == 0 and n_inf == 0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def box_average_decimate(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Full-coverage contiguous-block box average -- same anti-aliasing
    discipline as `eth_raster.py::extract_grid`, but done locally in one
    shot (the whole array is already in memory; no bandwidth constraint
    here, so there's no reason to do anything less than full coverage)."""
    in_h, in_w = arr.shape
    row_edges = np.linspace(0, in_h, out_h + 1).round().astype(np.int64)
    col_edges = np.linspace(0, in_w, out_w + 1).round().astype(np.int64)
    out = np.empty((out_h, out_w), dtype=np.float64)
    # Row-wise partial reduction first (cheap), then column-wise, to avoid
    # an out_h*out_w*avg_block_size python loop.
    row_reduced = np.empty((out_h, in_w), dtype=np.float64)
    for r in range(out_h):
        a, b = row_edges[r], row_edges[r + 1]
        row_reduced[r] = arr[a:b].mean(axis=0)
    for c in range(out_w):
        a, b = col_edges[c], col_edges[c + 1]
        out[:, c] = row_reduced[:, a:b].mean(axis=1)
    return out


if __name__ == "__main__":
    arr = load_elevation()
    print(f"shape={arr.shape} dtype={arr.dtype}")
    audit = verify_no_nodata(arr)
    print(audit)
