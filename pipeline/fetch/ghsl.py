#!/usr/bin/env python3
"""Load GHSL built-up surface (GHS-BUILT-S E2020, 30 arc-second, EPSG:4326,
CC BY) and decimate it to a target equirect FRACTION grid.

European Commission Joint Research Centre source archive:
`https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/
GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip`
Cached locally at `pipeline/cache/ghsl/` (gitignored).

Values are built-up AREA in m^2 per cell (uint32), not a fraction --
converting to a fraction requires dividing by each cell's own physical
area (a function of latitude), which is why decimation here is a proper
AREA BOX-SUM (conserves total built-up area under reprojection, exactly
analogous to `etopo.py`'s box-average for elevation) followed by a
per-output-cell normalization, not a naive mean-of-areas.

Coverage: 21384 rows (not the full 21600 a seamless 180 deg/0.0083333
would need) -- GHSL's own product clips a bit short of the poles, tiepoint
lat0=89.0996 not 90.0. Reasonable for a real-world product (no cities in
Antarctica), and cube texels beyond this extent are treated as u=0
(genuinely no built-up area, not "no data") rather than extrapolated.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tifffile

GHSL_PATH = (
    Path(__file__).parent.parent
    / "cache"
    / "ghsl"
    / "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
)
R_MEAN = 6_371_000.0


def load_ghsl():
    """Returns (area_m2_array, lon0, lat0, dlon, dlat)."""
    with tifffile.TiffFile(str(GHSL_PATH)) as tf:
        tags = {t.name: t.value for t in tf.pages[0].tags}
    arr = tifffile.imread(str(GHSL_PATH)).astype(np.float64)
    g_scale = tags["ModelPixelScaleTag"]
    g_tie = tags["ModelTiepointTag"]
    return arr, g_tie[3], g_tie[4], g_scale[0], g_scale[1]


def _row_band_area_m2(lat0: float, dlat: float, n_rows: int, dlon: float) -> np.ndarray:
    """Exact spherical per-pixel area (m^2) for each of n_rows rows of
    width dlon (deg), starting at lat0 and stepping south by dlat."""
    lat_edges = lat0 - np.arange(n_rows + 1) * dlat
    band_area = (
        R_MEAN**2
        * math.radians(dlon)
        * (np.sin(np.radians(lat_edges[:-1])) - np.sin(np.radians(lat_edges[1:])))
    )
    return band_area


def decimate_built_up_fraction(
    area_m2: np.ndarray,
    lon0: float,
    lat0: float,
    dlon: float,
    dlat: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Box-SUM native built-up area within each output cell's footprint,
    divide by that cell's own physical area -> fraction in [0,1]. Every
    native cell is counted in exactly one output cell (full coverage, no
    gaps), so total built-up area is exactly conserved by this step --
    the per-cell division into a fraction is the only lossy-looking part,
    and it's exactly invertible (multiply back by output pixel area)."""
    in_h, in_w = area_m2.shape
    row_edges = np.linspace(0, in_h, out_h + 1).round().astype(np.int64)
    col_edges = np.linspace(0, in_w, out_w + 1).round().astype(np.int64)

    row_sum = np.empty((out_h, in_w), dtype=np.float64)
    for r in range(out_h):
        a, b = row_edges[r], row_edges[r + 1]
        row_sum[r] = area_m2[a:b].sum(axis=0)
    out_sum = np.empty((out_h, out_w), dtype=np.float64)
    for c in range(out_w):
        a, b = col_edges[c], col_edges[c + 1]
        out_sum[:, c] = row_sum[:, a:b].sum(axis=1)

    out_dlon = dlon * in_w / out_w
    out_pixel_area = _row_band_area_m2(lat0, dlat * in_h / out_h, out_h, out_dlon)
    fraction = out_sum / out_pixel_area[:, None]
    return np.clip(fraction, 0.0, 1.0)


def total_built_up_area_m2(
    fraction_grid: np.ndarray, lat0: float, dlat_total_deg: float, dlon_total_deg: float
) -> float:
    """Area-weighted integral of fraction*pixel_area -- total built-up
    area (m^2) represented by a fraction grid, for the mass-conservation
    gate (built-up area is strictly positive, so a plain ratio works
    fine here, unlike the signed elevation field)."""
    out_h, out_w = fraction_grid.shape
    dlat = dlat_total_deg / out_h
    dlon = dlon_total_deg / out_w
    pixel_area = _row_band_area_m2(lat0, dlat, out_h, dlon)
    return float((fraction_grid * pixel_area[:, None]).sum())


if __name__ == "__main__":
    arr, lon0, lat0, dlon, dlat = load_ghsl()
    print(
        f"shape={arr.shape} lon0={lon0:.4f} lat0={lat0:.4f} dlon={dlon:.6f} dlat={dlat:.6f}"
    )
    frac = decimate_built_up_fraction(arr, lon0, lat0, dlon, dlat, 256, 512)
    total = total_built_up_area_m2(frac, lat0, dlat * arr.shape[0], dlon * arr.shape[1])
    print(
        f"decimated 512x256 total built-up area: {total:.4e} m^2 "
        f"({total / 1e6:.1f} km^2, {total / 1e12:.4f} Mkm^2)"
    )
