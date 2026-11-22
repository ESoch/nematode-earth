"""Shared equirect-grid bilinear sampling, used identically by
`bake_cube.py` (density) and `bake_elevation_cube.py` (ETOPO) so both cube
bakes go through the exact same interpolation code."""

from __future__ import annotations

import numpy as np


def bilinear_sample(
    grid: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    lon0: float,
    lat0: float,
    dlon: float,
    dlat: float,
) -> np.ndarray:
    """Bilinear-sample an equirect grid (row 0 = lat0, column 0 = lon0,
    uniform spacing dlon/dlat, longitude periodic) at arbitrary (lon,lat)
    query points (degrees). `grid` shape (height, width). Latitude is
    clamped to the grid's row range, not wrapped (there is no polar wrap
    for an equirect grid); longitude wraps (360 deg periodic)."""
    height, width = grid.shape
    col_f = (lon_deg - lon0) / dlon - 0.5
    col_f = np.mod(col_f, width)
    c0 = np.floor(col_f).astype(np.int64) % width
    c1 = (c0 + 1) % width
    fc = col_f - np.floor(col_f)

    row_f = (lat0 - lat_deg) / dlat - 0.5
    row_f = np.clip(row_f, 0.0, height - 1.0)
    r0 = np.floor(row_f).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, height - 1)
    fr = row_f - r0

    v00 = grid[r0, c0]
    v01 = grid[r0, c1]
    v10 = grid[r1, c0]
    v11 = grid[r1, c1]
    return (
        (1 - fr) * (1 - fc) * v00
        + (1 - fr) * fc * v01
        + fr * (1 - fc) * v10
        + fr * fc * v11
    )
