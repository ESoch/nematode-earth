"""Synthetic tests for geographic indexing and the city-window regression."""

from __future__ import annotations

import numpy as np
import pytest
from city_calibration import bootstrap_ci, ghsl_window_fraction
from city_windows import latlon_to_rowcol
from tiff_header import TiffInfo


def _fake_density_info():
    """Minimal TiffInfo matching the real density raster's georeferencing
    (tiepoint -180,88; 0.0083333 deg/px; 43201x21121), for latlon_to_rowcol."""
    return TiffInfo(
        url="fake",
        byte_order="<",
        big_tiff=True,
        ifd_offset=0,
        width=43201,
        height=21121,
        bits_per_sample=64,
        sample_format=3,
        samples_per_pixel=1,
        compression=1,
        rows_per_strip=1,
        strip_offsets=[0],
        strip_byte_counts=[0],
        model_pixel_scale=(0.008333333333333333, 0.008333333333333333, 0.0),
        model_tiepoint=(0.0, 0.0, 0.0, -180.0, 88.0, 0.0),
    )


def test_latlon_to_rowcol_known_points():
    info = _fake_density_info()
    # Tiepoint itself: (lon=-180, lat=88) -> (row=0, col=0).
    row, col = latlon_to_rowcol(88.0, -180.0, info)
    assert (row, col) == (0, 0)
    # One pixel step south and east.
    row, col = latlon_to_rowcol(
        88.0 - 0.008333333333333333, -180.0 + 0.008333333333333333, info
    )
    assert (row, col) == (1, 1)
    # Equator, prime meridian.
    row, col = latlon_to_rowcol(0.0, 0.0, info)
    assert row == pytest.approx(88.0 / 0.008333333333333333, abs=1)
    assert col == pytest.approx(180.0 / 0.008333333333333333, abs=1)


def test_ghsl_window_fraction_uniform_built_up():
    """A GHSL array where every cell is fully built-up (area == pixel
    area at that latitude) must produce fraction ~1.0 everywhere, and a
    fully-empty array must produce ~0.0 everywhere."""
    ghsl_h, ghsl_w = 200, 400
    g_lon0, g_lat0 = -10.0, 10.0
    g_dlon = g_dlat = 0.01
    R_MEAN = 6_371_000.0
    import math

    # Fill every GHSL cell with exactly its own pixel area (m^2) -> u should be ~1.
    lat_edges = g_lat0 - np.arange(ghsl_h + 1) * g_dlat
    band_area = (
        R_MEAN**2
        * math.radians(g_dlon)
        * (np.sin(np.radians(lat_edges[:-1])) - np.sin(np.radians(lat_edges[1:])))
    )
    full = np.broadcast_to(band_area[:, None], (ghsl_h, ghsl_w)).copy()
    empty = np.zeros((ghsl_h, ghsl_w))

    lat_n, lat_s = 9.0, 8.0
    lon_w, lon_e = -9.0, -8.0
    # Output grid at the SAME resolution as the GHSL fixture (100 px per
    # degree, matching g_dlon=g_dlat=0.01) -- this matches how the real
    # pipeline uses it (ETH density and GHSL are both ~30 arc-second
    # natively, i.e. output pixel size ~= GHSL native pixel size).
    # ghsl_window_fraction does a NEAREST-NEIGHBOR lookup per output
    # pixel, not an area-weighted aggregation over multiple native GHSL
    # cells, so it is only correct when the two grids are near the same
    # resolution -- an earlier version of this test used a 5x coarser
    # output grid and got 1/25 of the expected fraction as a result,
    # which was a test-fixture bug, not a bug in the function.
    u_full = ghsl_window_fraction(
        full,
        full.shape,
        g_lon0,
        g_lat0,
        g_dlon,
        g_dlat,
        lat_n,
        lat_s,
        lon_w,
        lon_e,
        100,
        100,
    )
    u_empty = ghsl_window_fraction(
        empty,
        empty.shape,
        g_lon0,
        g_lat0,
        g_dlon,
        g_dlat,
        lat_n,
        lat_s,
        lon_w,
        lon_e,
        100,
        100,
    )
    assert np.allclose(u_full, 1.0, atol=0.02)
    assert np.allclose(u_empty, 0.0)


def test_ghsl_window_fraction_bounded_0_1():
    """Fraction must be clipped to [0,1] even if GHSL area exceeds the
    naive pixel-area denominator (can happen near tile/rounding edges)."""
    ghsl = np.full((50, 50), 1e12)  # absurdly large built-up area
    u = ghsl_window_fraction(
        ghsl, ghsl.shape, -10.0, 10.0, 0.01, 0.01, 9.0, 8.0, -9.0, -8.0, 10, 10
    )
    assert np.all(u <= 1.0)
    assert np.all(u >= 0.0)


def test_bootstrap_ci_recovers_narrow_interval_on_noiseless_data():
    """On perfectly noiseless synthetic data, bootstrap resampling should
    converge to a tight CI centered on the true coefficients (since every
    resample gives the exact same answer)."""
    u = np.linspace(0, 1, 500)
    true_a, true_b = -0.5, 0.3
    m = 1 + true_a * u + true_b * u**2
    ci = bootstrap_ci(u, m, n_boot=200, seed=0)
    a_lo, a_hi = ci["a_ci95"]
    b_lo, b_hi = ci["b_ci95"]
    assert a_lo <= true_a <= a_hi
    assert b_lo <= true_b <= b_hi
    assert (a_hi - a_lo) < 0.01  # tight: no noise to resample variance from


def test_bootstrap_ci_widens_with_noise():
    rng = np.random.default_rng(5)
    u = rng.uniform(0, 1, 300)
    m_clean = 1 - 0.5 * u
    ci_clean = bootstrap_ci(u, m_clean, n_boot=300, seed=1)
    m_noisy = m_clean + rng.normal(0, 1.0, 300)
    ci_noisy = bootstrap_ci(u, m_noisy, n_boot=300, seed=1)
    width_clean = ci_clean["a_ci95"][1] - ci_clean["a_ci95"][0]
    width_noisy = ci_noisy["a_ci95"][1] - ci_noisy["a_ci95"][0]
    assert width_noisy > width_clean
