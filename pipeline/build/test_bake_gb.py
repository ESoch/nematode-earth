"""Pytest suite for `bake_cube.bake_face`'s G/B channel extension --
sampling the structure-fraction (G) and un-diluted bootstrap-CV (B) grids
through the exact same bilinear + polar-gap path already proven correct
for R/A (see test_bake.py). All fixtures are synthetic; no network.

The single most important property this suite guards: passing g_grid/
b_grid=None (the default) must reproduce the ORIGINAL reserved-zero
behaviour bit-for-bit -- this is what lets `rebake_gb.py` claim R and A
stayed byte-identical without re-deriving the whole bake, just by diffing
face bytes (see rebake_gb.py's own verification step)."""

from __future__ import annotations

import numpy as np
import pytest
from bake_cube import bake_face
from raster import bilinear_sample

PROJECTIONS = ["plain", "eac"]


def _uniform_fixture(value: float, out_h=64, out_w=128, lat0=88.0):
    grid = np.full((out_h, out_w), value, dtype=np.float64)
    lon0 = -180.0
    dlon = 360.0 / out_w
    dlat = (2 * lat0) / out_h
    return grid, lon0, lat0, dlon, dlat


# ---------------------------------------------------------------------------
# Omitting g_grid/b_grid must reproduce the exact original reserved-zero
# behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_omitting_g_and_b_grids_matches_original_reserved_zero_behaviour(projection):
    density_grid, lon0, lat0, dlon, dlat = _uniform_fixture(3_000_000.0)
    land_grid = np.full_like(density_grid, 1.0)
    size = 32
    rgba_default, gap_default = bake_face(
        0, size, density_grid, land_grid, lon0, lat0, dlon, dlat, lat0, projection
    )
    rgba_explicit_none, gap_explicit = bake_face(
        0,
        size,
        density_grid,
        land_grid,
        lon0,
        lat0,
        dlon,
        dlat,
        lat0,
        projection,
        g_grid=None,
        b_grid=None,
    )
    assert np.array_equal(rgba_default, rgba_explicit_none)
    assert gap_default == gap_explicit
    g_u16 = rgba_default[..., 1]
    b_u16 = rgba_default[..., 2]
    assert np.all(g_u16.view(np.float16) == np.float16(0.0))
    assert np.all(b_u16.view(np.float16) == np.float16(0.0))


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_r_and_a_channels_unaffected_by_presence_of_g_and_b_grids(projection):
    """The load-bearing regression check: adding non-trivial g_grid/b_grid
    must not perturb R or A by a single bit, since they are sampled from
    completely separate source grids through separately-computed
    intermediate arrays (see bake_face) -- only the fact that they share
    the same lon_deg/lat_sample/polar_gap should make them agree in
    SHAPE, not value."""
    density_grid, lon0, lat0, dlon, dlat = _uniform_fixture(3_000_000.0)
    land_grid = np.full_like(density_grid, 1.0)
    g_grid = np.full_like(density_grid, 0.37)
    b_grid = np.full_like(density_grid, 0.21)
    size = 32
    rgba_without, _ = bake_face(
        1, size, density_grid, land_grid, lon0, lat0, dlon, dlat, lat0, projection
    )
    rgba_with, _ = bake_face(
        1,
        size,
        density_grid,
        land_grid,
        lon0,
        lat0,
        dlon,
        dlat,
        lat0,
        projection,
        g_grid=g_grid,
        b_grid=b_grid,
    )
    assert np.array_equal(rgba_without[..., 0], rgba_with[..., 0])  # R
    assert np.array_equal(rgba_without[..., 3], rgba_with[..., 3])  # A


# ---------------------------------------------------------------------------
# G/B sampling itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_g_and_b_recover_uniform_source_values(projection):
    density_grid, lon0, lat0, dlon, dlat = _uniform_fixture(3_000_000.0)
    land_grid = np.full_like(density_grid, 1.0)
    g_grid = np.full_like(density_grid, 0.134)
    b_grid = np.full_like(density_grid, 0.182)
    size = 32
    rgba, _ = bake_face(
        2,
        size,
        density_grid,
        land_grid,
        lon0,
        lat0,
        dlon,
        dlat,
        lat0,
        projection,
        g_grid=g_grid,
        b_grid=b_grid,
    )
    g = rgba[..., 1].view(np.float16).astype(np.float64)
    b = rgba[..., 2].view(np.float16).astype(np.float64)
    # float16 round-trip tolerance, not exact equality
    assert np.allclose(g, 0.134, atol=1e-3)
    assert np.allclose(b, 0.182, atol=1e-3)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_g_is_clipped_to_zero_one_and_b_is_clipped_to_non_negative(projection):
    """Defensive clip at bake time -- same posture as the pipeline's
    existing land-fraction clip -- in case an upstream grid ever carries
    an out-of-range value (functional_group.structure_fraction and
    uncertainty.assert_bootstrap_cv_valid should already have caught this
    upstream, but bake_face doesn't trust that blindly)."""
    density_grid, lon0, lat0, dlon, dlat = _uniform_fixture(3_000_000.0)
    land_grid = np.full_like(density_grid, 1.0)
    g_grid = np.full_like(density_grid, 1.4)  # out of [0,1]
    b_grid = np.full_like(density_grid, -0.05)  # out of [0, inf)
    size = 16
    rgba, _ = bake_face(
        3,
        size,
        density_grid,
        land_grid,
        lon0,
        lat0,
        dlon,
        dlat,
        lat0,
        "plain",
        g_grid=g_grid,
        b_grid=b_grid,
    )
    g = rgba[..., 1].view(np.float16).astype(np.float64)
    b = rgba[..., 2].view(np.float16).astype(np.float64)
    assert np.allclose(g, 1.0)
    assert np.allclose(b, 0.0)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_g_and_b_are_zeroed_in_the_polar_gap_like_a(projection):
    """Texels beyond +/-88 deg geodetic are explicit no-data on every
    channel, not just A -- consistent with the density bake's own polar-
    gap handling (see bake_cube.bake_face's docstring)."""
    density_grid, lon0, lat0, dlon, dlat = _uniform_fixture(8_000_000.0)
    land_grid = np.full_like(density_grid, 1.0)
    g_grid = np.full_like(density_grid, 0.2)
    b_grid = np.full_like(density_grid, 0.3)
    size = 128  # matches test_bake.py's own choice for reliably hitting the gap
    found = False
    for face in (4, 5):  # +Z/-Z carry the poles in this app's convention
        rgba, n_gap = bake_face(
            face,
            size,
            density_grid,
            land_grid,
            lon0,
            lat0,
            dlon,
            dlat,
            lat0,
            projection,
            g_grid=g_grid,
            b_grid=b_grid,
        )
        assert n_gap > 0
        a = rgba[..., 3].view(np.float16).astype(np.float64)
        g = rgba[..., 1].view(np.float16).astype(np.float64)
        b = rgba[..., 2].view(np.float16).astype(np.float64)
        gap_mask = a == 0.0
        assert gap_mask.sum() == n_gap
        assert np.all(g[gap_mask] == 0.0)
        assert np.all(b[gap_mask] == 0.0)
        assert np.all(g[~gap_mask] > 0.0)
        assert np.all(b[~gap_mask] > 0.0)
        found = True
    assert found


def test_g_and_b_stay_pixel_registered_with_r_and_a_through_shared_sampling_point():
    """G/B are sampled at the SAME lon_deg/lat_sample as R/A (the whole
    point of reusing bake_face rather than a separate pass) -- verify
    directly against an independent bilinear_sample call at a single
    non-uniform probe, rather than only ever testing uniform fixtures."""
    out_h, out_w, lat0 = 64, 128, 88.0
    lon0, dlon, dlat = -180.0, 360.0 / out_w, (2 * lat0) / out_h
    rng = np.random.default_rng(7)
    density_grid = rng.uniform(1e5, 1e7, size=(out_h, out_w))
    land_grid = np.full((out_h, out_w), 1.0)
    g_grid = rng.uniform(0.0, 1.0, size=(out_h, out_w))
    b_grid = rng.uniform(0.0, 0.6, size=(out_h, out_w))
    size = 24
    face = 2
    rgba, _ = bake_face(
        face,
        size,
        density_grid,
        land_grid,
        lon0,
        lat0,
        dlon,
        dlat,
        lat0,
        "plain",
        g_grid=g_grid,
        b_grid=b_grid,
    )
    # Recompute the same texel-center lon/lat this call used internally,
    # to build an independent expectation via a direct bilinear_sample
    # call (not by importing bake_face's internals).
    idx = (np.arange(size) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(idx, idx)
    from eac import direction_to_lonlat, face_direction
    from wgs84 import geocentric_to_geodetic_lat

    x, y, z = face_direction(face, u, v, "plain")
    norm = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    lon_rad, lat_gc_rad = direction_to_lonlat(x, y, z)
    lat_gd_rad = np.vectorize(geocentric_to_geodetic_lat)(lat_gc_rad)
    lon_deg = np.degrees(lon_rad)
    lat_sample = np.clip(np.degrees(lat_gd_rad), -lat0, lat0)
    expected_g = bilinear_sample(g_grid, lon_deg, lat_sample, lon0, lat0, dlon, dlat)
    expected_b = bilinear_sample(b_grid, lon_deg, lat_sample, lon0, lat0, dlon, dlat)

    got_g = rgba[..., 1].view(np.float16).astype(np.float64)
    got_b = rgba[..., 2].view(np.float16).astype(np.float64)
    # float16 storage tolerance
    assert np.allclose(got_g, np.clip(expected_g, 0, 1), atol=2e-3)
    assert np.allclose(got_b, np.maximum(expected_b, 0), atol=2e-3)
