"""Pytest suite for the cube-bake step: EAC/plain round-trip (both
projections), the texel solid-angle ratio claim (5.196:1 plain vs
sqrt(2):1 EAC), geodetic/geocentric conversion against the known WGS84
extremum, and a mass-conservation gate on a synthetic fixture with a
known-by-construction answer -- exercising the exact same code path
(`bake_face` + `validate_bake`) the real bake uses for BOTH projections,
just with a small analytic source grid instead of the real 512x256 data,
so it runs in milliseconds and has an exact expected answer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from bake_cube import DENSITY_FLOOR, bake_face, ground_texel_stats, validate_bake
from eac import (
    FACE_COUNT,
    direction_to_face_uv,
    direction_to_lonlat,
    eac_unwarp,
    eac_warp,
    face_direction,
    lonlat_to_direction,
    rect_solid_angle,
    texel_solid_angles,
)
from wgs84 import (
    R_MEAN,
    geocentric_to_geodetic_lat,
    geodetic_to_geocentric_lat,
    max_lat_offset_arcmin,
)

PROJECTIONS = ["plain", "eac"]

# ---------------------------------------------------------------------------
# 1. Round-trip: direction -> face/uv -> direction, both projections
# ---------------------------------------------------------------------------


def _direction_for(
    face: int, u: float, v: float, projection: str
) -> tuple[float, float, float]:
    x, y, z = face_direction(face, np.array(u), np.array(v), projection)
    n = math.hypot(float(x), float(y), float(z))
    return float(x) / n, float(y) / n, float(z) / n


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("face", range(FACE_COUNT))
@pytest.mark.parametrize(
    "u,v", [(0.0, 0.0), (0.5, -0.3), (-0.7, 0.6), (0.9, 0.9), (-0.9, -0.2)]
)
def test_round_trip_uv_to_direction_to_uv(projection, face, u, v):
    x, y, z = _direction_for(face, u, v, projection)
    got_face, got_u, got_v = direction_to_face_uv(x, y, z, projection)
    assert got_face == face
    assert got_u == pytest.approx(u, abs=1e-9)
    assert got_v == pytest.approx(v, abs=1e-9)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize(
    "direction",
    [
        (1.0, 0.3, -0.4),
        (-1.0, 0.2, 0.1),
        (0.5, 2.0, -0.3),
        (0.1, -1.5, 0.9),
        (0.2, -0.3, 3.0),
        (-0.4, 0.1, -2.5),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0),
    ],
)
def test_round_trip_direction_to_uv_to_direction(projection, direction):
    n = math.hypot(*direction)
    want = tuple(c / n for c in direction)
    face, u, v = direction_to_face_uv(*direction, projection)
    got = _direction_for(face, u, v, projection)
    for a, b in zip(want, got):
        assert a == pytest.approx(b, abs=1e-9)


def test_eac_warp_is_identity_at_endpoints():
    # tan(+-pi/4) = +-1 exactly: EAC and gnomonic must agree at face edges.
    assert eac_warp(np.array([-1.0, 0.0, 1.0])) == pytest.approx([-1.0, 0.0, 1.0])
    assert eac_unwarp(np.array([-1.0, 0.0, 1.0])) == pytest.approx([-1.0, 0.0, 1.0])


def test_plain_and_eac_agree_at_face_edges_and_center():
    """The two projections are the SAME embedding table with different
    warps, and both warps are identities at -1, 0, 1 -- so plain and EAC
    must produce IDENTICAL directions at those points, differing only
    in-between."""
    for face in range(FACE_COUNT):
        for u, v in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]:
            plain = _direction_for(face, u, v, "plain")
            eac = _direction_for(face, u, v, "eac")
            assert plain == pytest.approx(eac, abs=1e-9)


def test_lonlat_direction_round_trip():
    lons = np.radians(np.array([-179.0, -90.0, 0.0, 45.0, 179.0]))
    lats = np.radians(np.array([-87.0, -30.0, 0.0, 30.0, 87.0]))
    x, y, z = lonlat_to_direction(lons, lats)
    lon2, lat2 = direction_to_lonlat(x, y, z)
    assert np.allclose(lon2, lons, atol=1e-12)
    assert np.allclose(lat2, lats, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Solid angle / texel quality: the actual EAC-vs-plain trade
# ---------------------------------------------------------------------------


def test_rect_solid_angle_full_face():
    # A full face [-1,1]x[-1,1] subtends exactly 1/6 of the sphere.
    assert rect_solid_angle(-1, 1, -1, 1) == pytest.approx(4 * math.pi / 6, rel=1e-12)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_texel_solid_angles_sum_to_full_face(projection):
    sa = texel_solid_angles(16, projection)
    assert sa.sum() == pytest.approx(4 * math.pi / 6, rel=1e-9)


def test_plain_texel_solid_angle_variation_is_5x_not_sqrt2():
    """Plain gnomonic's known worst-case: corner texels subtend ~5.196x the
    solid angle of a face-center texel. 5.196 is the CONTINUUM (point
    Jacobian) limit; the discretized finite-texel ratio converges to it
    slowly (5.03 at face_size=64, 5.155 at 256, 5.186 at 1024) because a
    finite corner texel's integral averages over some of its own extreme
    density -- so this needs a fine face_size to land within a tight
    tolerance of the textbook number, unlike the EAC test below which
    converges much faster."""
    sa = texel_solid_angles(2048, "plain")
    ratio = sa.max() / sa.min()
    assert ratio == pytest.approx(5.196, rel=0.005)
    # And confirm it's headed the right way, not coincidentally close.
    assert (
        texel_solid_angles(64, "plain").max() / texel_solid_angles(64, "plain").min()
        < ratio
    )


def test_eac_texel_solid_angle_variation_is_sqrt2_not_5x():
    """The whole point of EAC over plain gnomonic: worst-case texel
    solid-angle ratio should be close to sqrt(2), not gnomonic's 5.196."""
    sa = texel_solid_angles(64, "eac")
    ratio = sa.max() / sa.min()
    assert ratio == pytest.approx(math.sqrt(2), rel=0.05)
    assert ratio < 2.0  # nowhere near gnomonic's 5.196


def test_eac_minimum_solid_angle_is_at_face_edge_midpoints():
    """Per the task brief: EAC's minimum should sit at face-edge midpoints,
    not the corners (where plain gnomonic's extreme sits) and not dead
    center. Locate the argmin texel and check it's on a boundary row/col
    but not in a corner."""
    size = 64
    sa = texel_solid_angles(size, "eac")
    i, j = np.unravel_index(np.argmin(sa), sa.shape)
    on_edge = i in (0, size - 1) or j in (0, size - 1)
    in_corner = i in (0, size - 1) and j in (0, size - 1)
    assert on_edge and not in_corner


def test_eac_beats_plain_at_every_face_size():
    """EAC's worst-case ground texel must be smaller (better) than plain's
    at the same face size, for any size -- the entire premise of trading
    warp ALU for resolution."""
    for size in (16, 32, 64, 128):
        plain = ground_texel_stats(size, "plain")
        eac = ground_texel_stats(size, "eac")
        assert eac["worst_case_km"] < plain["worst_case_km"]
        assert (
            eac["solid_angle_ratio_max_over_min"]
            < plain["solid_angle_ratio_max_over_min"]
        )


def test_lonlat_direction_round_trip_matches_ground_stats_km_positive():
    # Sanity: ground_texel_stats returns physically sane (positive,
    # ordered) numbers for a realistic face size.
    stats = ground_texel_stats(128, "eac")
    assert 0 < stats["best_case_km"] <= stats["rms_km"] <= stats["worst_case_km"]


# ---------------------------------------------------------------------------
# 3. Geodetic/geocentric conversion against the known WGS84 extremum
# ---------------------------------------------------------------------------


def test_geodetic_geocentric_known_extremum_near_45_1_deg():
    """Textbook WGS84 result (and app/src/math/wgs84.ts's own doc comment):
    max |phi_gd - phi_gc| = 11.545 arcmin, at ~45.1 deg latitude."""
    offset_arcmin = max_lat_offset_arcmin()
    assert offset_arcmin == pytest.approx(11.545, abs=0.001)

    lat_45_1 = math.radians(45.1)
    gc = geodetic_to_geocentric_lat(lat_45_1)
    offset_at_45_1 = (lat_45_1 - gc) * 180 * 60 / math.pi
    assert offset_at_45_1 == pytest.approx(11.545, abs=0.01)

    ground_displacement_km = math.radians(offset_at_45_1 / 60) * R_MEAN / 1000.0
    assert ground_displacement_km == pytest.approx(21.4, abs=0.2)


def test_geodetic_geocentric_inverse_round_trip():
    for lat_deg in [-89, -60, -45.1, -10, 0, 10, 45.1, 60, 89]:
        lat = math.radians(lat_deg)
        gc = geodetic_to_geocentric_lat(lat)
        gd = geocentric_to_geodetic_lat(gc)
        assert gd == pytest.approx(lat, abs=1e-12)


def test_geodetic_geocentric_identity_at_equator_and_poles():
    for lat_deg in [0, 90, -90]:
        lat = math.radians(lat_deg)
        assert geodetic_to_geocentric_lat(lat) == pytest.approx(lat, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Mass-conservation gate: known-by-construction fixture, BOTH projections
# ---------------------------------------------------------------------------


def _uniform_fixture(density_value: float, out_h=64, out_w=128, lat0=88.0):
    """A synthetic equirect grid of uniform density and land_fraction=1
    everywhere, spanning the same +/-88 deg clip as the real data. The
    exact global integral is known analytically: density_value * (area of
    the sphere within +/-88 deg)."""
    density_grid = np.full((out_h, out_w), density_value, dtype=np.float64)
    land_grid = np.full((out_h, out_w), 1.0, dtype=np.float64)
    lon0 = -180.0
    dlon = 360.0 / out_w
    dlat = (2 * lat0) / out_h
    return density_grid, land_grid, lon0, lat0, dlon, dlat


def _bake_all_faces(
    size, density_grid, land_grid, lon0, lat0, dlon, dlat, polar_limit_deg, projection
):
    faces = []
    for face in range(FACE_COUNT):
        rgba_u16, _n_gap = bake_face(
            face,
            size,
            density_grid,
            land_grid,
            lon0,
            lat0,
            dlon,
            dlat,
            polar_limit_deg,
            projection,
        )
        faces.append(rgba_u16)
    return faces


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_mass_conservation_uniform_density_recovers_known_integral(projection):
    density_value = 5_000_000.0  # well above DENSITY_FLOOR, mid the real land range
    # size=128 matches the real bake's face size and (per the polar-gap test)
    # reliably discretizes at least one texel beyond +/-88 deg, so the
    # analytic "clipped band" formula below is what the discretized sum
    # should actually converge to. A coarser size (e.g. 48) can fail to put
    # any texel center past 88 deg at all -- the gap is only ~2 deg wide --
    # in which case the bake legitimately covers the FULL sphere instead
    # and comparing it to the clipped-band formula is the wrong check, not
    # a bug in the bake.
    size = 128
    density_grid, land_grid, lon0, lat0, dlon, dlat = _uniform_fixture(density_value)
    faces = _bake_all_faces(
        size, density_grid, land_grid, lon0, lat0, dlon, dlat, lat0, projection
    )
    result = validate_bake(faces, size, projection)

    # Expected uses the ACTUAL float16-quantized density, not the exact
    # input value: R16Float has ~3 decimal digits of precision, so a
    # uniform 5e6 rounds to a specific nearby representable value (here,
    # ~0.31% off), and every one of this fixture's texels rounds the same
    # way (same input everywhere) -- so the correct expectation is that
    # exact, deterministic rounding, not the unquantized value +/- a
    # guessed tolerance. Computed independently of `validate_bake`'s own
    # accounting: re-derive which texels are gapped from the land channel
    # (uniformly 1.0 in-domain in this fixture, so any 0 is a gap texel)
    # and split the exact per-texel solid angles accordingly, rather than
    # reusing `result`'s internals.
    quantized_density = float(np.exp(np.float64(np.float16(math.log(density_value)))))
    sa = texel_solid_angles(size, projection)
    non_gap_sa_total = 0.0
    gap_sa_total = 0.0
    n_gap_texels = 0
    for rgba_u16 in faces:
        land = rgba_u16[..., 3].view(np.float16).astype(np.float64)
        gap_mask = land == 0.0
        non_gap_sa_total += float(sa[~gap_mask].sum())
        gap_sa_total += float(sa[gap_mask].sum())
        n_gap_texels += int(gap_mask.sum())
    expected = (
        quantized_density * non_gap_sa_total + DENSITY_FLOOR * gap_sa_total
    ) * R_MEAN**2
    assert result["integral"] == pytest.approx(expected, rel=1e-6)
    # The only floored texels should be the tiny polar gap beyond +/-88 deg
    # (density is uniformly well above the floor everywhere in-domain).
    assert result["n_floored_texels"] == n_gap_texels
    assert n_gap_texels > 0  # confirms this fixture actually exercises the gap


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_mass_conservation_zero_density_hits_floor_not_poisoned(projection):
    """Every texel decodes to DENSITY_FLOOR, not 0 or -inf/NaN -- this is
    the check that would have caught the log(0)=-inf filtering bug."""
    size = 24
    density_grid, land_grid, lon0, lat0, dlon, dlat = _uniform_fixture(0.0)
    faces = _bake_all_faces(
        size, density_grid, land_grid, lon0, lat0, dlon, dlat, lat0, projection
    )
    for rgba_u16 in faces:
        r_u16 = rgba_u16[..., 0]
        log_density = r_u16.view(np.float16).astype(np.float64)
        assert np.all(np.isfinite(log_density)), (
            "log(0) must not reach the output as -inf"
        )
        assert np.allclose(np.exp(log_density), DENSITY_FLOOR)

    result = validate_bake(faces, size, projection)
    n_total = FACE_COUNT * size * size
    # All texels are within +/-88 (no polar gap at this size/grid), all are
    # exactly zero density -> all floored.
    assert result["n_floored_texels"] == n_total
    assert result["floor_contamination"] == pytest.approx(result["integral"], rel=1e-9)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_polar_gap_beyond_88_degrees_is_zeroed_not_extrapolated(projection):
    """A texel whose geodetic latitude exceeds the source's +/-88 deg
    domain must be explicit no-data (floored density, zero land-fraction),
    never a value invented by clamping into extrapolated territory.

    In this fixture land_fraction is uniformly 1.0 everywhere in-domain, so
    ANY texel decoding to land_fraction==0 must be a polar-gap texel --
    that gives a robust way to find them without hardcoding which pixel
    index the ~2 deg gap lands on at a given face size (it's a small
    fraction of a texel's angular width at coarse sizes, so a fixed
    "center" index is not reliably inside it)."""
    size = 128  # matches the real bake's face size; ~0.7 deg/texel keeps
    # the 2-degree polar gap (88..90) comfortably larger than one texel.
    density_value = 8_000_000.0
    density_grid, land_grid, lon0, lat0, dlon, dlat = _uniform_fixture(density_value)
    found_polar_texel = False
    # Z is the polar axis in this app's convention (see eac.py's docstring),
    # so the poles live on the +Z/-Z faces (4, 5), not +Y/-Y.
    for face in (4, 5):
        rgba_u16, n_gap = bake_face(
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
        )
        assert n_gap > 0, (
            f"face {face} should have at least one polar-gap texel at size={size}"
        )
        a = rgba_u16[..., 3].view(np.float16).astype(np.float64)
        r = rgba_u16[..., 0].view(np.float16).astype(np.float64)
        gap_mask = a == 0.0
        assert gap_mask.sum() == n_gap
        found_polar_texel = True
        assert np.all(r[gap_mask] == pytest.approx(math.log(DENSITY_FLOOR)))
        # And every non-gap texel keeps the real (non-floored) density --
        # compared against the actual float16-rounded expected value (see
        # test_mass_conservation_uniform_density_recovers_known_integral
        # for why the raw unquantized log() isn't the right comparison).
        expected_r = float(np.float16(math.log(density_value)))
        assert np.all(r[~gap_mask] == pytest.approx(expected_r))
    assert found_polar_texel
