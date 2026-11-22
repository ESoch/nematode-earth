"""Pytest suite for the ETOPO elevation bake: box-average decimation
correctness, signed-value / no-clamping encoding, and a volume-conservation
gate on a synthetic fixture with a known answer.

The gate in `test_volume_conservation_uniform_elevation` is exactly the
check that caught a real bug this session: `bake_elevation_face` was
sampling the per-tier INTERMEDIATE decimated grid using the SOURCE
raster's native (much finer) dlon/dlat instead of the intermediate grid's
own -- every face ended up repeatedly sampling a small corner, and baked
volume integrals came out 2-3 orders of magnitude off. Inspecting a few
decoded values didn't obviously look wrong (still finite, still vaguely
plausible-looking numbers); the mass/volume-conservation ratio caught it
immediately (0.0017-0.018 instead of ~1.0).

The `compute_land_mask` / `land_mask_stats` / `assert_land_mask_valid`
sections below cover the G-channel land-fraction mask: a real land mask
baked from ETOPO's bed AND surface rasters together (bed sign alone
misclassifies ice sheets sitting on below-sea-level bedrock as ocean --
see the module docstring in bake_elevation_cube.py for the Antarctica
probe table that motivated this).

The `is_grounded` section covers a second, real bug found after the first
bake shipped: a bare ice-thickness test (`surface-bed > 1.0`) also fires
on FLOATING sea ice/ice shelves over open ocean, not just grounded ice
sheets, because floating ice can be arbitrarily thick too. That
misclassified deep ocean as land -- visible in the render as a hard-edged
rotated rectangle around Greenland (the BedMachine Greenland source
mosaic's extent). The fix is a real flotation (Archimedes) test, not a
retuned threshold.

`test_land_mask_floating_ice_shelf_with_positive_freeboard_reads_ocean`
covers a THIRD, real bug found while verifying the second fix: even with
`is_grounded` correct, an `surface_elev>=0 OR is_grounded(...)` rule still
misclassified floating ICE SHELVES (Ross, Ronne-Filchner, Amery, Larsen
C) as land, because a floating shelf has positive freeboard by
construction. `compute_land_mask` is now `is_grounded` alone, with no OR
clause -- verified to still leave ordinary dry land untouched (is_grounded
reduces to bed_elev>0 when there's no ice).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "fetch"))
from bake_elevation_cube import (
    LAND_FRACTION_MAX,
    LAND_FRACTION_MIN,
    LAND_MASK_PROBES,
    RHO_ICE_OVER_RHO_WATER,
    assert_land_mask_valid,
    bake_elevation_face,
    baked_volume_integral,
    compute_land_mask,
    geodetic_probe_to_face_uv,
    is_grounded,
    land_mask_stats,
    volume_integral,
)
from eac import FACE_COUNT
from etopo import box_average_decimate, verify_no_nodata
from wgs84 import R_MEAN, geodetic_to_geocentric_lat

PROJECTIONS = ["plain", "eac"]


# ---------------------------------------------------------------------------
# box_average_decimate
# ---------------------------------------------------------------------------


def test_box_average_decimate_matches_numpy_reference():
    rng = np.random.default_rng(7)
    src = rng.uniform(-10000, 8000, size=(120, 240))
    out = box_average_decimate(src, 12, 24)
    row_edges = np.linspace(0, 120, 13).round().astype(int)
    col_edges = np.linspace(0, 240, 25).round().astype(int)
    ref = np.empty((12, 24))
    for r in range(12):
        for c in range(24):
            ref[r, c] = src[
                row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]
            ].mean()
    assert np.allclose(out, ref, rtol=1e-12)


def test_box_average_decimate_preserves_area_weighted_mean():
    """Full-coverage box averaging must not gain or lose mass: the plain
    (unweighted) mean of the decimated grid must equal the plain mean of
    the source, since every source cell is counted in exactly one output
    cell and every output cell averages the same-sized input block here
    (uniform grid, so plain mean == area-weighted-within-row mean)."""
    rng = np.random.default_rng(3)
    src = rng.uniform(-500, 500, size=(100, 200))
    out = box_average_decimate(src, 20, 40)
    assert out.mean() == pytest.approx(src.mean(), rel=1e-9)


def test_box_average_decimate_full_coverage_no_gaps():
    """Every source row must land in exactly one output row's block (and
    likewise columns) -- construct a grid where row i has value i, and
    confirm the decimated grid's row means partition [0, in_h) with no
    gaps or overlaps (sum of block sizes == in_h)."""
    in_h, out_h = 97, 10  # deliberately not evenly divisible
    src = np.tile(np.arange(in_h, dtype=np.float64)[:, None], (1, 5))
    out = box_average_decimate(src, out_h, 5)
    row_edges = np.linspace(0, in_h, out_h + 1).round().astype(int)
    block_sizes = np.diff(row_edges)
    assert block_sizes.sum() == in_h
    for r in range(out_h):
        expected_mean = np.arange(row_edges[r], row_edges[r + 1]).mean()
        assert out[r, 0] == pytest.approx(expected_mean)


# ---------------------------------------------------------------------------
# Signed encoding, no clamping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_elevation_encoding_preserves_sign_no_clamping(projection):
    """Deep-ocean negative values must survive the bake unclamped -- this
    is the entire point of shipping signed elevation."""
    h, w = 32, 64
    # Half the grid deeply negative (trench-like), half strongly positive
    # (mountain-like), nothing near zero, so clamping-to-zero would be
    # obvious in the output.
    grid = np.where(
        (np.arange(h)[:, None] + np.arange(w)[None, :]) % 2 == 0, -9000.0, 7000.0
    )
    mask_grid = np.zeros((h, w))  # content irrelevant to this R/A-only check
    rgba = bake_elevation_face(0, 16, grid, mask_grid, projection)
    r = rgba[..., 0].view(np.float16).astype(np.float64)
    assert r.min() < -1000, "negative elevation got clamped away"
    assert r.max() > 1000
    a = rgba[..., 3].view(np.float16).astype(np.float64)
    assert np.all(a == 1.0), (
        "validity channel should be 1.0 everywhere (no nodata in ETOPO)"
    )


# ---------------------------------------------------------------------------
# Volume-conservation gate: known-by-construction fixture, BOTH projections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("size", [16, 32])
def test_volume_conservation_uniform_elevation(projection, size):
    """A uniform elevation field's exact global volume is
    elevation * 4*pi*R^2 (full sphere, no polar clipping -- ETOPO covers
    +/-90 exactly, unlike the density raster). Bake it through the real
    pipeline (bake_elevation_face for all 6 faces) and confirm the
    quantized, decoded, re-integrated result recovers it."""
    elevation_value = -3500.0  # representative abyssal-plain-ish depth
    inter_h, inter_w = 64, 128
    grid = np.full((inter_h, inter_w), elevation_value, dtype=np.float64)
    mask_grid = np.zeros((inter_h, inter_w))  # irrelevant to R's volume

    faces = [
        bake_elevation_face(f, size, grid, mask_grid, projection)
        for f in range(FACE_COUNT)
    ]
    baked = baked_volume_integral(faces, size, projection)

    quantized = float(
        np.float16(elevation_value)
    )  # exact float16 rounding of -3500 is exact (power-of-2-friendly magnitude)
    expected = quantized * 4 * math.pi * R_MEAN**2
    assert baked == pytest.approx(expected, rel=1e-6)


def test_volume_integral_source_side_matches_uniform_analytic():
    """Sanity-check the SOURCE-side (pre-bake) volume_integral helper
    independently against the same analytic full-sphere formula."""
    elevation_value = 1200.0
    out_h, out_w = 30, 60
    grid = np.full((out_h, out_w), elevation_value, dtype=np.float64)
    integral = volume_integral(grid, out_h, out_w)
    expected = elevation_value * 4 * math.pi * R_MEAN**2
    assert integral == pytest.approx(expected, rel=1e-9)


def test_volume_integral_scales_linearly():
    out_h, out_w = 20, 40
    grid = np.full((out_h, out_w), 1.0, dtype=np.float64)
    i1 = volume_integral(grid, out_h, out_w)
    i2 = volume_integral(grid * 5, out_h, out_w)
    assert i2 / i1 == pytest.approx(5.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Nodata audit helper
# ---------------------------------------------------------------------------


def test_verify_no_nodata_detects_sentinel_when_present():
    """The exhaustive nodata check must actually detect the sentinel if
    it WERE present -- proving the real bake's "confirmed_never_present:
    true" result reflects a working check, not a check that can't fail."""
    arr = np.full((10, 10), 100.0, dtype=np.float32)
    arr[3, 4] = -99999.0
    audit = verify_no_nodata(arr)
    assert audit["n_nodata_sentinel"] == 1
    assert audit["confirmed_never_present"] is False


def test_verify_no_nodata_clean_grid():
    arr = np.full((10, 10), 100.0, dtype=np.float32)
    audit = verify_no_nodata(arr)
    assert audit["confirmed_never_present"] is True


# ---------------------------------------------------------------------------
# compute_land_mask: is_grounded(bed_elev, surface_elev), nothing else --
# see compute_land_mask's docstring for why the two earlier, wrong versions
# (a bare thickness test, then is_grounded OR'd with a naive surface>=0
# clause) each shipped a real misclassification bug.
# ---------------------------------------------------------------------------


def test_land_mask_fraction_not_boolean_half_above_datum():
    """A texel whose contributing source cells are half above datum and
    half below must decode to a FRACTION (0.5), not a boolean 0 or 1 --
    this is the whole point of area-weighting through box_average_decimate
    instead of thresholding a single representative cell."""
    bed = np.full((4, 4), -100.0)
    surface = np.array(
        [
            [-100.0, -100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0, -100.0],
            [500.0, 500.0, 500.0, 500.0],
            [500.0, 500.0, 500.0, 500.0],
        ]
    )
    mask = compute_land_mask(bed, surface).astype(np.float64)
    frac = box_average_decimate(mask, 1, 1)
    assert frac[0, 0] == pytest.approx(0.5)


def test_land_mask_ice_clause_fires_antarctica_case():
    """Cells with surface = +2000, bed = -500 must read as land (1.0) --
    the Antarctica case: bed alone (-500, i.e. bed>=0 as a naive land
    rule) would wrongly call this ocean, exactly the 'hole in the
    continent' bug the probe table in the module docstring documents."""
    bed = np.array([[-500.0]])
    surface = np.array([[2000.0]])
    assert compute_land_mask(bed, surface)[0, 0] == True


def test_land_mask_ice_clause_does_not_over_fire_open_ocean():
    """Open ocean where surface == bed exactly (no ice) must read as
    water (0.0) -- `surface - bed == 0` must NOT count as ice."""
    bed = np.array([[-4000.0]])
    surface = np.array([[-4000.0]])
    assert compute_land_mask(bed, surface)[0, 0] == False


def test_land_mask_dry_land_no_ice_reads_as_land():
    """Ordinary dry land (surface == bed, both positive): with no OR
    clause, this must still read land -- is_grounded reduces exactly to
    bed_elev>0 when thickness is 0, so dropping the old surface>=0 clause
    changes nothing here."""
    bed = np.array([[300.0]])
    surface = np.array([[300.0]])
    assert compute_land_mask(bed, surface)[0, 0] == True


def test_land_mask_floating_ice_shelf_with_positive_freeboard_reads_ocean():
    """THE SECOND BUG: real Ross Ice Shelf numbers (bed=-635.70,
    surf=+48.05, measured on this ETOPO grid at -80.0,-178.0) -- a
    `surface_elev>=0 OR is_grounded(...)` rule reads this as land purely
    via the surface>=0 clause (positive freeboard is normal for a
    floating shelf), even though is_grounded alone correctly says False.
    compute_land_mask (no OR clause) must read ocean."""
    bed = np.array([[-635.70]])
    surface = np.array([[48.05]])
    assert is_grounded(bed, surface)[0, 0] == False
    assert surface[0, 0] >= 0  # confirms this really would trip a surface>=0 clause
    assert compute_land_mask(bed, surface)[0, 0] == False


# ---------------------------------------------------------------------------
# is_grounded: the Archimedes flotation test, and the bug it replaces.
#
# A bare ice-thickness test (surface-bed > 1.0) was tried first and is
# WRONG: floating sea ice and ice shelves also have real thickness sitting
# on deep water, so it misclassified open ocean under floating ice as
# land -- shipped, then spotted by the user as a hard-edged rotated
# rectangle around Greenland (the BedMachine Greenland source mosaic's
# polar-stereographic extent, reprojected onto the sphere) and, less
# visibly, a broad contaminated band around the Antarctic ice shelves.
# ---------------------------------------------------------------------------


def test_is_grounded_thick_grounded_ice_reads_grounded():
    """Greenland-like: bed=-200, surface=+3000 (T=3200) -- thick ice on
    shallow bedrock, well past flotation. Must read grounded."""
    bed = np.array([[-200.0]])
    surface = np.array([[3000.0]])
    assert is_grounded(bed, surface)[0, 0] == True


def test_is_grounded_floating_ice_over_deep_water_is_not_grounded():
    """THE BUG, isolated: bed=-3000, T=1000 is real thickness (the old
    bare `surface-bed > 1.0` rule fired here), but
    0.917*1000=917 < 3000 -- the ice would float long before reaching a
    bed this deep. `is_grounded` must say False."""
    bed = np.array([[-3000.0]])
    surface = np.array([[-2000.0]])  # T = 1000
    assert is_grounded(bed, surface)[0, 0] == False


def test_is_grounded_reproduces_baffin_bay_measurement():
    """Reproduces the bug report's real numbers: bed ~-1444m, ice/snow
    surface just below sea level (T~1444) -- a real Baffin Bay ETOPO texel
    the old bare-thickness rule marked as land. Surface deliberately kept
    just NEGATIVE (not >=0) so this exercises is_grounded's contribution
    to the combined rule, not the surface clause alone: 0.917*1444 ~=
    1324, which doesn't reach a bed 1444m down, so this must NOT be
    grounded, and the combined land mask must read ocean."""
    bed = np.array([[-1444.0]])
    surface = np.array([[-0.86]])  # matches the measured Baffin Bay texel
    assert is_grounded(bed, surface)[0, 0] == False
    assert compute_land_mask(bed, surface)[0, 0] == False


def test_is_grounded_boundary_is_strictly_greater_than_zero():
    """`bed + rho*T > 0`, not `>= 0`: exactly at the flotation boundary
    must NOT be grounded; a hair past it must be."""
    thickness = 1000.0
    bed_at_boundary = -RHO_ICE_OVER_RHO_WATER * thickness  # bed + rho*T == 0
    bed = np.array([[bed_at_boundary]])
    surface = np.array([[bed_at_boundary + thickness]])
    assert is_grounded(bed, surface)[0, 0] == False

    bed2 = np.array([[bed_at_boundary + 0.5]])  # nudge the sum just past 0
    surface2 = np.array([[bed_at_boundary + 0.5 + thickness]])
    assert is_grounded(bed2, surface2)[0, 0] == True


def test_is_grounded_no_ice_open_ocean_is_not_grounded():
    """T=0 (no ice at all): grounded reduces to bed>0, so ordinary open
    ocean (bed deeply negative) is never grounded."""
    bed = np.array([[-4000.0]])
    surface = np.array([[-4000.0]])
    assert is_grounded(bed, surface)[0, 0] == False


# ---------------------------------------------------------------------------
# Registration: R and G must sample the SAME source cells for a given texel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_registration_mask_and_elevation_sample_same_cells(projection):
    """`elev_grid` and `mask_grid` are constructed from the SAME
    unique-per-cell pattern (just differently scaled), so R and G decoded
    from any baked texel must imply the SAME source cell. If the two
    channels' sampling ever drifted apart (e.g. one used a stale dlon/dlat,
    or the grids silently disagreed in shape), this fixture is built so
    that disagreement shows up directly as R/scale != G at some texel."""
    inter_h, inter_w = 40, 80
    idx_grid = (
        np.arange(inter_h)[:, None] * inter_w + np.arange(inter_w)[None, :]
    ).astype(np.float64)
    elev_grid = idx_grid  # "elevation" in meters, unique per cell
    mask_grid = idx_grid / idx_grid.max()  # identical pattern, scaled into [0,1]

    for face in range(FACE_COUNT):
        rgba = bake_elevation_face(face, 8, elev_grid, mask_grid, projection)
        r = rgba[..., 0].view(np.float16).astype(np.float64)
        g = rgba[..., 1].view(np.float16).astype(np.float64)
        assert np.allclose(r / idx_grid.max(), g, atol=1e-2)


def test_registration_mismatched_shapes_rejected():
    """A mask_grid that isn't the SAME shape as elev_grid can't possibly be
    registered -- must fail loudly, not silently broadcast/misalign."""
    elev_grid = np.zeros((40, 80))
    mask_grid = np.zeros((20, 40))  # wrong shape
    with pytest.raises(ValueError):
        bake_elevation_face(0, 8, elev_grid, mask_grid, "plain")


# ---------------------------------------------------------------------------
# R is unchanged: mask_grid content must never leak into the R channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_r_channel_byte_identical_regardless_of_mask_content(projection):
    """The elevation atlas's R channel is already validated (see
    validation.baked_stats in the sidecar) -- baking with the new mask
    input must not perturb it by even one bit, regardless of what the
    mask contains."""
    rng = np.random.default_rng(11)
    inter_h, inter_w = 32, 64
    elev_grid = rng.uniform(-8000, 6000, size=(inter_h, inter_w))
    mask_all_zero = np.zeros((inter_h, inter_w))
    mask_all_one = np.ones((inter_h, inter_w))
    mask_random = rng.uniform(0, 1, size=(inter_h, inter_w))

    for face in range(FACE_COUNT):
        r_bytes = [
            bake_elevation_face(face, 12, elev_grid, m, projection)[..., 0].tobytes()
            for m in (mask_all_zero, mask_all_one, mask_random)
        ]
        assert r_bytes[0] == r_bytes[1] == r_bytes[2]


# ---------------------------------------------------------------------------
# land_mask_stats: area-weighted global fraction arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_land_mask_stats_uniform_fraction_recovers_constant(projection):
    """A uniform G value's area-weighted global average must equal that
    constant exactly (up to float16 rounding), regardless of the
    non-uniform per-texel solid angle weighting -- the weighted mean of a
    constant is the constant."""
    size = 16
    c = 0.4375  # exactly representable in float16
    faces = []
    for _face in range(FACE_COUNT):
        rgba = np.zeros((size, size, 4), dtype=np.float16)
        rgba[..., 1] = np.float16(c)
        faces.append(rgba.view(np.uint16))
    stats = land_mask_stats(faces, size, projection)
    assert stats["global_land_fraction"] == pytest.approx(c, rel=1e-6)


# ---------------------------------------------------------------------------
# assert_land_mask_valid: the hard-fail gate must actually be able to fail
# ---------------------------------------------------------------------------


def test_land_mask_validation_gate_raises_on_all_ocean():
    """An all-zero G channel (no land anywhere) must raise, not warn --
    this is the property the brief calls out explicitly: 'must FAIL the
    bake, not warn'."""
    size = 8
    faces = []
    for _face in range(FACE_COUNT):
        rgba = np.zeros((size, size, 4), dtype=np.float16)
        faces.append(rgba.view(np.uint16))
    with pytest.raises(AssertionError):
        assert_land_mask_valid(faces, size, "plain")


def test_land_mask_validation_gate_raises_on_out_of_range_uniform_fraction():
    """A uniform, in-range-looking-but-wrong global fraction (0.05, far
    outside accepted [0.285, 0.299]) must also raise."""
    size = 8
    faces = []
    for _face in range(FACE_COUNT):
        rgba = np.zeros((size, size, 4), dtype=np.float16)
        rgba[..., 1] = np.float16(0.05)
        faces.append(rgba.view(np.uint16))
    with pytest.raises(AssertionError):
        assert_land_mask_valid(faces, size, "plain")


def test_land_fraction_accepted_range_matches_brief():
    assert LAND_FRACTION_MIN == pytest.approx(0.285)
    assert LAND_FRACTION_MAX == pytest.approx(0.299)


def test_land_mask_probes_include_required_locations():
    names = {p[0] for p in LAND_MASK_PROBES}
    required_substrings = [
        "Vostok",
        "Byrd",
        "Dome A",
        "Greenland",
        "Amazon",
        "Pacific",
        "Atlantic",
        "Baffin",
        "Arctic",
        "Ross",
    ]
    for needle in required_substrings:
        assert any(needle in name for name in names), f"missing probe near {needle!r}"


def test_land_mask_floating_ice_probes_expect_ocean():
    """The three floating-ice regression probes (added after the two
    misclassification bugs -- see compute_land_mask's docstring) must be
    registered as OCEAN expectations, not land -- these are exactly the
    points that were silently wrong under one or both earlier rules."""
    baffin = next(p for p in LAND_MASK_PROBES if "Baffin" in p[0])
    arctic = next(p for p in LAND_MASK_PROBES if "Arctic" in p[0])
    ross = next(p for p in LAND_MASK_PROBES if "Ross" in p[0])
    assert baffin[3] == "ocean"
    assert arctic[3] == "ocean"
    assert ross[3] == "ocean"


# ---------------------------------------------------------------------------
# Geodetic probe -> cube direction: must use geodetic_to_geocentric, not the
# inverse -- this is the "known trap" the task brief calls out explicitly.
# ---------------------------------------------------------------------------


def test_probe_conversion_moves_equatorward_in_northern_hemisphere():
    """A geodetic latitude converted to geocentric must sit CLOSER to the
    equator in the northern hemisphere (smaller magnitude) -- the standard
    WGS84 effect. Getting this backwards (or applying the inverse
    conversion) is the exact class of bug that once silently zeroed an
    entire bake elsewhere in this pipeline."""
    lat_gd = math.radians(72.58)  # Greenland summit probe latitude
    lat_gc = geodetic_to_geocentric_lat(lat_gd)
    assert 0 < lat_gc < lat_gd


def test_geodetic_probe_maps_to_expected_polar_face():
    """Vostok (-78.46 deg lat) is deep in the southern polar region; in
    this app's Z-polar convention that must land on the -Z face (index 5),
    not some equatorial face -- confirms the probe helper wires geodetic
    latitude through geocentric conversion into a direction vector
    correctly, not just that it doesn't crash."""
    face, _u, _v = geodetic_probe_to_face_uv(-78.46, 106.84, "plain")
    assert face == 5


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_geodetic_probe_to_face_uv_round_trips_through_direction(projection):
    """u,v returned for any probe must stay within the valid NDC range."""
    for _name, lat, lon, _expect in LAND_MASK_PROBES:
        face, u, v = geodetic_probe_to_face_uv(lat, lon, projection)
        assert 0 <= face < FACE_COUNT
        assert -1.0 <= u <= 1.0
        assert -1.0 <= v <= 1.0
