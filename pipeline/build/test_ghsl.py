"""Pytest suite for the GHSL built-up-fraction pipeline: box-sum area
decimation (must conserve total built-up area exactly, unlike a plain
mean), and the cube bake's mass-conservation gate on a synthetic fixture
with a known answer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "fetch"))
from bake_ghsl_cube import (
    assert_built_up_area_valid,
    bake_ghsl_face,
    baked_total_built_up_area,
)
from eac import FACE_COUNT
from ghsl import decimate_built_up_fraction, total_built_up_area_m2

R_MEAN = 6_371_000.0


def _uniform_area_fixture(fraction: float, in_h=200, in_w=400, lat0=80.0):
    """A native-resolution area grid where every cell is exactly
    `fraction` of its own physical area -- so decimating it (box-sum then
    normalize) should reproduce `fraction` everywhere, and the total
    built-up area is exactly `fraction * total grid area`, known
    analytically."""
    dlat = 160.0 / in_h  # matches GHSL's own "clips a bit short of the poles" shape
    dlon = 360.0 / in_w
    lat_edges = lat0 - np.arange(in_h + 1) * dlat
    band_area = (
        R_MEAN**2
        * math.radians(dlon)
        * (np.sin(np.radians(lat_edges[:-1])) - np.sin(np.radians(lat_edges[1:])))
    )
    area = np.broadcast_to((band_area * fraction)[:, None], (in_h, in_w)).copy()
    return area, lat0, dlon, dlat


def test_decimate_conserves_total_area_at_various_output_sizes():
    """Box-sum decimation must conserve total built-up area exactly
    (every native cell counted once), regardless of output resolution."""
    fraction = 0.15
    area, lat0, dlon, dlat = _uniform_area_fixture(fraction)
    expected_total = float(area.sum())
    for out_h, out_w in [(20, 40), (50, 100), (100, 200)]:
        frac_grid = decimate_built_up_fraction(
            area, -180.0, lat0, dlon, dlat, out_h, out_w
        )
        recovered_total = total_built_up_area_m2(
            frac_grid, lat0, dlat * area.shape[0], 360.0
        )
        assert recovered_total == pytest.approx(expected_total, rel=1e-9)


def test_decimate_reproduces_uniform_fraction():
    area, lat0, dlon, dlat = _uniform_area_fixture(0.3)
    frac_grid = decimate_built_up_fraction(area, -180.0, lat0, dlon, dlat, 40, 80)
    assert np.allclose(frac_grid, 0.3, atol=1e-9)


def test_decimate_conserves_area_with_nonuniform_input():
    """A checkerboard input (half cells 0, half cells at their full pixel
    area) must still conserve the exact total when decimated, even though
    individual output cells now average a genuine mix."""
    in_h, in_w, lat0 = 60, 120, 40.0
    dlat, dlon = 80.0 / in_h, 360.0 / in_w
    lat_edges = lat0 - np.arange(in_h + 1) * dlat
    band_area = (
        R_MEAN**2
        * math.radians(dlon)
        * (np.sin(np.radians(lat_edges[:-1])) - np.sin(np.radians(lat_edges[1:])))
    )
    area = np.zeros((in_h, in_w))
    checkerboard = (np.arange(in_h)[:, None] + np.arange(in_w)[None, :]) % 2 == 0
    area[checkerboard] = np.broadcast_to(band_area[:, None], (in_h, in_w))[checkerboard]
    expected_total = float(area.sum())

    frac_grid = decimate_built_up_fraction(area, -180.0, lat0, dlon, dlat, 15, 30)
    recovered_total = total_built_up_area_m2(frac_grid, lat0, dlat * in_h, 360.0)
    assert recovered_total == pytest.approx(expected_total, rel=1e-9)


@pytest.mark.parametrize("projection", ["plain", "eac"])
def test_ghsl_cube_bake_mass_conservation_uniform_fraction(projection):
    """Bake a uniform-fraction intermediate grid through the real cube
    bake and confirm the quantized, decoded, re-integrated total built-up
    area recovers the analytic answer: fraction * (area of the sphere
    within the grid's own lat coverage)."""
    fraction = 0.08
    inter_h, inter_w, lat0 = 64, 128, 85.0
    grid_lat_min = -80.0  # deliberately short of full +/-90, like real GHSL
    dlat = (lat0 - grid_lat_min) / inter_h
    dlon = 360.0 / inter_w
    frac_grid = np.full((inter_h, inter_w), fraction, dtype=np.float64)

    size = 48
    faces = []
    for face in range(FACE_COUNT):
        rgba_u16, _n_beyond = bake_ghsl_face(
            face, size, frac_grid, -180.0, lat0, dlon, dlat, grid_lat_min, projection
        )
        faces.append(rgba_u16)
    baked_total = baked_total_built_up_area(faces, size, projection)

    quantized_fraction = float(np.float16(fraction))
    expected = (
        quantized_fraction
        * R_MEAN**2
        * 2
        * math.pi
        * (math.sin(math.radians(lat0)) - math.sin(math.radians(grid_lat_min)))
    )
    # Not rel=1e-6: this fixture's coverage window (85 to -80) isn't a
    # round fraction of the size=48 discretization, so a handful of
    # texels straddle the boundary in a way that doesn't cancel exactly.
    # Plain gnomonic's much larger texel-size variation near the poles
    # (5.196:1 vs EAC's 1.406:1) makes its boundary slop bigger too --
    # observed ~1.5e-5 relative for eac, ~8.3e-4 for plain, both utterly
    # unlike the 2-3-order-of-magnitude errors the real bugs this session
    # produced, which is the distinction that matters here.
    assert baked_total == pytest.approx(expected, rel=3e-3)


def test_ghsl_cube_bake_beyond_coverage_is_zero_not_extrapolated():
    """Texels beyond the grid's actual lat coverage must decode to
    exactly 0 built-up fraction (real: no cities there), never a value
    carried over from clamping into unmeasured territory."""
    inter_h, inter_w, lat0 = 64, 128, 85.0
    grid_lat_min = -80.0
    dlat = (lat0 - grid_lat_min) / inter_h
    dlon = 360.0 / inter_w
    frac_grid = np.full((inter_h, inter_w), 0.5, dtype=np.float64)

    size = 64
    found_beyond = False
    for face in (4, 5):  # +Z/-Z poles, per this app's Z-polar convention
        rgba_u16, n_beyond = bake_ghsl_face(
            face, size, frac_grid, -180.0, lat0, dlon, dlat, grid_lat_min, "eac"
        )
        if n_beyond > 0:
            found_beyond = True
            r = rgba_u16[..., 0].view(np.float16).astype(np.float64)
            a = rgba_u16[..., 3].view(np.float16).astype(np.float64)
            # Can't cheaply isolate exactly which texels without redoing
            # the geometry, but at least the pole-center texel (near u=v=0)
            # on a size=64 EAC face is deep enough into the gap to check.
            center = size // 2
            assert r[center, center] == 0.0
    assert found_beyond, "fixture should produce at least one beyond-coverage texel"


# ---------------------------------------------------------------------------
# assert_built_up_area_valid: hard-fail gate against a wholesale-zeroing
# regression. The bake previously had exactly this bug once already -- an
# earlier version of bake_ghsl_face compared `abs(lat) > grid_lat_min`
# against a SIGNED, negative grid_lat_min, which is true for every latitude
# on Earth once the bound is negative, silently zeroing the entire bake
# (see bake_ghsl_face's docstring). That version had no assertion to catch
# it -- ratio_baked_to_source and texels_beyond_coverage were only ever
# written to the meta JSON for a human to notice, never gated. This is the
# gate that would have caught it (n_beyond == n_texels_total, and
# baked_total_area == 0), now added ahead of baking a finer tier.
# ---------------------------------------------------------------------------


def _n_texels_total(face_size: int) -> int:
    return FACE_COUNT * face_size * face_size


def test_assert_built_up_area_valid_passes_on_healthy_bake():
    # Matches the real shipped urban_plain_512 bake's order of magnitude:
    # ratio very close to 1, a small sliver of texels beyond coverage.
    result = assert_built_up_area_valid(
        ratio_baked_to_source=0.9989,
        baked_total_area_m2=4.64e11,
        n_beyond=104,
        face_size=512,
    )
    assert result["passed"] is True


def test_assert_built_up_area_valid_rejects_zero_total_area():
    """The most direct possible signature of a wholesale-zeroing bug:
    baked_total_area_m2 == 0. Must hard-fail even if, hypothetically,
    something upstream also zeroed source_total_area and the ratio
    happened to look benign (0/0 or NaN) -- this check does not depend on
    the ratio at all."""
    with pytest.raises(AssertionError, match="zero|positive"):
        assert_built_up_area_valid(
            ratio_baked_to_source=1.0,  # would look "perfect" on a naive ratio-only gate
            baked_total_area_m2=0.0,
            n_beyond=0,
            face_size=64,
        )


def test_assert_built_up_area_valid_rejects_all_beyond_coverage():
    """The exact historical bug signature: every single texel flagged
    'beyond coverage' (the sign-flipped abs(lat) > grid_lat_min
    comparison, true everywhere once grid_lat_min is negative). Must
    hard-fail on n_beyond == n_texels_total regardless of what the ratio
    says."""
    size = 64
    with pytest.raises(AssertionError, match="beyond coverage"):
        assert_built_up_area_valid(
            ratio_baked_to_source=1.0,
            baked_total_area_m2=1.0,  # nonzero on its own would pass the previous check
            n_beyond=_n_texels_total(size),
            face_size=size,
        )


def test_assert_built_up_area_valid_rejects_far_too_many_beyond_coverage():
    """GHSL's real polar gap is tiny (0.0066% of texels at size=512, see
    the shipped meta) -- a much larger fraction beyond coverage would mean
    something silently ate more of the globe than just the poles, even if
    it isn't literally 100%."""
    size = 64
    with pytest.raises(AssertionError, match="beyond coverage"):
        assert_built_up_area_valid(
            ratio_baked_to_source=1.0,
            baked_total_area_m2=1.0,
            n_beyond=int(
                _n_texels_total(size) * 0.5
            ),  # half the globe, absurd for GHSL
            face_size=size,
        )


def test_assert_built_up_area_valid_rejects_ratio_far_from_one():
    with pytest.raises(AssertionError, match="ratio"):
        assert_built_up_area_valid(
            ratio_baked_to_source=0.3,
            baked_total_area_m2=1.0,
            n_beyond=0,
            face_size=64,
        )


def test_assert_built_up_area_valid_returns_diagnostics_dict():
    result = assert_built_up_area_valid(
        ratio_baked_to_source=1.001,
        baked_total_area_m2=1.0,
        n_beyond=10,
        face_size=64,
    )
    assert set(result) >= {
        "passed",
        "ratio_baked_to_source",
        "beyond_coverage_fraction",
    }
    assert result["beyond_coverage_fraction"] == pytest.approx(10 / _n_texels_total(64))
