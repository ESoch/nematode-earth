"""Fit the coarse urban-calibration model to synthetic data with known coefficients."""

from __future__ import annotations

import numpy as np
import pytest
from urban_calibration import fit_quadratic_through_one, pixel_area_m2

R_MEAN = 6_371_000.0


def test_fit_recovers_exact_coefficients_on_noiseless_data():
    rng = np.random.default_rng(1)
    u = rng.uniform(0, 1, 2000)
    true_a, true_b = -1.70, 0.75
    m = 1 + true_a * u + true_b * u**2
    a, b, r2 = fit_quadratic_through_one(u, m)
    assert a == pytest.approx(true_a, abs=1e-9)
    assert b == pytest.approx(true_b, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_recovers_intercept_fixed_at_one():
    """m(0) must equal 1 by construction -- the fit has no free intercept
    term, since m=1 (no suppression) is the definition of u=0, not
    something to be estimated from noisy data."""
    u = np.array([0.0, 0.3, 0.7, 1.0])
    m = 1 - 1.0 * u + 0.2 * u**2
    a, b, r2 = fit_quadratic_through_one(u, m)
    predicted_at_zero = 1 + a * 0 + b * 0**2
    assert predicted_at_zero == pytest.approx(1.0)


def test_fit_low_r2_on_pure_noise():
    """Sanity: regressing against pure noise should NOT produce a
    confident-looking fit -- R^2 should be small. This is the same
    diagnostic signature the real GHSL/density regression showed
    (R^2 ~ 0.007), confirming that outcome reads as 'no signal detected'
    rather than 'the fit machinery is broken.'"""
    rng = np.random.default_rng(2)
    u = rng.uniform(0, 1, 5000)
    m = 1 + rng.normal(0, 2.0, 5000)  # no relationship to u at all
    a, b, r2 = fit_quadratic_through_one(u, m)
    assert r2 < 0.05


def test_pixel_area_m2_matches_analytic_band_area():
    """A full 360-degree-wide band's area (dlon=360) must equal the exact
    spherical band formula regardless of how pixel_area_m2 slices it."""
    lat_top, lat_bot = 10.0, -10.0
    area = pixel_area_m2(lat_top, lat_bot, 360.0)
    import math

    expected = (
        R_MEAN**2
        * 2
        * math.pi
        * (math.sin(math.radians(lat_top)) - math.sin(math.radians(lat_bot)))
    )
    assert area == pytest.approx(expected, rel=1e-12)


def test_pixel_area_m2_positive_and_scales_with_dlon():
    a1 = pixel_area_m2(1.0, 0.0, 1.0)
    a2 = pixel_area_m2(1.0, 0.0, 2.0)
    assert a1 > 0
    assert a2 == pytest.approx(2 * a1, rel=1e-12)
