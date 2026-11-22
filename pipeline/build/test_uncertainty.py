"""Pytest suite for the B channel: bootstrap coefficient-of-variation, per
the channel ruling -- real per-pixel model uncertainty from
`TotalNumber_Bootstrap_CoefVar.tif` (100 stratified bootstrap iterations),
explicitly NOT `TotalNumber_Ensemble_CoefVar.tif` (seed jitter of one
model, not real uncertainty -- see the ruling for why that file is
never touched).

The bootstrap raster ships on a DIFFERENT native grid than the density
raster (43200x21120 vs 43201x21121) but the SAME origin/pixel-scale
tiepoint (confirmed live against both real headers -- see
`rebake_gb.py`'s report), so `eth_raster.extract_grid`'s existing
box-averaging (unmodified, already covered by `fetch/test_synthetic.py`)
correctly resamples it onto the same 512x256 target grid used for
R/A/G, with no new fetch code needed. What IS new here is the
UN-DILUTION step: `extract_grid`'s box average treats no-data (ocean) as
0 for mass conservation, which is correct for a density but WRONG for a
per-soil-sample statistic like CV -- diluting a coastal box's CV toward 0
because part of the box is ocean would silently understate uncertainty
exactly where the coastline makes the model least reliable.
"""

from __future__ import annotations

import numpy as np
import pytest
from uncertainty import assert_bootstrap_cv_valid, land_mean_from_diluted

# ---------------------------------------------------------------------------
# land_mean_from_diluted: the un-dilution step
# ---------------------------------------------------------------------------


def test_land_mean_fully_land_cell_is_unchanged():
    raw_avg = np.array([0.2])
    land_frac = np.array([1.0])
    got = land_mean_from_diluted(raw_avg, land_frac)
    assert got[0] == pytest.approx(0.2)


def test_land_mean_recovers_true_value_from_diluted_box_average():
    """The exact scenario the un-dilution exists for: a coastal output box
    is half ocean, half land at true CV 0.4. `extract_grid`'s mass-
    conserving box average (NaN->0) reports 0.2 for that box (0.4*0.5 +
    0*0.5). Un-diluting must recover the true land-only value 0.4, not
    leave the diluted 0.2."""
    true_land_cv = 0.4
    land_frac = np.array([0.5])
    raw_avg = np.array([true_land_cv * 0.5])  # what extract_grid would report
    got = land_mean_from_diluted(raw_avg, land_frac)
    assert got[0] == pytest.approx(true_land_cv)


def test_land_mean_ocean_cell_is_zero_not_nan_or_inf():
    """land_frac == 0 (pure ocean, or the same no-data convention as
    R/A/G) must decode to exactly 0.0, never a 0/0 NaN or inf reaching the
    shader -- the same class of requirement as the density bake's
    DENSITY_FLOOR."""
    raw_avg = np.array([0.0])
    land_frac = np.array([0.0])
    got = land_mean_from_diluted(raw_avg, land_frac)
    assert got[0] == 0.0
    assert np.isfinite(got).all()


def test_land_mean_never_decreases_magnitude_vs_diluted_input():
    """Un-diluting (dividing by land_frac in (0,1]) can only hold steady
    or INCREASE magnitude relative to the raw diluted box average --
    never decrease it. This is the mathematical signature that
    distinguishes "correctly un-diluted" from "accidentally reused the
    density convention as-is" (which would just be `raw_avg`, always
    <= the un-diluted value away from land_frac==1)."""
    rng = np.random.default_rng(0)
    raw_avg = rng.uniform(0.0, 1.0, size=50)
    land_frac = rng.uniform(0.01, 1.0, size=50)
    got = land_mean_from_diluted(raw_avg, land_frac)
    assert np.all(got >= raw_avg - 1e-12)


def test_land_mean_no_runaway_amplification_at_tiny_land_fraction():
    """A near-zero land_frac (a lone coastal sliver in an otherwise ocean
    box) must not blow up to an absurd value if the underlying raw CV is
    itself small -- pinned against the real observed behaviour (see
    rebake_gb.py's report: min land_frac among land cells 0.0059, and the
    un-diluted value there was a sane 0.106-0.29, not a runaway
    amplification)."""
    raw_avg = np.array([0.001])  # true_cv=0.17 diluted by land_frac=0.006
    land_frac = np.array([0.0059])
    got = land_mean_from_diluted(raw_avg, land_frac)
    assert got[0] == pytest.approx(0.001 / 0.0059, rel=1e-9)
    assert got[0] < 1.0  # still a sane CV magnitude, not a blow-up


# ---------------------------------------------------------------------------
# assert_bootstrap_cv_valid: the hard-fail gate
# ---------------------------------------------------------------------------


def _valid_fixture():
    land_frac = np.array([[1.0, 0.5, 0.0, 0.8]])
    raw_avg = np.array([[0.3, 0.1, 0.0, 0.24]])  # box averages (diluted)
    b = land_mean_from_diluted(raw_avg, land_frac)
    return b, land_frac, raw_avg


def test_assert_bootstrap_cv_valid_passes_on_well_formed_data():
    b, land_frac, raw_avg = _valid_fixture()
    result = assert_bootstrap_cv_valid(b, land_frac, raw_avg)
    assert result["n_land_cells"] == 3
    assert result["min"] >= 0.0


def test_assert_bootstrap_cv_valid_rejects_negative_values():
    """CV cannot be negative -- a hard, principled requirement (it is a
    ratio of a standard deviation, itself non-negative, to a mean), not a
    data-dependent threshold."""
    b = np.array([[0.3, -0.01, 0.0]])
    land_frac = np.array([[1.0, 1.0, 0.0]])
    raw_avg = np.array([[0.3, -0.01, 0.0]])
    with pytest.raises(AssertionError, match="non-negative"):
        assert_bootstrap_cv_valid(b, land_frac, raw_avg)


def test_assert_bootstrap_cv_valid_rejects_non_finite_values():
    b = np.array([[0.3, np.nan, 0.0]])
    land_frac = np.array([[1.0, 1.0, 0.0]])
    raw_avg = np.array([[0.3, 0.2, 0.0]])
    with pytest.raises(AssertionError, match="finite"):
        assert_bootstrap_cv_valid(b, land_frac, raw_avg)


def test_assert_bootstrap_cv_valid_rejects_un_dilution_that_shrank_values():
    """If B ever comes out SMALLER than the raw diluted box average at a
    land cell, the un-dilution step was skipped or done backwards (e.g.
    multiplying by land_frac instead of dividing) -- this is exactly the
    bug class the ruling's 'shifts the uncertainty field' warning is
    about, and it must hard-fail, not warn."""
    land_frac = np.array([[0.5]])
    raw_avg = np.array([[0.2]])
    b_wrong = raw_avg * land_frac  # backwards: multiplied instead of divided
    with pytest.raises(AssertionError, match="un-dilut"):
        assert_bootstrap_cv_valid(b_wrong, land_frac, raw_avg)


def test_assert_bootstrap_cv_valid_reports_stats_dict():
    b, land_frac, raw_avg = _valid_fixture()
    result = assert_bootstrap_cv_valid(b, land_frac, raw_avg)
    assert set(result) >= {"n_land_cells", "min", "max", "mean", "median"}
    assert result["min"] <= result["median"] <= result["max"]
