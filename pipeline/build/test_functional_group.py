"""Pytest suite for the G channel: structure fraction
`(omnivores + predators) / (sum of the five functional-group grids)`, per
the channel ruling (see the team-lead's binding ruling document -- G is a
continuous structure-fraction scalar, NOT a categorical dominant-group
index, and the denominator is the SUM OF THE FIVE GROUPS, not
`TotalNumber`, because the two do not reconcile on real data -- see
`test_summation_ratio_*` below and `rebake_gb.py`'s real-data report for
the measured numbers).

Small synthetic grids only here (fixture-driven, no network) -- exercising
`functional_group.py`'s pure functions exactly the way `test_bake.py`
exercises `bake_cube.py`'s pure functions.
"""

from __future__ import annotations

import numpy as np
import pytest
from functional_group import (
    global_structure_fraction,
    global_summation_ratio,
    group_sum,
    structure_fraction,
    summation_ratio,
)

# ---------------------------------------------------------------------------
# group_sum
# ---------------------------------------------------------------------------


def test_group_sum_is_elementwise_total():
    bact = np.array([1.0, 2.0])
    fungi = np.array([3.0, 0.0])
    herb = np.array([5.0, 1.0])
    omni = np.array([2.0, 4.0])
    pred = np.array([1.0, 3.0])
    total = group_sum(bact, fungi, herb, omni, pred)
    assert np.array_equal(total, np.array([12.0, 10.0]))


# ---------------------------------------------------------------------------
# structure_fraction: the actual G channel formula
# ---------------------------------------------------------------------------


def test_structure_fraction_basic_ratio():
    omni = np.array([2.0, 4.0])
    pred = np.array([1.0, 3.0])
    sum_five = np.array([10.0, 14.0])
    g, n_clipped, extreme = structure_fraction(omni, pred, sum_five)
    assert g == pytest.approx([0.3, 0.5])
    assert n_clipped == 0
    assert extreme == 0.0


def test_structure_fraction_zero_denominator_is_zero_not_nan():
    """Ocean / no-data cells (sum_five == 0, matching extract_grid's
    NaN->0 mass-conserving convention -- see eth_raster.py) must decode to
    exactly 0.0, never NaN or inf. This is the same class of bug the
    density bake's DENSITY_FLOOR exists to prevent (see bake_cube.py)."""
    omni = np.array([0.0, 5.0])
    pred = np.array([0.0, 2.0])
    sum_five = np.array([0.0, 10.0])
    g, n_clipped, extreme = structure_fraction(omni, pred, sum_five)
    assert g[0] == 0.0
    assert np.isfinite(g).all()
    assert g[1] == pytest.approx(0.7)


def test_structure_fraction_is_dilution_invariant():
    """The whole point of a per-pixel RATIO (vs. a density): scaling every
    one of the five groups by the same land-fraction dilution factor (as
    happens automatically at a mixed coastal box -- see
    eth_raster.extract_grid's box-average-including-ocean-as-0 convention)
    must leave G unchanged, because the dilution factor cancels between
    numerator and denominator. Confirmed on real data in
    rebake_gb.py's report (land_frac identical across all 5 group files
    to floating-point precision), but this pins the mathematical property
    directly."""
    omni = np.array([4.0])
    pred = np.array([2.0])
    sum_five = np.array([20.0])
    g_full, _, _ = structure_fraction(omni, pred, sum_five)
    for dilution in (0.5, 0.1, 0.01):
        g_diluted, _, _ = structure_fraction(
            omni * dilution, pred * dilution, sum_five * dilution
        )
        assert g_diluted == pytest.approx(g_full)


def test_structure_fraction_clips_out_of_range_and_reports_it():
    """Random-forest regression can predict slightly negative per-group
    counts near zero; if that ever pushes the raw ratio outside [0,1], G
    must be clipped (matching the pipeline's existing defensive-clip
    convention on R/A/land-fraction channels) AND the clip must be
    reported, not silently absorbed -- scientific honesty per the task
    brief."""
    # omni+pred slightly exceeds sum_five because one of the other three
    # groups (not passed here, folded into sum_five) predicted marginally
    # negative.
    omni = np.array([6.0])
    pred = np.array([5.0])
    sum_five = np.array([10.0])  # omni+pred = 11 > 10
    g, n_clipped, extreme = structure_fraction(omni, pred, sum_five)
    assert g[0] == 1.0
    assert n_clipped == 1
    assert extreme == pytest.approx(0.1)  # raw ratio was 1.1, 0.1 over


def test_structure_fraction_matches_real_data_range():
    """Real-data sanity pin: on the actual fetched 512x256 grids (see
    rebake_gb.py), G over land ranges [0.0246, 0.4443] with zero
    out-of-[0,1] cells before clipping. This test doesn't refetch the
    real data (no network in the test suite) -- it just pins that a
    plausible land-cell input in that observed range round-trips with no
    clipping, catching a regression that would start clipping data that
    never needed it."""
    omni = np.array([100.0, 2728.0])
    pred = np.array([50.0, 2004.0])
    sum_five = np.array([6000.0, 12973.75])
    g, n_clipped, extreme = structure_fraction(omni, pred, sum_five)
    assert n_clipped == 0
    assert extreme == 0.0
    assert np.all((g >= 0.0) & (g <= 1.0))


# ---------------------------------------------------------------------------
# summation_ratio / global_summation_ratio: the "does sum-of-five equal
# TotalNumber" trap the ruling requires checking before picking a
# denominator.
# ---------------------------------------------------------------------------


def test_summation_ratio_undefined_where_total_is_zero():
    sum_five = np.array([0.0, 10.0])
    total = np.array([0.0, 8.0])
    ratio = summation_ratio(sum_five, total)
    assert np.isnan(ratio[0])
    assert ratio[1] == pytest.approx(1.25)


def test_summation_ratio_one_means_groups_reconcile_to_total():
    sum_five = np.array([10.0])
    total = np.array([10.0])
    assert summation_ratio(sum_five, total)[0] == pytest.approx(1.0)


def test_global_summation_ratio_area_weighted():
    sum_five = np.array([[10.0, 20.0]])
    total = np.array([[5.0, 40.0]])  # ratios 2.0 and 0.5
    area = np.array([[1.0, 1.0]])  # equal weight -> not a plain mean of ratios
    # area-weighted: sum(area*sum_five)/sum(area*total) = 30/45 = 0.6667,
    # NOT the unweighted mean-of-ratios (1.25) -- confirms this is a
    # pooled ratio-of-sums, matching how the structure-fraction global
    # check is computed (and how Table 1's published percentages are
    # themselves pooled across individuals, not averaged per-site).
    got = global_summation_ratio(sum_five, total, area)
    assert got == pytest.approx(30.0 / 45.0)


def test_global_summation_ratio_ignores_zero_total_cells():
    sum_five = np.array([[0.0, 20.0]])
    total = np.array([[0.0, 40.0]])
    area = np.array([[1.0, 1.0]])
    got = global_summation_ratio(sum_five, total, area)
    assert got == pytest.approx(0.5)  # the zero-total ocean cell drops out cleanly


# ---------------------------------------------------------------------------
# global_structure_fraction: the headline number compared against Table 1's
# published 13.4% (8.9% omnivores + 4.5% predators).
# ---------------------------------------------------------------------------


def test_global_structure_fraction_pools_by_area_and_abundance():
    # Two cells: one huge-area near-equator cell dominated by
    # bacterivores (low structure fraction), one tiny-area cell that's
    # ALL omnivores/predators (structure fraction 1.0). Pooling by
    # area*abundance must land close to the big cell's own fraction, not
    # a naive 50/50 average of the two cells' fractions.
    omni = np.array([[1.0, 45.0]])
    pred = np.array([[1.0, 45.0]])
    sum_five = np.array([[100.0, 90.0]])
    area = np.array([[1000.0, 0.001]])
    got = global_structure_fraction(omni, pred, sum_five, area)
    naive_mean = ((1.0 + 1.0) / 100.0 + (45.0 + 45.0) / 90.0) / 2  # 0.51
    assert got == pytest.approx(0.02, abs=1e-4)
    assert got != pytest.approx(naive_mean, abs=0.1)


def test_global_structure_fraction_zero_cells_drop_out():
    """Ocean cells (sum_five == 0 everywhere) must contribute nothing to
    either sum, never a division warning or a NaN poisoning the global
    figure."""
    omni = np.array([[0.0, 4.0]])
    pred = np.array([[0.0, 2.0]])
    sum_five = np.array([[0.0, 10.0]])
    area = np.array([[500.0, 1.0]])
    got = global_structure_fraction(omni, pred, sum_five, area)
    assert got == pytest.approx(0.6)
    assert np.isfinite(got)


def test_global_structure_fraction_real_data_pin():
    """Pin against the actual measured real-data result (see rebake_gb.py's
    printed report and its own hard gate): running this exact function over
    the real 512x256 extraction gives Sum(area*(omni+pred)) =
    4.27600376625101e16 and Sum(area*sum_five) = 3.140683690391254e17,
    ratio 0.13614881942212786 (13.61%) -- within 1.6% relative of Table 1's
    published 13.4% (8.9% omnivores + 4.5% predators). Reconstructed here
    as a single-cell grid carrying exactly those pooled sums (area=1, so
    the weighted sum equals the raw value) -- isolates
    global_structure_fraction's arithmetic from the extraction step, which
    is not repeated in the test suite (no network)."""
    omni = np.array([[4.27600376625101e16]])
    pred = np.array([[0.0]])
    sum_five = np.array([[3.140683690391254e17]])
    area = np.array([[1.0]])
    got = global_structure_fraction(omni, pred, sum_five, area)
    assert got == pytest.approx(0.13614881942212786, rel=1e-9)
    published = 0.134
    assert abs(got - published) / published < 0.10  # matches rebake_gb.py's gate
