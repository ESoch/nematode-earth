"""Scientific contract tests: incorrect unit/support/count semantics must fail."""
import pytest

from marine.normalize import density_per_m2, integrate_slices, sum_identifications


@pytest.mark.parametrize("value,expected", [(12.5, 12500), (0, 0), (None, None)])
def test_per_ten_cm2_and_zero_vs_unknown(value, expected):
    assert density_per_m2(value, "individuals/10cm2") == expected


def test_count_requires_area_and_accounted_subsample():
    assert density_per_m2(50, "count", area_cm2=10, fraction=0.5) == 100000
    assert density_per_m2(50, "count", area_cm2=10, fraction=None) is None
    assert density_per_m2(0, "count", area_cm2=None, fraction=1) is None
    assert density_per_m2(50, "identified_specimens", area_cm2=10, fraction=1) is None


@pytest.mark.parametrize("kwargs", [dict(value=-1, unit="individuals/m2"),
    dict(value=float("nan"), unit="individuals/m2"),
    dict(value=2, unit="count", area_cm2=0, fraction=1),
    dict(value=2, unit="count", area_cm2=10, fraction=2)])
def test_reject_invalid_quantity_or_support(kwargs):
    with pytest.raises(ValueError):
        density_per_m2(**kwargs)


def test_genus_sum_is_identification_count_not_population():
    rows = [dict(core="A", top_cm=0, bottom_cm=1, taxon="X", count=5),
            dict(core="A", top_cm=0, bottom_cm=1, taxon="Y", count=7),
            dict(core="B", top_cm=0, bottom_cm=1, taxon="X", count=3)]
    result = sum_identifications(rows)
    assert result[0] == dict(core="A", top_cm=0, bottom_cm=1,
        identified_count=12, density_per_m2=None, population_accounting=False)
    assert result[1]["identified_count"] == 3
    with pytest.raises(ValueError, match="duplicate"):
        sum_identifications(rows + [rows[0]])


def test_slices_sum_areal_densities_only_when_complete():
    slices = [(0, 1, 10000), (1, 3, 20000), (3, 5, 0)]
    assert integrate_slices(slices, 0, 5) == 30000
    assert integrate_slices([(0, 1, 10000), (2, 5, 20000)], 0, 5) is None
    assert integrate_slices([(0, 1, 10000), (1, 5, None)], 0, 5) is None
    with pytest.raises(ValueError, match="overlap|duplicate"):
        integrate_slices(slices + [(0, 1, 10000)], 0, 5)
