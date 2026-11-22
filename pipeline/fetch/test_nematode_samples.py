"""Pins the raw CSV parse and the biome aggregation against the authors' own
published summary (van den Hoogen et al. 2020, *Scientific Data* 7:103,
doi:10.1038/s41597-020-0437-3 -- knitted analysis `20191029_SciData.Rmd`,
github.com/hooge104/2020_global_nematode_dataset) so a future re-fetch that
silently changes the underlying data fails loudly instead of drifting
unnoticed.

Network-dependent: downloads (and caches under `pipeline/cache/`, gitignored)
the real figshare CSVs on first run. Unlike the flaky ETH Research
Collection `eth_raster.py` works around, figshare/ndownloader is a fast,
reliable CDN and the download is verified against the figshare API's own
byte-size + MD5 for each file, so a live fetch here is the right test rather
than a synthetic fixture -- the whole point is pinning against the REAL
dataset's actual content.
"""

from __future__ import annotations

import statistics

import pytest
from nematode_samples import (
    biome_summary,
    clean_biome_index,
    ensure_all_downloaded,
    load_full_dataset,
)


@pytest.fixture(scope="module")
def samples():
    paths = ensure_all_downloaded()
    return load_full_dataset(paths["full_dataset"])


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def test_row_count_matches_published_sample_count(samples):
    # The figshare CC0 release: 6,825 samples. Not to be confused with the
    # 6,759 the Nature *model* used -- see module docstring's provenance
    # note in nematode_samples.py.
    assert len(samples) == 6825


def test_first_row_matches_known_values(samples):
    s = samples[0]
    assert s.sample_id == 1
    assert s.total_number == pytest.approx(3124.0)
    assert s.latitude == pytest.approx(53.09239939)
    assert s.longitude == pytest.approx(6.626211158)
    assert s.wwf_biome_raw == pytest.approx(4.0)
    assert s.data_provider == "Christian Mulder"


def test_last_row_matches_known_values(samples):
    s = samples[-1]
    assert s.sample_id == 20066
    assert s.total_number == pytest.approx(1764.561803)
    assert s.data_provider == "Aidan Keith, Olaf Schmidt"


def test_total_number_never_missing(samples):
    assert all(s.total_number is not None for s in samples)


# ---------------------------------------------------------------------------
# Biome cleaning
# ---------------------------------------------------------------------------


def test_clean_biome_index_rounds_floating_point_noise():
    # Observed in the raw file: codes like 10.99999999 are floating-point
    # representation noise around an integer biome code, not a genuine
    # fractional area-weighted value -- confirmed by an exhaustive scan:
    # max deviation from the nearest integer across all 6,825 rows is
    # 3e-8, and the authors' own script does exactly `round(WWF_Biome, 0)`.
    assert clean_biome_index(10.99999999, latitude=10.0) == 11
    assert clean_biome_index(11.00000003, latitude=10.0) == 11
    assert clean_biome_index(3.999999996, latitude=10.0) == 4
    assert clean_biome_index(4.0, latitude=10.0) == 4


def test_clean_biome_index_applies_antarctic_override():
    # Matches the authors' own cleaning step verbatim (20191029_SciData.Rmd):
    #   full_data_wBiome[full_data_wBiome$Latitude < -60,]$WWF_Biome <- 25
    # Samples below -60 deg get recoded to 25 ("Antarctica") regardless of
    # whatever biome code the source raster extraction produced for them --
    # near-pole raster extraction is unreliable, so the authors just
    # override it rather than trust it.
    assert clean_biome_index(11.0, latitude=-70.0) == 25
    assert clean_biome_index(4.0, latitude=-61.0) == 25
    assert (
        clean_biome_index(4.0, latitude=-60.0) == 4
    )  # boundary: strict '<', not overridden
    assert clean_biome_index(4.0, latitude=-59.9) == 4


def test_all_samples_south_of_60_get_overridden(samples):
    n_antarctic_lat = sum(1 for s in samples if s.latitude < -60)
    assert n_antarctic_lat == 503
    assert all(
        clean_biome_index(s.wwf_biome_raw, s.latitude) == 25
        for s in samples
        if s.latitude < -60
    )


# ---------------------------------------------------------------------------
# Biome aggregation -- the point of the whole task
# ---------------------------------------------------------------------------


def test_biome_summary_matches_authors_published_table(samples):
    summary = {row["biome"]: row for row in biome_summary(samples)}

    tundra = summary["Tundra"]
    assert tundra["n"] == 148
    assert tundra["median"] == pytest.approx(2695, abs=1)
    assert tundra["mean"] == pytest.approx(7298, abs=1)

    broadleaf = summary["Temperate Broadleaf Forests"]
    assert broadleaf["n"] == 2175
    assert broadleaf["median"] == pytest.approx(2120, abs=1)
    assert broadleaf["mean"] == pytest.approx(4465, abs=1)

    assert summary["Tropical Dry Forests"]["n"] == 11
    assert summary["Flooded Grasslands"]["n"] == 7
    assert summary["Antarctica"]["n"] == 503

    # Full reference table (mean/median/n), from the authors' own knitted
    # analysis -- pin every row, not just the headline ones.
    reference = {
        "Tundra": (7298, 2695, 148),
        "Temperate Broadleaf Forests": (4465, 2120, 2175),
        "Boreal Forests": (3959, 2016, 669),
        "Montane Grasslands": (6096, 1120, 116),
        "Tropical Coniferous Forests": (1000, 970, 8),
        "Temperate Conifer Forests": (1800, 670, 158),
        "Tropical Grasslands": (863, 657, 272),
        "Tropical Moist Forests": (914, 601, 968),
        "Temperate Grasslands": (945, 565, 627),
        "Tropical Dry Forests": (430, 431, 11),
        "Mediterranean Forests": (619, 374, 704),
        "Flooded Grasslands": (183, 124, 7),
        "Antarctica": (2245, 89, 503),
        "Deserts": (193, 44, 361),
    }
    assert set(summary) == set(reference)
    for biome, (mean, median, n) in reference.items():
        row = summary[biome]
        assert row["n"] == n, biome
        assert row["mean"] == pytest.approx(mean, abs=1), biome
        assert row["median"] == pytest.approx(median, abs=1), biome


def test_biome_summary_excludes_lake_and_no_match_codes(samples):
    summary = {row["biome"] for row in biome_summary(samples)}
    assert "Lake" not in summary
    assert 0 not in summary and "0" not in summary


def test_total_number_median_and_n_match_published_reference(samples):
    totals = [s.total_number for s in samples]
    assert len(totals) == 6825
    assert statistics.median(totals) == pytest.approx(857, abs=1)
