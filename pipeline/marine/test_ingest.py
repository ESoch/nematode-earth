import csv
import io
import math
import pytest

from marine import normalize


def table(rows, comment=""):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return f"/* {comment} */\n" + stream.getvalue()


def ccz_row(**changes):
    row = {"Event": "SO239_84-1", "Sample label": "SO239_84_11", "Date/Time": "2015-04-02",
           "Latitude": "11.0788", "Longitude": "-119.6580", "Elevation [m]": "-4431",
           "Depth top [m]": "0", "Depth bot [m]": "0.05", "Station": "IOM control", "Nematoda [#]": "267.300"}
    return row | changes


def haus_row(**changes):
    row = {"Event": "PS57/272-1", "Event 2": "HG_I", "Core": "272-1_A", "Subsample ID": "1",
           "Date/Time": "2000-08-19", "Latitude": "79.138", "Longitude": "6.103", "Elevation [m]": "-1246",
           "Depth sed top [m]": "0", "Depth sed bot [m]": "0.01", "Genus UID": "Acantholaimus",
           "Ident remark": "", "Sample area [m**2]": "0.62832", "Subsample total factor": "0.00604947",
           "Mesh s [µm]": "32", "Method/Device": "MultiCorer", "Ind No [#]": "", "Abund": "50"}
    return row | changes


def test_ccz_reads_explicit_nematodes_not_meiofauna_and_rejects_duplicate():
    assert hasattr(normalize, "parse_ccz"), "CCZ normalization not implemented"
    rows = [ccz_row(), ccz_row(**{"Sample label": "B", "Nematoda [#]": "0"}),
            ccz_row(**{"Sample label": "C", "Nematoda [#]": ""})]
    result = normalize.parse_ccz(table(rows, "individuals per 10 cm**2"))
    assert [r["density_per_m2"] for r in result] == [267300, 0, None]
    assert result[0]["sediment_bottom_cm"] == 5
    assert result[0]["water_depth_m"] == 4431
    assert result[2]["unknown_density"]
    with pytest.raises(ValueError, match="duplicate"):
        normalize.parse_ccz(table([rows[0], rows[0]], "individuals per 10 cm**2"))


def test_haus_published_abundance_sums_genera_without_reapplying_area():
    assert hasattr(normalize, "parse_hausgarten"), "HAUSGARTEN normalization not implemented"
    rows = [haus_row(), haus_row(**{"Genus UID": "Cervonema", "Abund": "25"}),
            haus_row(**{"Core": "B", "Abund": "", "Ind No [#]": "20", "Date/Time": "2010-07-06"})]
    result = normalize.parse_hausgarten(table(rows, "Abundance (#/10 cm2)"))
    assert result[0]["density_per_m2"] == 75000
    assert len(result[0]["source_record_ids"]) == 2
    assert result[1]["density_per_m2"] is None
    assert result[1]["original_quantity"] == 20
    assert result[1]["sampling"]["nominal_syringe_area_cm2"] == pytest.approx(math.pi * 1.1**2)
    assert "area_metadata_conflict" in result[1]["flags"]
    with pytest.raises(ValueError, match="duplicate"):
        normalize.parse_hausgarten(table(rows + [rows[0]], "Abundance (#/10 cm2)"))


def test_haus_distinct_subsample_ids_are_not_silently_combined():
    assert hasattr(normalize, "parse_hausgarten")
    rows = [haus_row(), haus_row(**{"Subsample ID": "2"})]
    slices = normalize.parse_hausgarten(table(rows, "Abundance (#/10 cm2)"))
    assert len(slices) == 2
    cores = normalize.aggregate_cores(slices)
    assert cores[0]["density_per_m2"] is None
    assert "overlapping_sediment_intervals" in cores[0]["exclusion_reasons"]


def test_missing_slice_excludes_whole_core_but_keeps_measured_evidence():
    assert hasattr(normalize, "parse_ccz")
    obs = normalize.parse_ccz(table([ccz_row(**{"Depth bot [m]": "0.01"})], "individuals per 10 cm**2"))
    result = normalize.aggregate_cores(obs)
    assert obs[0]["density_per_m2"] == 267300
    assert result[0]["density_per_m2"] is None
    assert not result[0]["eligible_regional_training"]


def test_core_sampling_metadata_covers_all_component_slices():
    rows = [haus_row(**{"Depth sed bot [m]": "0.02"}),
            haus_row(**{"Depth sed top [m]": "0.02", "Depth sed bot [m]": "0.05", "Subsample ID": "2"})]
    core = normalize.aggregate_cores(normalize.parse_hausgarten(table(rows, "Abundance (#/10 cm2)")))[0]
    assert core["density_per_m2"] == 100000
    assert core["sampling"]["component_slices"] == 2
    assert core["sampling"]["reported_taxon_rows"] == 2
    assert "subsample_id" not in core


def test_canyon_taxonomic_subsample_counts_never_become_density():
    assert hasattr(normalize, "parse_canyon"), "canyon count audit not implemented"
    row = ccz_row() | {"Acantholaimus [#]": "10", "Indet [#] (unidentified)": "0"}
    row.pop("Nematoda [#]")
    result = normalize.parse_canyon(table([row], "Counts are total numbers per core or 25.51759 cm**2"))
    assert result[0]["original_quantity"] == 10
    assert result[0]["density_per_m2"] is None
    assert "identified_subsample_not_population" in result[0]["exclusion_reasons"]


def test_canyon_event_date_is_read_from_archive_metadata():
    row = ccz_row() | {"Acantholaimus [#]": "10"}
    row.pop("Nematoda [#]")
    row.pop("Date/Time")
    result = normalize.parse_canyon(table([row],
        "Event(s):\tSO239_84-1 * LATITUDE: 11 * DATE/TIME: 2006-06-24T00:00:00 * ELEVATION: -740 m"))
    assert result[0]["date"] == "2006-06-24T00:00:00"


def test_haus_station_aliases_do_not_create_independent_stations():
    rows = [haus_row(), haus_row(**{"Core": "B", "Event 2": "HGI"})]
    result = normalize.parse_hausgarten(table(rows, "Abundance (#/10 cm2)"))
    assert result[0]["regional_group"] == result[1]["regional_group"] == "HG_I"
    assert result[1]["station_original"] == "HGI"


def test_duplicate_or_overlapping_observations_cannot_be_summed_into_a_core():
    obs = normalize.parse_ccz(table([ccz_row()], "individuals per 10 cm**2"))
    cores = normalize.aggregate_cores(obs + obs)
    assert not cores[0]["eligible_regional_training"]
    assert cores[0]["density_per_m2"] is None


def test_wei_filters_category_and_preserves_unknown_depth_zero_and_replicates():
    assert hasattr(normalize, "parse_wei"), "Wei KML normalization not implemented"
    def place(category, value):
        return f'''<Placemark><ExtendedData><SchemaData><SimpleData name="CATEGORY">{category}</SimpleData>
        <SimpleData name="STATION">AC1</SimpleData><SimpleData name="REFERENCE">Study</SimpleData>
        <SimpleData name="ABUNDANCE.per.m2">{value}</SimpleData><SimpleData name="DEPTH.m">2440</SimpleData>
        </SchemaData></ExtendedData><Point><coordinates>-94.5731,26.3936,0</coordinates></Point></Placemark>'''
    kml = '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>' + place("nematode", "0") + place("meiofauna", "400") + place("nematode", "") + '</Document></kml>'
    rows = normalize.parse_wei(kml)
    assert len(rows) == 2
    assert [r["density_per_m2"] for r in rows] == [0, None]
    assert rows[0]["latitude"] == 26.3936
    assert rows[0]["sediment_bottom_cm"] is None
    assert all(not r["eligible_regional_training"] for r in rows)
