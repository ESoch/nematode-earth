"""Offline end-to-end regression using the checksummed publisher snapshot."""
import importlib
import json
from pathlib import Path
import pytest


def test_real_downloads_produce_measured_artifacts_and_heldout_predictions(tmp_path):
    cached = Path(__file__).resolve().parents[1] / "out/marine"
    if not (cached / "sources.v1.json").exists():
        pytest.skip("run python -m marine fetch to enable the archived-source integration test")
    spec = importlib.util.find_spec("marine.experiment")
    assert spec is not None, "measured end-to-end experiment not implemented"
    (tmp_path / "raw").symlink_to(cached / "raw", target_is_directory=True)
    (tmp_path / "sources.v1.json").write_bytes((cached / "sources.v1.json").read_bytes())
    importlib.import_module("marine.experiment").run(tmp_path)
    observations = json.loads((tmp_path / "observations.v1.json").read_text())["observations"]
    assert len([r for r in observations if r["source_id"] == "wei2010"]) == 230
    assert all(r["density_per_m2"] is None for r in observations if r["source_id"] == "776716")
    cores = json.loads((tmp_path / "cores-0-5cm.v1.json").read_text())["cores"]
    core = next(r for r in cores if r["core_id"] == "873272:SO239_84-1:SO239_84_11")
    assert core["density_per_m2"] == pytest.approx(409524)
    report = json.loads((tmp_path / "regional-model.v1.json").read_text())
    assert report["public_layer_allowed"] is False
    assert report["experiments"]["ccz_area"]["n_cores"] == 31
    assert report["experiments"]["hausgarten_station"]["n_groups"] == 3
    haus = report["experiments"]["hausgarten_station"]
    assert haus["population_accounting_status"] == "reported_taxon_sum_whole_population_accounting_unverified"
    assert "not a verified whole-population census" in haus["target_definition"]
    assert "sum_of_reported_genus_abundances" in haus["flags"]
    for fold in haus["folds"]:
        for prediction in fold["predictions"]:
            assert prediction["target_definition"] == haus["target_definition"]
            assert prediction["population_accounting_status"] == haus["population_accounting_status"]
            assert prediction["study"] == "LTER HAUSGARTEN"
            assert "sum_of_reported_genus_abundances" in prediction["flags"]
            assert "area_metadata_conflict" in prediction["flags"]
    for experiment in report["experiments"].values():
        assert sum(len(f["predictions"]) for f in experiment["folds"]) == experiment["n_cores"]
        for fold in experiment["folds"]:
            assert set(fold["train_core_ids"]).isdisjoint(fold["test_core_ids"])
