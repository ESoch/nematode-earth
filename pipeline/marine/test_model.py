import importlib
import math
import pytest


def model():
    spec = importlib.util.find_spec("marine.model")
    assert spec is not None, "grouped held-out model not implemented"
    return importlib.import_module("marine.model")


def observations():
    # ln(1 + density) = 2 + depth_km. Every holdout has a known exact prediction.
    return [dict(observation_id=f"r{i}", core_id=f"c{i}", regional_group=f"g{i // 2}",
                 event=f"e{i // 2}", water_depth_m=1000 * i,
                 density_per_m2=math.expm1(2 + i), source_id="873272",
                 study="CCZ SO239 2015", flags=[],
                 sediment_top_cm=0, sediment_bottom_cm=5,
                 eligible_regional_training=True, latitude=10+i, longitude=20+i,
                 date="2000-01-01") for i in range(1, 9)]


def test_groups_never_cross_folds_and_every_core_is_held_out_once():
    rows = observations()
    folds = model().grouped_folds(rows, "regional_group")
    tested = []
    for train, test in folds:
        assert {rows[i]["regional_group"] for i in train}.isdisjoint({rows[i]["regional_group"] for i in test})
        tested += list(test)
    assert sorted(tested) == list(range(8))


def test_heldout_depth_relation_beats_training_only_intercept_baseline():
    result = model().evaluate(observations(), "regional_group")
    assert result["pooled"]["depth"]["rmse_log1p"] < 1e-10
    assert result["pooled"]["intercept"]["rmse_log1p"] > 1
    assert result["pooled"]["depth"]["multiplicative_mae_shifted"] == pytest.approx(1)
    fold = result["folds"][0]  # g0 holds out only depth 1
    assert fold["intercept_log1p"] == pytest.approx(7)
    assert fold["predictions"][0]["depth_log1p"] == pytest.approx(3)
    assert result["public_layer_allowed"] is False


def test_heldout_labels_cannot_change_their_fitted_prediction():
    rows = observations()
    before = model().evaluate(rows, "regional_group")["folds"][0]
    rows[0]["density_per_m2"] = 1e30
    after = model().evaluate(rows, "regional_group")["folds"][0]
    assert before["coefficients"] == after["coefficients"]
    assert before["intercept_log1p"] == after["intercept_log1p"]
    assert before["predictions"][0]["depth_log1p"] == after["predictions"][0]["depth_log1p"]


def test_duplicate_cores_and_incompatible_support_are_rejected():
    rows = observations()
    with pytest.raises(ValueError, match="duplicate"):
        model().evaluate(rows + [rows[0]], "regional_group")
    rows[0]["sediment_bottom_cm"] = None
    with pytest.raises(ValueError, match="support"):
        model().evaluate(rows, "regional_group")


def test_unknown_excluded_and_measured_zero_remains_zero():
    rows = observations()
    rows[0]["density_per_m2"] = 0
    rows[1]["density_per_m2"] = None
    rows[1]["eligible_regional_training"] = False
    result = model().evaluate(rows, "regional_group")
    assert result["n_cores"] == 7
    assert result["folds"][0]["predictions"][0]["observed_log1p"] == 0


def test_same_core_cannot_be_split_by_conflicting_group_labels():
    rows = observations()
    rows[1]["core_id"] = rows[0]["core_id"]
    with pytest.raises(ValueError, match="core"):
        model().grouped_folds(rows, "regional_group")


@pytest.mark.parametrize("source,study,flags,definition,status", [
    ("873272", "CCZ SO239 2015", [],
     "Publisher-reported total Nematoda abundance per m2 in 0–5 cm sediment",
     "publisher_reported_total_nematoda_abundance"),
    ("973147", "LTER HAUSGARTEN", ["sum_of_reported_genus_abundances", "area_metadata_conflict"],
     "Sum of publisher-reported taxon abundances per m2 in 0–5 cm sediment; not a verified whole-population census",
     "reported_taxon_sum_whole_population_accounting_unverified"),
])
def test_target_definition_and_accounting_survive_every_prediction(source, study, flags, definition, status):
    rows = observations()
    for row in rows:
        row.update(source_id=source, study=study, flags=list(flags))
    # Different non-target flags must survive on their own observations, not bleed across predictions.
    rows[0]["flags"].append("source_specific_caveat")
    result = model().evaluate(rows, "regional_group")
    assert result["target_definition"] == definition
    assert result["population_accounting_status"] == status
    assert result["study"] == study
    assert result["source_id"] == source
    assert set(result["flags"]) == set(flags + ["source_specific_caveat"])
    assert definition in result["model"]
    by_core = {r["core_id"]: r for r in rows}
    for fold in result["folds"]:
        for prediction in fold["predictions"]:
            assert prediction["target_definition"] == definition
            assert prediction["population_accounting_status"] == status
            assert prediction["study"] == study
            assert prediction["source_id"] == source
            assert prediction["flags"] == by_core[prediction["core_id"]]["flags"]


@pytest.mark.parametrize("change", [
    {"flags": ["sum_of_reported_genus_abundances"]},
    {"source_id": "973147", "study": "LTER HAUSGARTEN", "flags": ["sum_of_reported_genus_abundances"]},
    {"study": "A different study"},
])
def test_mixed_target_or_study_cohorts_fail_before_modeling(change):
    rows = observations()
    rows[-1].update(change)
    with pytest.raises(ValueError, match="target|study"):
        model().evaluate(rows, "regional_group")


def test_hausgarten_without_taxon_sum_provenance_is_not_treated_as_total_abundance():
    rows = observations()
    for row in rows:
        row.update(source_id="973147", study="LTER HAUSGARTEN")
    with pytest.raises(ValueError, match="target"):
        model().evaluate(rows, "regional_group")
