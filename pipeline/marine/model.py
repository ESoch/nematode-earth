"""Small regional log-density experiment, with whole groups held out."""
import math
import numpy as np


def _target_context(rows):
    """Keep the audited response definition attached to every exported result."""
    source = rows[0]["source_id"]
    definitions = {
        "873272": (
            "Publisher-reported total Nematoda abundance per m2 in 0–5 cm sediment",
            "publisher_reported_total_nematoda_abundance",
        ),
        "973147": (
            "Sum of publisher-reported taxon abundances per m2 in 0–5 cm sediment; not a verified whole-population census",
            "reported_taxon_sum_whole_population_accounting_unverified",
        ),
    }
    if source not in definitions:
        raise ValueError("unaudited model target source")
    studies = {r.get("study") for r in rows}
    if len(studies) != 1 or not rows[0].get("study"):
        raise ValueError("mixed or missing study definition")
    for row in rows:
        flags = row.get("flags")
        if not isinstance(flags, list) or any(not isinstance(f, str) for f in flags):
            raise ValueError("missing or malformed target provenance flags")
        taxon_sum = "sum_of_reported_genus_abundances" in flags
        if taxon_sum != (source == "973147"):
            raise ValueError("mixed or inconsistent model target accounting")
    definition, status = definitions[source]
    return dict(source_id=source, study=rows[0]["study"],
                target_definition=definition, population_accounting_status=status)


def grouped_folds(rows, group_field):
    core_groups = {}
    for row in rows:
        group = row[group_field]
        if not group:
            raise ValueError("missing held-out group")
        previous = core_groups.setdefault(row["core_id"], group)
        if previous != group:
            raise ValueError("same core assigned to multiple holdout groups")
    groups = sorted(set(core_groups.values()))
    if len(groups) < 3:
        raise ValueError("at least three independent groups required")
    return [(np.array([i for i, r in enumerate(rows) if r[group_field] != group]),
             np.array([i for i, r in enumerate(rows) if r[group_field] == group]))
            for group in groups]


def _metrics(errors):
    errors = np.asarray(errors)
    mae = float(np.mean(np.abs(errors)))
    return dict(rmse_log1p=float(np.sqrt(np.mean(errors**2))), mae_log1p=mae,
                multiplicative_mae_shifted=math.exp(mae), bias_log1p=float(np.mean(errors)))


def evaluate(observations, group_field):
    rows = [r for r in observations if r["eligible_regional_training"]]
    ids = [r["core_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate core in model targets")
    if len({r["source_id"] for r in rows}) != 1 or any(
        r["sediment_top_cm"] != 0 or r["sediment_bottom_cm"] != 5 for r in rows
    ):
        raise ValueError("incompatible study or sediment support")
    target = _target_context(rows)
    if any(r["density_per_m2"] is None or not math.isfinite(r["density_per_m2"]) or
           r["density_per_m2"] < 0 or r["water_depth_m"] is None or
           not math.isfinite(r["water_depth_m"]) or r["water_depth_m"] <= 0 for r in rows):
        raise ValueError("invalid model quantity or depth")
    x = np.array([r["water_depth_m"] / 1000 for r in rows])
    y = np.log1p([r["density_per_m2"] for r in rows])
    design = np.column_stack((np.ones(len(x)), x))
    folds, errors, baseline_errors = [], [], []
    for train, test in grouped_folds(rows, group_field):
        coefficients, _, rank, _ = np.linalg.lstsq(design[train], y[train], rcond=None)
        if rank != 2:
            raise ValueError("training depth has no variation")
        intercept = float(np.mean(y[train]))
        predictions = design[test] @ coefficients
        residuals, baseline = predictions - y[test], intercept - y[test]
        errors.extend(residuals)
        baseline_errors.extend(baseline)
        folds.append(dict(held_out_group=rows[int(test[0])][group_field],
            train_core_ids=[ids[int(i)] for i in train], test_core_ids=[ids[int(i)] for i in test],
            coefficients=coefficients.tolist(), intercept_log1p=intercept,
            train_water_depth_m=[float(x[train].min()*1000), float(x[train].max()*1000)],
            depth=_metrics(residuals), intercept=_metrics(baseline),
            predictions=[dict(**target, flags=list(rows[int(i)]["flags"]),
                core_id=ids[int(i)], observed_log1p=float(y[i]),
                observed_density_per_m2=rows[int(i)]["density_per_m2"],
                depth_log1p=float(predictions[j]), intercept_log1p=intercept,
                outside_training_depth=bool(x[i] < x[train].min() or x[i] > x[train].max()))
                for j, i in enumerate(test)]))
    pooled = dict(depth=_metrics(errors), intercept=_metrics(baseline_errors))
    macro = {}
    for name in ("depth", "intercept"):
        mae = float(np.mean([f[name]["mae_log1p"] for f in folds]))
        macro[name] = dict(rmse_log1p=float(np.sqrt(np.mean([f[name]["rmse_log1p"]**2 for f in folds]))),
                           mae_log1p=mae, multiplicative_mae_shifted=math.exp(mae))
    return dict(schema_version=1, **target,
        flags=sorted({flag for r in rows for flag in r["flags"]}),
        n_cores=len(rows), n_groups=len(folds),
        n_events=len({r["event"] for r in rows}), grouping=group_field,
        model="OLS ln(1 + target_per_m2) = intercept + slope * water_depth_km; target: " + target["target_definition"],
        baseline="mean ln(1 + target_per_m2) computed from training cores only",
        weighting="one complete core per target; pooled core-weighted and equal-group macro metrics",
        multiplicative_interpretation="exp(MAE) is geometric mean absolute fold error of (density + 1), not an uncertainty interval",
        folds=folds, pooled=pooled, equal_group=macro,
        improves_pooled_rmse=pooled["depth"]["rmse_log1p"] < pooled["intercept"]["rmse_log1p"],
        improves_equal_group_rmse=macro["depth"]["rmse_log1p"] < macro["intercept"]["rmse_log1p"],
        support=dict(water_depth_m=[min(r["water_depth_m"] for r in rows), max(r["water_depth_m"] for r in rows)],
                     latitude=[min(r["latitude"] for r in rows), max(r["latitude"] for r in rows)],
                     longitude=[min(r["longitude"] for r in rows), max(r["longitude"] for r in rows)],
                     date=[min(r["date"] for r in rows), max(r["date"] for r in rows)],
                     sediment_interval_cm=[0, 5]),
        public_layer_allowed=False, deployment_decision="experimental_only_no_public_layer",
        uncertainty_intervals=None,
        limitations=["Few geographic groups; regional validation is not global transfer validation.",
                     "Water depth alone omits food supply, sediment, oxygen, disturbance, and temporal covariates.",
                     "Within-station/deployment folds cannot establish spatial transfer.",
                     "Replicates and years are not independent stations; no inferential confidence intervals."])
