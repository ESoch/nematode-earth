"""Verified offline normalization and two deliberately regional experiments."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from .model import evaluate
from .normalize import aggregate_cores, parse_ccz, parse_hausgarten, parse_canyon, parse_wei
from .sources import verify_download


def _write(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def run(out):
    out = Path(out)
    manifest_path = out / "sources.v1.json"
    manifest = json.loads(manifest_path.read_text())
    parsers = {"873272": parse_ccz, "973147": parse_hausgarten,
               "776716": parse_canyon, "wei2010": parse_wei}
    sources = {s["id"]: s for s in manifest["sources"]}
    if set(sources) != set(parsers):
        raise ValueError("four expected dataset sources required")
    observations, counts = [], {}
    for source_id, parse in parsers.items():
        source = sources[source_id]
        verify_download(out / source["landing"]["path"], source["landing"])
        data = verify_download(out / source["data"]["path"], source["data"])
        rows = parse(data.decode("utf-8-sig"))
        for row in rows:
            row.update(source_url=source["landing_url"], license_url=source["license_url"],
                       source_sha256=source["data"]["sha256"])
        observations.extend(rows)
        counts[source_id] = dict(normalized_observations=len(rows),
            source_rows_referenced=len({i for r in rows for i in r["source_record_ids"]}),
            known_density=sum(r["density_per_m2"] is not None for r in rows),
            unknown_density=sum(r["density_per_m2"] is None for r in rows),
            measured_zero=sum(r["density_per_m2"] == 0 for r in rows),
            named_stations=len({r["station"] for r in rows}),
            core_identifiers=len({r["core_id"] for r in rows}),
            station_count_caveat="Wei station names may repeat across references; CCZ labels are areas/treatments, not independent deployments.",
            flags=dict(Counter(f for r in rows for f in r["flags"])))
    cores = aggregate_cores(observations)
    for source_id in ("873272", "973147"):
        subset = [r for r in cores if r["source_id"] == source_id]
        counts[source_id].update(cores_0_5cm=len(subset),
            eligible_cores=sum(r["eligible_regional_training"] for r in subset),
            core_exclusions=dict(Counter(f for r in subset for f in r["exclusion_reasons"])))
    experiments = {
        "ccz_area": evaluate([r for r in cores if r["source_id"] == "873272"], "regional_group"),
        "hausgarten_station": evaluate([r for r in cores if r["source_id"] == "973147"], "regional_group"),
    }
    # Compute everything before publishing any normalized output.
    observation_hash = _write(out / "observations.v1.json", dict(schema_version=1,
        provenance="sources.v1.json; source_record_ids are 1-based TSV data rows or KML placemarks",
        record_grain="reported core-slice/subsample; Wei grain follows original compilation",
        public_layer_allowed=False, observations=observations))
    core_hash = _write(out / "cores-0-5cm.v1.json", dict(schema_version=1,
        transformation="sum explicitly reported areal densities over exactly covered non-overlapping 0–5 cm intervals",
        public_layer_allowed=False, cores=cores))
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob("*.py")) if not p.name.startswith("test_")}
    report = dict(schema_version=1, public_layer_allowed=False,
        deployment_decision="experimental_only_no_public_marine_layer",
        command="uv run --no-sync python -m marine experiment", runtime=dict(python=platform.python_version(), numpy=np.__version__),
        input_sha256={"observations.v1.json": observation_hash, "cores-0-5cm.v1.json": core_hash,
                      "sources.v1.json": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
        code_sha256=code_hashes, experiments=experiments)
    _write(out / "regional-model.v1.json", report)
    _write(out / "audit.v1.json", dict(schema_version=1, counts=counts,
        decisions={
            "873272": "Use Nematoda [#] only; its parameter comment specifies individuals per 10 cm2. Sum IOM slices; exclude the impacted core from the reference experiment. Group by license/APEI area, never random rows.",
            "973147": "Use published Abund (#/10 cm2) values, summing reported genera per event/core/subsample/interval. Missing genera are not individually zero-filled. Preserve 2010/2014/2019 counts but do not infer population density. The 2010 methods state all nematodes were handpicked; this alone does not resolve the archive area discrepancy or establish later years' accounting. Exclude incomplete and overlapping core intervals.",
            "776716": "Archive says total counts per 25.51759 cm2, but the linked paper states 100–150 individuals were selected per slice for nematode analysis. Genus totals are identification counts, not audited population totals. No density conversion.",
            "wei2010": "230 CATEGORY=nematode records retain ABUNDANCE.per.m2 without any area rescaling. No sediment interval is given in the KML; retain measurements as evidence but exclude all from 0–5 cm training. Preserve source date strings verbatim and unknown dates as null.",
        },
        hausgarten_area_audit=dict(reported_total_area_m2=0.62832,
            eight_10cm_cores_geometry_m2=8 * math.pi * 0.05**2,
            reported_subsample_fraction=0.00604947,
            product_area_cm2=0.62832 * 0.00604947 * 10000,
            syringe_2_2cm_geometry_cm2=math.pi * 1.1**2,
            action="retain discrepancy; do not silently correct metadata or convert counts"),
        method_citations=[
            "https://doi.org/10.5194/bg-16-3475-2019",
            "https://doi.org/10.3389/fmars.2023.1271447",
            "https://doi.org/10.1371/journal.pone.0020094",
            "https://doi.org/10.1371/journal.pone.0015323",
        ]))
    print(json.dumps(dict(counts=counts, metrics={k: dict(n_cores=v["n_cores"], n_groups=v["n_groups"],
        pooled=v["pooled"], equal_group=v["equal_group"]) for k, v in experiments.items()}), indent=2))
