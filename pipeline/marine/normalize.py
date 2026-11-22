"""Conservative unit and sampling-support transformations."""
from collections import defaultdict
import math
import csv
import io
import xml.etree.ElementTree as ET
import re


def number(value):
    if value is None or str(value).strip() in {"", "NA", "N/A", "NaN"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite numeric field")
    return result


def _table(text):
    if "*/" not in text:
        raise ValueError("missing PANGAEA metadata")
    metadata, data = text.split("*/", 1)
    reader = csv.DictReader(io.StringIO(data.lstrip()), delimiter="\t")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("duplicate column names")
    rows = list(reader)
    if any(None in r or None in r.values() for r in rows):
        raise ValueError("ragged table")
    return metadata, rows


def _base(source, study, row, index, core, station, top, bottom):
    elevation = number(row.get("Elevation [m]"))
    return dict(schema_version=1, observation_id=f"{source}:row:{index}", source_id=source,
        source_record_ids=[f"{source}:row:{index}"], study=study, station=station,
        event=row.get("Event"), core_id=f"{source}:{row.get('Event')}:{core}",
        date=row.get("Date/Time") or None, latitude=number(row.get("Latitude")),
        longitude=number(row.get("Longitude")), water_depth_m=None if elevation is None else -elevation,
        sediment_top_cm=None if top is None else top * 100,
        sediment_bottom_cm=None if bottom is None else bottom * 100,
        density_per_m2=None, original_quantity=None, original_unit=None,
        unknown_density=True, sampling={}, flags=[], exclusion_reasons=[],
        eligible_regional_training=False, eligibility_scope="slice_evidence_requires_complete_core_aggregation")


def _finish(obs, density, quantity, unit, reasons=()):
    obs.update(density_per_m2=density, original_quantity=quantity, original_unit=unit,
               unknown_density=density is None, exclusion_reasons=list(reasons))
    if density is None:
        obs["flags"].append("unknown_population_density")
    return obs


def parse_ccz(text):
    metadata, rows = _table(text)
    if "individuals per 10 cm**2" not in metadata:
        raise ValueError("CCZ unit declaration changed")
    output, seen = [], set()
    for i, r in enumerate(rows, 1):
        top, bottom = number(r["Depth top [m]"]), number(r["Depth bot [m]"])
        key = (r["Event"], r["Sample label"], top, bottom)
        if key in seen:
            raise ValueError(f"duplicate CCZ core slice: {key}")
        seen.add(key)
        obs = _base("873272", "CCZ SO239 2015", r, i, r["Sample label"], r["Station"], top, bottom)
        obs["regional_group"] = r["Station"].replace(" control", "").replace(" impacted", "")
        obs["sampling"] = dict(method="multicorer; publisher-standardized Nematoda abundance",
                               reported_reference_area_cm2=10, sampled_area_cm2=None,
                               nominal_core_diameter_cm=9.4,
                               count_subsample_fraction=None, mesh_um=32,
                               method_source="https://doi.org/10.5194/bg-16-3475-2019")
        if "impacted" in r["Station"]:
            obs["flags"].append("experimental_disturbance")
        value = number(r["Nematoda [#]"])
        output.append(_finish(obs, density_per_m2(value, "individuals/10cm2"), value, "individuals/10cm2"))
    return output


def parse_hausgarten(text):
    metadata, rows = _table(text)
    if "Abundance (#/10 cm2)" not in metadata:
        raise ValueError("HAUSGARTEN abundance unit declaration changed")
    grouped, seen = defaultdict(list), set()
    for i, r in enumerate(rows, 1):
        key = (r["Event"], r["Core"], r["Subsample ID"], r["Depth sed top [m]"], r["Depth sed bot [m]"])
        taxon_key = (*key, r["Genus UID"], r["Ident remark"])
        if taxon_key in seen:
            raise ValueError(f"duplicate HAUSGARTEN genus/core/subsample: {taxon_key}")
        seen.add(taxon_key)
        grouped[key].append((i, r))
    output = []
    for key, items in grouped.items():
        i, r = items[0]
        obs = _base("973147", "LTER HAUSGARTEN", r, i, r["Core"], r["Event 2"], number(key[3]), number(key[4]))
        obs["observation_id"] = f"{obs['core_id']}:subsample:{key[2]}:{key[3]}:{key[4]}"
        obs["source_record_ids"] = [f"973147:row:{j}" for j, _ in items]
        obs["station_original"] = r["Event 2"]
        station = "HG_" + r["Event 2"].removeprefix("HG").lstrip("_")
        obs["station"] = obs["regional_group"] = station
        obs["subsample_id"] = key[2]
        area, factor = number(r["Sample area [m**2]"]), number(r["Subsample total factor"])
        obs["sampling"] = dict(method=r["Method/Device"], mesh_um=number(r["Mesh s [µm]"]),
            reported_sample_area_m2=area, reported_area_fraction=factor,
            nominal_syringe_diameter_cm=2.2, nominal_syringe_area_cm2=math.pi * 1.1**2,
            count_subsample_fraction=None, genus_rows=len(items),
            method_source="https://doi.org/10.3389/fmars.2023.1271447")
        if area is not None and factor is not None and not math.isclose(area * factor * 10000, math.pi * 1.1**2, rel_tol=0.02):
            obs["flags"].append("area_metadata_conflict")
        abundances = [number(item["Abund"]) for _, item in items]
        counts = [number(item["Ind No [#]"]) for _, item in items]
        if all(a is not None for a in abundances):
            # Already normalized by the publisher. Do not divide by syringe area again.
            value = sum(density_per_m2(a, "individuals/10cm2") for a in abundances)
            obs["flags"].append("sum_of_reported_genus_abundances")
            output.append(_finish(obs, value, sum(abundances), "individuals/10cm2"))
        else:
            value = None if None in counts else sum(counts)
            output.append(_finish(obs, None, value, "identified_specimens",
                                  ["count_to_population_conversion_not_verified", "area_metadata_conflict"]))
    return output


def parse_canyon(text):
    metadata, rows = _table(text)
    output, seen = [], set()
    for i, r in enumerate(rows, 1):
        top, bottom = number(r["Depth top [m]"]), number(r["Depth bot [m]"])
        key = r["Event"], top, bottom
        if key in seen:
            raise ValueError(f"duplicate canyon core slice: {key}")
        seen.add(key)
        obs = _base("776716", "Gollum Channels / Whittard Slope", r, i, r["Event"], r["Event"].split("_")[0], top, bottom)
        event_date = re.search(re.escape(r["Event"]) + r" \*[^\n]*?DATE/TIME: ([\dT:-]+)", metadata)
        if obs["date"] is None and event_date:
            obs["date"] = event_date.group(1)
        counts = [number(v) for k, v in r.items() if "[#]" in k]
        obs["sampling"] = dict(method="MidiCorer; 100–150 individuals picked per sediment slice for identification",
            reported_core_area_cm2=25.51759, count_subsample_fraction=None,
            method_source="https://doi.org/10.1371/journal.pone.0020094")
        output.append(_finish(obs, None, None if None in counts else sum(counts), "identified_specimens",
                              ["identified_subsample_not_population"]))
    return output


def parse_wei(text):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(text)
    output = []
    for index, place in enumerate(root.findall(".//k:Placemark", ns), 1):
        data = {e.attrib["name"]: e.text for e in place.findall(".//k:SimpleData", ns)}
        if data.get("CATEGORY", "").lower() != "nematode":
            continue
        coords = place.findtext(".//k:coordinates", namespaces=ns)
        lon, lat = coords.strip().split(",")[:2] if coords else (None, None)
        row = {"Latitude": lat, "Longitude": lon, "Date/Time": data.get("DATE.UTC"), "Event": data.get("STATION")}
        obs = _base("wei2010", data.get("REFERENCE"), row, index, f"placemark:{index}", data.get("STATION"), None, None)
        obs["water_depth_m"] = number(data.get("DEPTH.m"))
        obs["eligibility_scope"] = "evidence_only_unknown_sediment_interval"
        obs["sampling"] = dict(method=data.get("GEAR"), secondary_gear=data.get("GEAR2"),
            reported_area_m2=number(data.get("AREA.m2")), mesh_um=number(data.get("SIEVE.SIZE..gm")),
            replication=data.get("REPLICATION"), region=data.get("REGION"),
            comments=data.get("COMMENTS"), count_subsample_fraction=None,
            method_source="https://doi.org/10.1371/journal.pone.0015323.s001")
        value = number(data.get("ABUNDANCE.per.m2"))
        output.append(_finish(obs, density_per_m2(value, "individuals/m2"), value,
                              "individuals/m2", ["unknown_sediment_interval", "heterogeneous_methods"]))
    return output


def aggregate_cores(observations):
    """Produce complete 0–5 cm targets; preserve failed cores with explicit reasons."""
    grouped = defaultdict(list)
    for obs in observations:
        if obs["source_id"] in {"873272", "973147"}:
            grouped[obs["core_id"]].append(obs)
    output = []
    for core, rows in sorted(grouped.items()):
        first = rows[0]
        obs = {**first, "observation_id": core + ":0-5cm", "flags": sorted({f for r in rows for f in r["flags"]})}
        obs.pop("subsample_id", None)
        obs["sampling"] = {**first["sampling"], "component_slices": len(rows)}
        if "genus_rows" in obs["sampling"]:
            obs["sampling"].pop("genus_rows")
            obs["sampling"]["reported_taxon_rows"] = sum(r["sampling"]["genus_rows"] for r in rows)
        reasons = sorted({f for r in rows for f in r["exclusion_reasons"]})
        slices = [(r["sediment_top_cm"], r["sediment_bottom_cm"], r["density_per_m2"]) for r in rows]
        if any(top is None or bottom is None for top, bottom, _ in slices):
            density = None
            reasons.append("unknown_sediment_interval")
        else:
            try:
                density = integrate_slices(slices, 0, 5)
            except ValueError:
                density = None
                reasons.append("overlapping_sediment_intervals")
        if density is None:
            reasons.append("incomplete_or_unknown_0_5cm_density")
        if "experimental_disturbance" in obs["flags"]:
            reasons.append("disturbed_site_excluded_from_reference_experiment")
        for field in ("latitude", "longitude", "water_depth_m"):
            if first[field] is None:
                reasons.append(f"unknown_{field}")
            elif any(r[field] != first[field] for r in rows):
                raise ValueError(f"inconsistent core metadata: {core} {field}")
        obs.update(source_record_ids=[v for r in rows for v in r["source_record_ids"]],
                   component_observation_ids=[r["observation_id"] for r in rows],
                   sediment_top_cm=0, sediment_bottom_cm=5, density_per_m2=density,
                   original_quantity=[r["original_quantity"] for r in rows],
                   unknown_density=density is None, exclusion_reasons=sorted(set(reasons)),
                   eligible_regional_training=not reasons, eligibility_scope="complete_0_5cm_core")
        output.append(obs)
    return output


def density_per_m2(value, unit, *, area_cm2=None, fraction=None):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("quantity must be finite and nonnegative")
    if unit == "individuals/m2":
        return value
    if unit == "individuals/10cm2":
        return value * 1000
    if unit == "identified_specimens":
        return None  # taxonomic subsampling is not population accounting
    if unit != "count":
        raise ValueError(f"unsupported unit: {unit}")
    if area_cm2 is None or fraction is None:
        return None
    if not math.isfinite(area_cm2) or area_cm2 <= 0 or not 0 < fraction <= 1:
        raise ValueError("invalid area or subsample fraction")
    return value * 10000 / area_cm2 / fraction


def sum_identifications(rows):
    totals = defaultdict(list)
    seen = set()
    for row in rows:
        group = row["core"], row["top_cm"], row["bottom_cm"]
        key = (*group, row["taxon"])
        if key in seen:
            raise ValueError(f"duplicate taxon/core/slice: {key}")
        seen.add(key)
        totals[group].append(row["count"])
    return [dict(core=key[0], top_cm=key[1], bottom_cm=key[2],
                 identified_count=None if None in counts else sum(counts),
                 density_per_m2=None, population_accounting=False)
            for key, counts in sorted(totals.items())]


def integrate_slices(slices, top_cm, bottom_cm):
    """Sum areal densities over an exactly covered interval; never fill gaps."""
    cursor, total, complete = top_cm, 0.0, True
    for top, bottom, density in sorted(slices, key=lambda s: (s[0], s[1])):
        if bottom <= top or top < cursor - 1e-8 or bottom > bottom_cm + 1e-8:
            raise ValueError("overlap, duplicate, or invalid slice interval")
        complete &= math.isclose(top, cursor, abs_tol=1e-8)
        if density is None:
            complete = False
        else:
            total += density_per_m2(density, "individuals/m2")
        cursor = bottom
    return total if complete and math.isclose(cursor, bottom_cm, abs_tol=1e-8) else None
