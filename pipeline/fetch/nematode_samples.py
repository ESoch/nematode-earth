#!/usr/bin/env python3
"""Fetch the van den Hoogen et al. 2020 global soil nematode sample-location
dataset. The 2020 release provides 6,825 measured sample records. The
2019 study compiled 6,759 samples, aggregated them into 1,915 pixels,
and excluded 39 Antarctic pixels before modelling the remaining 1,876.
The biome codes in the later release
identify geographic groups and do not establish membership in the earlier
model training data.

van den Hoogen, J. et al. *A global database of soil nematode abundance and
functional group composition.* Scientific Data 7:103 (2020).
doi:10.1038/s41597-020-0437-3

The Figshare full-dataset item (doi:10.6084/m9.figshare.11733897.v1)
carries a CC0 1.0 dedication. The three items below belong to collection
doi:10.6084/m9.figshare.c.4718003. Sizes and MD5s are from the item records.

- `nematode_full_dataset_wBiome.csv` -- 1,256,599 bytes, MD5
  9414fbdebce2340ae649851106c6c5a1 -- doi:10.6084/m9.figshare.11733897
  (download id 21358326). The raw 6,825-sample table this module parses.
- `nematode_aggregated_wCovariateData.csv` -- 1,407,568 bytes, MD5
  d920b3e00ab2cc61cde67b7a370074f8 -- doi:10.6084/m9.figshare.11733891
  (download id 21902778). 1,933 unique 1km pixels x 73 covariates -- the
  aggregated with covariates. The overlay uses the separate sample-level CSV.
- `metadata.csv` -- 13,306 bytes, MD5 8df6e8076f629ef1a9887df8dab5da76 --
  doi:10.6084/m9.figshare.11733894 (download id 21902781). Column
  definitions for both CSVs above; no biome-code lookup table in it (see
  below).

Mirrored (same content, plus the authors' own analysis script) at
github.com/hooge104/2020_global_nematode_dataset.

Per sample, `nematode_full_dataset_wBiome.csv` carries the five
functional-group counts plus `Unidentified` and `Total_Number` (all
individuals per 100 g dry soil), `Latitude`/`Longitude`, `Pixel_Lat`/
`Pixel_Long` (the covariate-raster pixel centroid, absent for ~573
Antarctic-adjacent rows per `metadata.csv`), `WWF_Biome`, sampling method/
reference/depth, `DOI/URL`, `Data_provider`.

WWF_Biome cleaning -- the load-bearing discovery this session: the raw
column is a *float* with two data-quality quirks, neither documented in
`metadata.csv`, both confirmed by reproducing the authors' own knitted
analysis (`20191029_SciData.Rmd` in the GitHub mirror) exactly:

1. Codes are near-integer floating-point noise (e.g. `10.99999999`,
   `11.00000003`), not real fractional/area-weighted values -- an
   exhaustive scan puts the max deviation from the nearest integer at
   3e-8 across all 6,825 rows. The authors' fix is `round(WWF_Biome, 0)`;
   `clean_biome_index` below does the same.
2. Samples south of -60 deg latitude get their biome code forcibly
   overridden to 25 ("Antarctica") regardless of what the source raster
   extraction produced -- near-pole categorical-raster extraction is
   unreliable, so the authors just discard it:
   `full_data_wBiome[full_data_wBiome$Latitude < -60,]$WWF_Biome <- 25`.
   Skipping this override was the difference between reproducing the
   published Tundra row (median 2,695, n=148) and getting n=651 instead --
   503 of those extra "Tundra" rows are actually Antarctic samples whose
   raster extraction happened to land near WWF biome 11's code.

Biome codes 1-14 are the standard WWF/Olson terrestrial biome numbering
(the same `BIOME_NUM` scheme GEE's `RESOLVE/ECOREGIONS/2017` and the
classic `wwf_terr_ecos` shapefile use); 0 and 98 are non-terrestrial/
no-match codes the authors' own summary explicitly filters out (0: no
biome polygon at that point -- coastal/edge artifact; 98: Lake); 25 is
the authors' own Antarctica override above, not a WWF code.

Different extraction methods recover different fractions of the nematodes
present. Incomplete recovery can bias observations low. Quantifying errors
in predictions from the 2019 model requires evidence on that model.

Usage:
    uv run fetch/nematode_samples.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

COLLECTION_DOI = "10.6084/m9.figshare.c.4718003"
LICENSE = "CC0 1.0 Universal (Public Domain Dedication)"

FULL_DATASET_URL = "https://ndownloader.figshare.com/files/21358326"
FULL_DATASET_DOI = "10.6084/m9.figshare.11733897"
FULL_DATASET_EXPECTED_SIZE = 1_256_599
FULL_DATASET_EXPECTED_MD5 = "9414fbdebce2340ae649851106c6c5a1"

AGGREGATED_URL = "https://ndownloader.figshare.com/files/21902778"
AGGREGATED_DOI = "10.6084/m9.figshare.11733891"
AGGREGATED_EXPECTED_SIZE = 1_407_568
AGGREGATED_EXPECTED_MD5 = "d920b3e00ab2cc61cde67b7a370074f8"

METADATA_URL = "https://ndownloader.figshare.com/files/21902781"
METADATA_DOI = "10.6084/m9.figshare.11733894"
METADATA_EXPECTED_SIZE = 13_306
METADATA_EXPECTED_MD5 = "8df6e8076f629ef1a9887df8dab5da76"

CACHE_DIR = Path(__file__).parent.parent / "cache" / "nematode_samples"

# Standard WWF/Olson terrestrial biome numbering (BIOME_NUM), matching the
# authors' own knitted analysis's naming exactly (20191029_SciData.Rmd).
# 0 and 98 are the two non-terrestrial/no-match codes present in the raw
# data; 25 is the authors' Antarctica override (see module docstring), not
# a WWF code.
BIOME_NAMES: dict[int, str] = {
    0: "No biome match",
    1: "Tropical Moist Forests",
    2: "Tropical Dry Forests",
    3: "Tropical Coniferous Forests",
    4: "Temperate Broadleaf Forests",
    5: "Temperate Conifer Forests",
    6: "Boreal Forests",
    7: "Tropical Grasslands",
    8: "Temperate Grasslands",
    9: "Flooded Grasslands",
    10: "Montane Grasslands",
    11: "Tundra",
    12: "Mediterranean Forests",
    13: "Deserts",
    25: "Antarctica",
    98: "Lake",
}

# Biome codes the authors' own summary table excludes (not real terrestrial
# biomes; see BIOME_NAMES).
_SUMMARY_EXCLUDED_CODES = {0, 98}


@dataclass
class Sample:
    sample_id: int
    bacterivores: float | None
    fungivores: float | None
    herbivores: float | None
    omnivores: float | None
    predators: float | None
    unidentified: float | None
    total_number: float
    latitude: float
    longitude: float
    pixel_lat: float | None
    pixel_long: float | None
    wwf_biome_raw: float
    sampling_method: str
    sampling_ref: str
    sampling_depth: str
    doi_url: str
    data_provider: str


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_downloaded(
    cache_path: Path, url: str, expected_size: int, expected_md5: str
) -> Path:
    """Download `url` to `cache_path` if not already present, then verify
    both byte size and MD5 against the figshare API's own reported values
    (queried this session, recorded as constants above) -- catches a
    truncated download or a silently-changed upstream file either one."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {cache_path} ...", file=sys.stderr)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        tmp = cache_path.with_suffix(cache_path.suffix + ".partial")
        tmp.write_bytes(resp.content)
        tmp.rename(cache_path)

    actual_size = cache_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{cache_path}: expected {expected_size} bytes, got {actual_size} "
            f"-- upstream file may have changed"
        )
    actual_md5 = _md5(cache_path)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"{cache_path}: expected MD5 {expected_md5}, got {actual_md5} "
            f"-- upstream file may have changed"
        )
    return cache_path


def ensure_all_downloaded(cache_dir: Path = CACHE_DIR) -> dict[str, Path]:
    return {
        "full_dataset": ensure_downloaded(
            cache_dir / "nematode_full_dataset_wBiome.csv",
            FULL_DATASET_URL,
            FULL_DATASET_EXPECTED_SIZE,
            FULL_DATASET_EXPECTED_MD5,
        ),
        "aggregated": ensure_downloaded(
            cache_dir / "nematode_aggregated_wCovariateData.csv",
            AGGREGATED_URL,
            AGGREGATED_EXPECTED_SIZE,
            AGGREGATED_EXPECTED_MD5,
        ),
        "metadata": ensure_downloaded(
            cache_dir / "metadata.csv",
            METADATA_URL,
            METADATA_EXPECTED_SIZE,
            METADATA_EXPECTED_MD5,
        ),
    }


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def _float_or_none(s: str) -> float | None:
    s = s.strip()
    return float(s) if s else None


def load_full_dataset(path: Path) -> list[Sample]:
    """Parse `nematode_full_dataset_wBiome.csv` (6,825 rows). Encoding is
    utf-8-sig: the file ships a BOM on its header's first column."""
    samples = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                Sample(
                    sample_id=int(row["Sample_ID"]),
                    bacterivores=_float_or_none(row["Bacterivores"]),
                    fungivores=_float_or_none(row["Fungivores"]),
                    herbivores=_float_or_none(row["Herbivores"]),
                    omnivores=_float_or_none(row["Omnivores"]),
                    predators=_float_or_none(row["Predators"]),
                    unidentified=_float_or_none(row["Unidentified"]),
                    total_number=float(row["Total_Number"]),
                    latitude=float(row["Latitude"]),
                    longitude=float(row["Longitude"]),
                    pixel_lat=_float_or_none(row["Pixel_Lat"]),
                    pixel_long=_float_or_none(row["Pixel_Long"]),
                    wwf_biome_raw=float(row["WWF_Biome"]),
                    sampling_method=row["sampling method"],
                    sampling_ref=row["sampling_ref"],
                    sampling_depth=row["sampling depth"],
                    doi_url=row["DOI/URL"],
                    data_provider=row["Data_provider"],
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Biome cleaning + aggregation
# ---------------------------------------------------------------------------


def clean_biome_index(wwf_biome_raw: float, latitude: float) -> int:
    """Round-to-nearest-int (clears floating-point noise around integer
    codes), then apply the authors' own Antarctica override: any sample
    south of -60 deg gets forced to code 25, overriding whatever the
    source raster extraction produced. See module docstring."""
    if latitude < -60:
        return 25
    return round(wwf_biome_raw)


def biome_summary(samples: list[Sample]) -> list[dict]:
    """Mean, median and n of Total_Number grouped by cleaned biome index,
    excluding the two non-terrestrial codes (0: no biome match, 98: Lake)
    -- replicates the authors' own knitted-analysis table
    (`20191029_SciData.Rmd`) exactly, sorted by descending median to match
    its presentation order."""
    by_biome: dict[int, list[float]] = {}
    for s in samples:
        code = clean_biome_index(s.wwf_biome_raw, s.latitude)
        if code in _SUMMARY_EXCLUDED_CODES:
            continue
        by_biome.setdefault(code, []).append(s.total_number)

    rows = []
    for code, totals in by_biome.items():
        rows.append(
            {
                "biome": BIOME_NAMES.get(code, f"Unknown ({code})"),
                "biome_code": code,
                "mean": statistics.mean(totals),
                "median": statistics.median(totals),
                "n": len(totals),
            }
        )
    rows.sort(key=lambda r: r["median"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    paths = ensure_all_downloaded()
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)", file=sys.stderr)

    samples = load_full_dataset(paths["full_dataset"])
    print(f"\nparsed {len(samples)} samples", file=sys.stderr)

    totals = [s.total_number for s in samples]
    summary = biome_summary(samples)

    result = {
        "n_samples": len(samples),
        "total_number_mean": statistics.mean(totals),
        "total_number_median": statistics.median(totals),
        "biome_summary": summary,
        "license": LICENSE,
        "collection_doi": COLLECTION_DOI,
        "source_dois": {
            "full_dataset": FULL_DATASET_DOI,
            "aggregated": AGGREGATED_DOI,
            "metadata": METADATA_DOI,
        },
    }

    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nematode_biome_summary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== biome summary (Total_Number: mean / median / n) ===")
    for row in summary:
        print(
            f"  {row['biome']:30} mean={row['mean']:9.1f} "
            f"median={row['median']:9.1f} n={row['n']:5}"
        )
    print(
        f"\n  {'TOTAL':30} mean={result['total_number_mean']:9.1f} "
        f"median={result['total_number_median']:9.1f} n={len(samples):5}"
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
