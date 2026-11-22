#!/usr/bin/env python3
"""Bake the measured soil sample overlay from the 2020 Figshare dataset.

The public asset at app/public/data/nematode_samples_v2/ contains one packed
13-byte record per source row in CSV order. Each little-endian record (<fffB)
stores latitude, longitude, reported abundance per 100 g dry soil, and the
cleaned biome code. The sidecar describes the layout and credits the source.

All 6,825 records are included, including codes 0 (no biome match) and 98
(lake). The biome summary follows the source analysis and omits those two
codes. The Antarctic biome override and the other biome codes describe sample
locations. The source table does not identify individual training records for
the separate 2019 abundance model.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fetch"))
from nematode_samples import (
    BIOME_NAMES,
    COLLECTION_DOI,
    FULL_DATASET_DOI,
    FULL_DATASET_EXPECTED_MD5,
    LICENSE,
    Sample,
    biome_summary,
    clean_biome_index,
    ensure_all_downloaded,
    load_full_dataset,
)

# little-endian, unpadded: float32 lat, float32 lon, float32 total-per-100g,
# uint8 biome index. struct.calcsize confirms 13 bytes, no alignment padding.
RECORD_FORMAT = "<fffB"
RECORD_SIZE = struct.calcsize(RECORD_FORMAT)
assert RECORD_SIZE == 13


def pack_samples(samples: list[Sample]) -> bytes:
    """Pack every sample, in input order, into the 13-byte record layout.
    Does NOT filter non-terrestrial biome codes (0/98) -- see module
    docstring; that filtering is `biome_summary()`'s job, for statistics,
    not this function's."""
    out = bytearray(RECORD_SIZE * len(samples))
    for i, s in enumerate(samples):
        biome = clean_biome_index(s.wwf_biome_raw, s.latitude)
        struct.pack_into(
            RECORD_FORMAT,
            out,
            i * RECORD_SIZE,
            s.latitude,
            s.longitude,
            s.total_number,
            biome,
        )
    return bytes(out)


def build_meta(samples: list[Sample], biome_summary_rows: list[dict]) -> dict:
    n = len(samples)
    return {
        "description": "Soil nematode sample records from the 2020 dataset. "
        "Each point marks the reported location of one sample.",
        "n_samples": n,
        "format": "packed binary, little-endian, no padding, one "
        f"{RECORD_SIZE}-byte record per sample in source-CSV row order",
        "record_bytes": RECORD_SIZE,
        "total_bytes": RECORD_SIZE * n,
        "field_order": [
            "latitude_deg",
            "longitude_deg",
            "total_per_100g_dry_soil",
            "biome_index",
        ],
        "field_layout": {
            "latitude_deg": {
                "offset": 0,
                "type": "float32",
                "units": "decimal degrees, WGS84",
            },
            "longitude_deg": {
                "offset": 4,
                "type": "float32",
                "units": "decimal degrees, WGS84",
            },
            "total_per_100g_dry_soil": {
                "offset": 8,
                "type": "float32",
                "units": "nematode individuals per 100 g dry soil",
                "source_column": "Total_Number",
            },
            "biome_index": {
                "offset": 12,
                "type": "uint8",
                "note": "round(WWF_Biome) with the authors' Antarctica override "
                "applied (latitude < -60 deg forces code 25); UNFILTERED -- "
                "includes the two non-terrestrial codes (0, 98) that the "
                "biome_summary table below excludes. See biome_legend.",
            },
        },
        "decode_note": "13 is not a multiple of 4, so records are NOT "
        "Float32Array-alignable across the whole buffer -- decode per-record "
        "with DataView (getFloat32/getUint8 at record_index*13 + field offset), "
        "not a strided typed-array view.",
        "biome_legend": {str(code): name for code, name in sorted(BIOME_NAMES.items())},
        "biome_summary": biome_summary_rows,
        "license": "CC0",
        "source": {
            "dataset_citation": "Geisen, S., van den Hoogen, J., Data Providers, "
            "and Crowther, T. (2020). Nematode_abundance_dataset, version 1. Figshare.",
            "citation": "van den Hoogen, J. et al. A global database of soil "
            "nematode abundance and functional group composition. Scientific "
            "Data 7:103 (2020). doi:10.1038/s41597-020-0437-3",
            "collection_doi": COLLECTION_DOI,
            "full_dataset_doi": FULL_DATASET_DOI,
            "full_dataset_version_doi": FULL_DATASET_DOI + ".v1",
            "source_csv_md5": FULL_DATASET_EXPECTED_MD5,
            "license_full": LICENSE,
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "provenance_note": "The full dataset item containing the source CSV carries "
            "the CC0 dedication. The collection DOI identifies its enclosing collection.",
        },
        "caveats": [
            "Point coordinates locate samples. Estimating the fraction of land sampled "
            "would require information on the area each sample covered.",
            "Extraction methods differ in how many nematodes they recover. "
            "This affects measured abundance. Model predictions also have estimation error.",
            "Sampling effort varies across biomes. The biome summary reports the "
            "number of sample records per included biome.",
            "The 2019 study compiled 6,759 samples, aggregated them by pixel, and used "
            "1,876 pixels for geospatial modelling after excluding 39 Antarctic pixels. "
            "This separate 2020 release contains "
            "6,825 records. A biome code describes location and cannot establish whether "
            "an individual record contributed to the earlier model.",
        ],
    }


def main():
    paths = ensure_all_downloaded()
    samples = load_full_dataset(paths["full_dataset"])
    print(f"loaded {len(samples)} samples", file=sys.stderr)

    blob = pack_samples(samples)
    summary_rows = biome_summary(samples)
    meta = build_meta(samples, summary_rows)

    out_dir = (
        Path(__file__).parent.parent.parent
        / "app"
        / "public"
        / "data"
        / "nematode_samples_v2"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples.bin").write_bytes(blob)
    with open(out_dir / "samples_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {out_dir / 'samples.bin'} ({len(blob)} bytes)")
    print(f"wrote {out_dir / 'samples_meta.json'}")


if __name__ == "__main__":
    main()
