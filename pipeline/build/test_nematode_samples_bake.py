"""Tests for the compact point-cloud artifact baked from the nematode
sample-location dataset (`fetch/nematode_samples.py`) into
`app/public/data/nematode_samples_v2/`. Pins the binary layout (so a decoder
written once in the app keeps working) and checks the artifact actually
stays small -- tens of KB, not a re-shipped copy of the 1.25 MB source
CSV.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "fetch"))
from bake_nematode_samples import (
    RECORD_FORMAT,
    RECORD_SIZE,
    build_meta,
    pack_samples,
)
from nematode_samples import (
    BIOME_NAMES,
    Sample,
    biome_summary,
    ensure_all_downloaded,
    load_full_dataset,
)


def _sample(sample_id, lat, lon, total, biome_raw, **kw):
    defaults = dict(
        bacterivores=None,
        fungivores=None,
        herbivores=None,
        omnivores=None,
        predators=None,
        unidentified=None,
        pixel_lat=None,
        pixel_long=None,
        sampling_method="x",
        sampling_ref="x",
        sampling_depth="",
        doi_url="",
        data_provider="x",
    )
    defaults.update(kw)
    return Sample(
        sample_id=sample_id,
        total_number=total,
        latitude=lat,
        longitude=lon,
        wwf_biome_raw=biome_raw,
        **defaults,
    )


def test_record_size_is_13_bytes_packed_no_padding():
    assert struct.calcsize(RECORD_FORMAT) == RECORD_SIZE
    assert RECORD_SIZE == 13


def test_pack_round_trips_lat_lon_total_biome():
    samples = [_sample(1, 53.09239939, 6.626211158, 3124.0, 4.0)]
    blob = pack_samples(samples)
    assert len(blob) == RECORD_SIZE
    lat, lon, total, biome = struct.unpack(RECORD_FORMAT, blob)
    assert lat == pytest.approx(53.09239939, abs=1e-3)
    assert lon == pytest.approx(6.626211158, abs=1e-3)
    assert total == pytest.approx(3124.0, abs=1e-1)
    assert biome == 4


def test_pack_applies_antarctic_override():
    samples = [_sample(2, -70.0, 100.0, 500.0, 11.0)]
    blob = pack_samples(samples)
    _lat, _lon, _total, biome = struct.unpack(RECORD_FORMAT, blob)
    assert biome == 25


def test_pack_preserves_non_terrestrial_codes_unfiltered():
    # The compact artifact ships ALL 6,825 points (it's a "where did
    # someone actually put a corer" map) -- unlike biome_summary(), which
    # drops codes 0/98, packing must NOT drop them.
    samples = [
        _sample(3, 10.0, 10.0, 1.0, 0.0),
        _sample(4, 20.0, 20.0, 1.0, 98.0),
    ]
    blob = pack_samples(samples)
    assert len(blob) == 2 * RECORD_SIZE
    _, _, _, biome0 = struct.unpack(RECORD_FORMAT, blob[:RECORD_SIZE])
    _, _, _, biome1 = struct.unpack(RECORD_FORMAT, blob[RECORD_SIZE:])
    assert biome0 == 0
    assert biome1 == 98


def test_artifact_is_tens_of_kb_not_megabytes():
    samples = [_sample(i, 0.0, 0.0, float(i), 1.0) for i in range(6825)]
    blob = pack_samples(samples)
    assert len(blob) == 6825 * RECORD_SIZE
    source_csv_bytes = 1_256_599
    assert len(blob) < source_csv_bytes / 10
    assert len(blob) < 150_000  # "tens of KB" per the task brief


def test_total_number_above_uint16_range_is_not_truncated():
    # Real data has samples above 65,535 per 100g (max observed: 169,206) --
    # float32 total, not uint16, so these must round-trip exactly, not clip.
    samples = [_sample(5, 0.0, 0.0, 169206.0, 4.0)]
    blob = pack_samples(samples)
    _, _, total, _ = struct.unpack(RECORD_FORMAT, blob)
    assert total == pytest.approx(169206.0, rel=1e-5)


def test_meta_documents_byte_layout_biome_legend_and_provenance():
    samples = [_sample(1, 1.0, 2.0, 3.0, 4.0)]
    meta = build_meta(samples, biome_summary_rows=[])
    assert meta["n_samples"] == 1
    assert meta["record_bytes"] == RECORD_SIZE
    assert meta["total_bytes"] == RECORD_SIZE
    assert meta["license"] == "CC0"
    assert meta["field_order"] == [
        "latitude_deg",
        "longitude_deg",
        "total_per_100g_dry_soil",
        "biome_index",
    ]
    # biome legend covers every code BIOME_NAMES defines, keyed as strings
    # (JSON object keys are always strings)
    for code, name in BIOME_NAMES.items():
        assert meta["biome_legend"][str(code)] == name
    assert "figshare" in meta["source"]["collection_doi"] or meta["source"][
        "collection_doi"
    ].startswith("10.6084")


def test_public_sample_overlay_matches_baker_and_source_item():
    asset_dir = Path(__file__).resolve().parents[2] / "app/public/data/nematode_samples_v2"
    samples = load_full_dataset(ensure_all_downloaded()["full_dataset"])
    data = (asset_dir / "samples.bin").read_bytes()
    metadata = json.loads((asset_dir / "samples_meta.json").read_text())
    assert metadata == build_meta(samples, biome_summary(samples))
    assert data == pack_samples(samples)
    assert hashlib.sha256(data).hexdigest() == "e92e05f56992ddaea64810549815195d6359e79ac9d4a14d3f8997611d127a0a"
    assert metadata["source"]["full_dataset_doi"] == "10.6084/m9.figshare.11733897"
    assert metadata["source"]["license_url"] == "https://creativecommons.org/publicdomain/zero/1.0/"
