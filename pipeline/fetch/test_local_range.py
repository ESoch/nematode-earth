"""Local downloads must preserve the ranged-reader and source-identity gates."""
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_conservative import fixture
from tiff_header import TiffHeaderReader
from conservative import extract_conservative


def reader(path, **kwargs):
    assert importlib.util.find_spec("local_range") is not None, "local download reader is missing"
    return importlib.import_module("local_range").LocalRangeClient(path, **kwargs)


def test_local_ranges_are_inclusive_bounded_and_content_identified(tmp_path):
    path = tmp_path / "source.tif"
    path.write_bytes(b"abcdefgh")
    client = reader(path, max_response_bytes=4)
    assert client.fetch(2, 5) == b"cdef"
    assert client.source_identity == (8, "sha256:" + hashlib.sha256(b"abcdefgh").hexdigest(), None)
    assert client.bytes_fetched == 4 and client.reads_made == 1
    assert client.requests_made == 0 and client.retries_used == 0
    assert client.fetch(7, 7) == b"h"


@pytest.mark.parametrize("start,end", [(-1, 0), (3, 2), (0, 4), (7, 8)])
def test_local_invalid_oversized_and_past_eof_ranges_fail(tmp_path, start, end):
    path = tmp_path / "source.tif"
    path.write_bytes(b"abcdefgh")
    client = reader(path, max_response_bytes=4)
    with pytest.raises(ValueError):
        client.fetch(start, end)


@pytest.mark.parametrize("operation", ["rewrite", "truncate", "replace"])
def test_modified_source_is_rejected_even_when_size_is_unchanged(tmp_path, operation):
    path = tmp_path / "source.tif"
    path.write_bytes(b"abcdefgh")
    client = reader(path)
    old = path.stat()
    if operation == "replace":
        replacement = tmp_path / "replacement.tif"
        replacement.write_bytes(b"ABCDEFGH")
        replacement.replace(path)
    else:
        path.write_bytes(b"ABCDEFGH" if operation == "rewrite" else b"a")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
    with pytest.raises(RuntimeError, match="changed"):
        client.fetch(0, 0)
    with pytest.raises(RuntimeError, match="changed"):
        client.verify_unchanged()


def test_local_resume_reuses_only_matching_content(tmp_path):
    path = tmp_path / "source.tif"
    path.write_bytes(fixture([[100, np.nan], [100, np.nan]])[0])

    def run():
        client = reader(path)
        info = TiffHeaderReader("https://example.invalid/soil", client.fetch).parse()
        return extract_conservative(info.url, info, [0, 2], [60, 0],
                                    cache_dir=tmp_path / "cache", source_version="v1",
                                    client=client, block_rows=1)

    first = run()
    assert first["density"][0, 0] == pytest.approx(50)
    assert run()["cache_hits"] == 2
    path.write_bytes(fixture([[200, np.nan], [200, np.nan]])[0])
    changed = run()
    assert changed["cache_hits"] == 0
    assert changed["density"][0, 0] == pytest.approx(100)
    assert changed["cache_identity"] != first["cache_identity"]


def cli_command(tmp_path, local_path):
    return [sys.executable, str(Path(__file__).with_name("eth_raster.py")), "conservative",
            "https://example.invalid/soil", "--local-file", str(local_path),
            "--source-version", "v1", "--citation", "https://example.invalid/record",
            "--out-width", "1", "--out-height", "1", "--west", "0", "--east", "2",
            "--north", "60", "--south", "0", "--block-rows", "1",
            "--cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path / "out"),
            "--tag", "local", "--max-retries", "1", "--timeout", "1"]


def test_local_cli_uses_all_rows_no_network_and_preserves_archive_provenance(tmp_path):
    path = tmp_path / "download.tif"
    raw, _ = fixture([[100, np.nan], [100, np.nan]])
    path.write_bytes(raw)
    command = cli_command(tmp_path, path)
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    meta = json.loads((tmp_path / "out/local_meta.json").read_text())
    assert meta["complete"] and meta["publishable"] and meta["rows_touched"] == 2
    assert meta["row_coverage_actual"] == 1
    assert meta["requests_made"] == 0 and meta["source_http_identity"] is None
    assert meta["provenance"]["source_url"] == "https://example.invalid/soil"
    assert meta["provenance"]["transport"] == "local-file"
    assert meta["provenance"]["local_file_sha256"] == hashlib.sha256(raw).hexdigest()
    with np.load(tmp_path / "out/local_grid.npz", allow_pickle=False) as grid:
        np.testing.assert_allclose(grid["density"], [[50]])
        np.testing.assert_allclose(grid["valid_model_fraction"], [[0.5]])
    again = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert again.returncode != 0
    assert path.read_bytes() == raw


def test_truncated_local_tiff_never_publishes_a_product(tmp_path):
    path = tmp_path / "download.tif"
    path.write_bytes(fixture([[100, 100], [100, 100]])[0][:-8])
    result = subprocess.run(cli_command(tmp_path, path), capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "outside local file" in result.stderr
    assert not (tmp_path / "out/local_meta.json").exists()
    assert not (tmp_path / "out/local_grid.npz").exists()
