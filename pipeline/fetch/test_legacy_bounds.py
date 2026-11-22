"""Legacy reproduction must split large bands and never publish failed rows."""
import sys

import numpy as np
import pytest

import eth_raster
from test_conservative import fixture


def test_large_legacy_band_is_split_without_changing_samples():
    # Tiny request ceiling exercises the same >32MiB production branch without
    # allocating a >32MiB fixture under the user's memory pressure.
    data = np.array([[1, np.nan], [3, 4], [5, 6]], dtype=float)
    raw, info = fixture(data)
    class Client:
        max_response_bytes = 20
        calls = []

        def fetch(self, start, end):
            if end - start + 1 > self.max_response_bytes:
                raise ValueError("Range exceeds bounded response size")
            self.calls.append((start, end))
            return raw[start:end + 1]

    client = Client()
    got = eth_raster._fetch_row_block(client, info, 0, 3)
    np.testing.assert_equal(got, data)
    assert len(client.calls) == 3


def test_legacy_cli_refuses_unresolved_rows_before_writing(tmp_path, monkeypatch):
    _, info = fixture([[100]])
    # Keep CLI publication real; isolate only source I/O and its exhausted
    # failure result, already covered by the range/extraction tests.
    monkeypatch.setattr(eth_raster, "verify_header", lambda url: info)
    monkeypatch.setattr(eth_raster, "extract_grid", lambda *args, **kwargs: {
        "density": np.array([[np.nan]]), "land_fraction": np.array([[np.nan]]),
        "failed_rows": [0], "rows_touched": 0, "row_coverage_actual": 0,
        "bytes_fetched": 0, "requests_made": 3, "retries_used": 3, "elapsed_s": 1})
    monkeypatch.setattr(sys, "argv", ["eth_raster.py", "extract", "fixture://soil",
                                       "--out-dir", str(tmp_path), "--out-width", "1", "--out-height", "1"])
    with pytest.raises(RuntimeError, match="incomplete"):
        eth_raster.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("layout", ["padding", "byte_count", "compressed", "multi_sample", "multi_row"])
def test_legacy_rejects_unsupported_strip_layout_before_fetch(layout):
    raw, info = fixture([[1, 2], [3, 4], [5, 6]])
    if layout == "padding":
        # First two strip offsets look contiguous; only a later strip has a gap.
        info.strip_offsets[2] += 8
    elif layout == "byte_count":
        info.strip_byte_counts[2] += 8
    elif layout == "compressed":
        info.compression = 5
    elif layout == "multi_sample":
        info.samples_per_pixel = 2
    else:
        info.rows_per_strip = 2

    class Client:
        max_response_bytes = 20
        calls = 0

        def fetch(self, start, end):
            self.calls += 1
            return raw[start:end + 1]

    client = Client()
    with pytest.raises(ValueError, match="layout"):
        eth_raster._fetch_row_block(client, info, 0, 3)
    assert client.calls == 0
