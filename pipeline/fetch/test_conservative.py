"""Behavioral contracts for geographic reduction and resumable small checkpoints."""
import importlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tiff_header import TiffHeaderReader


def implementation():
    assert importlib.util.find_spec("conservative") is not None, "conservative extraction is missing"
    return importlib.import_module("conservative")


def fixture(data, *, point=False, lon=0, lat=60, dx=1, dy=30):
    """Real TIFF bytes, explicit EPSG:4326 and RasterPixelIsArea/Point keys."""
    buffer = io.BytesIO()
    keys = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 2 if point else 1, 2048, 0, 1, 4326)
    tifffile.imwrite(buffer, np.asarray(data, dtype="<f8"), rowsperstrip=1,
                     metadata=None, extratags=[
                         (33550, "d", 3, (dx, dy, 0), False),
                         (33922, "d", 6, (0, 0, 0, lon, lat, 0), False),
                         (34735, "H", len(keys), keys, False)])
    raw = buffer.getvalue()
    info = TiffHeaderReader("fixture://soil", lambda a, b: raw[a:b + 1],
                           source_identity=(len(raw), "fixture-v1", None)).parse()
    return raw, info


class BytesClient:
    def __init__(self, raw, fail_after=None):
        self.raw, self.fail_after = raw, fail_after
        self.bytes_fetched = self.requests_made = self.retries_used = 0
        # This in-memory fixture models a client already verified while parsing
        # its source. Fresh HTTP clients are exercised separately below.
        self.source_identity = (len(raw), "fixture-v1", None)

    def fetch(self, start, end):
        if self.fail_after is not None and self.requests_made >= self.fail_after:
            raise RuntimeError("interrupted download")
        self.requests_made += 1
        result = self.raw[start:end + 1]
        self.bytes_fetched += len(result)
        return result


def run(tmp_path, data, lon_edges, lat_edges, **kwargs):
    mod = implementation()
    raw, info = fixture(data, **kwargs.pop("geo", {}))
    client = kwargs.pop("client", BytesClient(raw))
    return mod.extract_conservative(info.url, info, np.array(lon_edges), np.array(lat_edges),
                                    cache_dir=tmp_path, source_version="fixture-v1",
                                    client=client, block_rows=1, **kwargs)


def test_constant_and_half_valid_cell(tmp_path):
    result = run(tmp_path, [[100, np.nan], [100, np.nan]], [0, 2], [60, 0])
    np.testing.assert_allclose(result["density"], [[50]])
    np.testing.assert_allclose(result["valid_model_fraction"], [[0.5]])
    assert result["rows_touched"] == 2 and result["complete"]
    assert result["conservation_relative_error"] < 1e-12


def test_latitude_gradient_uses_sine_area_not_equal_pixel_mean(tmp_path):
    result = run(tmp_path, [[100], [200]], [0, 1], [60, 0])
    # Independent analytic integral: 100*(sin60-sin30) + 200*(sin30-sin0).
    expected = (100 * (np.sqrt(3) / 2 - 0.5) + 200 * 0.5) / (np.sqrt(3) / 2)
    assert result["density"][0, 0] == pytest.approx(expected)


def test_fractional_longitude_overlap(tmp_path):
    result = run(tmp_path, [[0, 100]], [0, 0.5, 1.5, 2], [60, 30])
    np.testing.assert_allclose(result["density"], [[0, 50, 100]])
    np.testing.assert_allclose(result["valid_model_fraction"], 1)


def test_missing_is_separate_from_measured_zero(tmp_path):
    result = run(tmp_path, [[0, np.nan]], [0, 1, 2], [60, 30])
    np.testing.assert_equal(result["density"], [[0, 0]])
    np.testing.assert_equal(result["valid_model_fraction"], [[1, 0]])


def test_raster_point_edges_and_unequal_native_grids(tmp_path):
    mod = implementation()
    _, point = fixture([[100, 100, 100], [100, 100, 100]], point=True, lon=0, lat=60)
    assert getattr(point, "raster_type", None) == 2, "GeoTIFF point tag must be parsed"
    x, y = mod.source_edges(point)
    np.testing.assert_equal(x, [-0.5, 0.5, 1.5, 2.5])
    np.testing.assert_equal(y, [75, 45, 15])
    area = run(tmp_path / "a", [[100, 100], [100, 100]], [0, 2], [60, 30])
    point_result = run(tmp_path / "b", [[100, 100, 100], [100, 100, 100]],
                       [0, 2], [60, 30], geo={"point": True})
    np.testing.assert_allclose(area["density"], point_result["density"])
    np.testing.assert_allclose(point_result["valid_model_fraction"], 1)


def test_resume_corrupt_incomplete_and_config_identity(tmp_path):
    mod = implementation()
    raw, info = fixture([[100], [200]])
    args = (info.url, info, np.array([0, 1]), np.array([60, 0]))
    kw = dict(cache_dir=tmp_path, source_version="v1", block_rows=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        mod.extract_conservative(*args, client=BytesClient(raw, fail_after=1), **kw)
    client = BytesClient(raw)
    result = mod.extract_conservative(*args, client=client, **kw)
    assert result["cache_hits"] == 1 and client.requests_made == 1
    checkpoint = next(tmp_path.rglob("block-*.npz"))
    checkpoint.write_bytes(b"corrupt")
    client = BytesClient(raw)
    fixed = mod.extract_conservative(*args, client=client, **kw)
    assert client.requests_made == 1
    np.testing.assert_equal(fixed["density"], result["density"])
    # An incomplete marker must also force a refetch.
    next(tmp_path.rglob("block-*.json")).write_text("{}")
    client = BytesClient(raw)
    mod.extract_conservative(*args, client=client, **kw)
    assert client.requests_made == 1
    client = BytesClient(raw)
    mod.extract_conservative(*args, client=client, **{**kw, "source_version": "v2"})
    assert client.requests_made == 2


def test_partial_run_is_explicitly_not_publishable(tmp_path):
    result = run(tmp_path, [[100], [200]], [0, 1], [60, 0], row_stop=1)
    assert not result["complete"]
    assert result["row_coverage_actual"] == 0.5
    assert 0 < result["processed_fraction"][0, 0] < 1


def test_checkpoint_budget_rejected_before_network(tmp_path):
    mod = implementation()
    raw, info = fixture([[100], [200]])
    client = BytesClient(raw)
    with pytest.raises(ValueError, match="budget"):
        mod.extract_conservative(info.url, info, np.array([0, 1]), np.array([60, 0]),
                                cache_dir=tmp_path, source_version="v1", client=client,
                                cache_budget_bytes=1)
    assert client.requests_made == 0


def test_checkpoint_budget_includes_other_source_namespaces(tmp_path):
    mod = implementation()
    raw, info = fixture([[100], [200]])
    # Preexisting reduced data belongs to another source; never delete it.
    other = tmp_path / "other-source"
    other.mkdir()
    (other / "retained.npz").write_bytes(b"x" * 20000)
    client = BytesClient(raw)
    with pytest.raises(ValueError, match="budget"):
        mod.extract_conservative(info.url, info, np.array([0, 1]), np.array([60, 0]),
                                cache_dir=tmp_path, source_version="v1", client=client,
                                cache_budget_bytes=21000)
    assert client.requests_made == 0
    assert (other / "retained.npz").stat().st_size == 20000


def test_missing_raster_type_fails_instead_of_guessing(tmp_path):
    mod = implementation()
    _, info = fixture([[100]])
    info.raster_type = None
    with pytest.raises(ValueError, match="RasterType"):
        mod.source_edges(info)


def test_parser_binds_header_to_verified_fetch_identity():
    raw, _ = fixture([[100], [100]])
    client = BytesClient(raw)
    info = TiffHeaderReader("fixture://soil", client.fetch).parse()
    assert getattr(info, "source_identity", None) == client.source_identity


@pytest.mark.parametrize("default_client", [False, True])
def test_stale_header_rejected_before_cache_or_pixel_reads(tmp_path, monkeypatch, default_client):
    mod = implementation()
    old_raw, _ = fixture([[100], [100]], lat=60)
    new_raw, _ = fixture([[200], [200]], lat=30)
    assert len(old_raw) == len(new_raw)
    old = BytesClient(old_raw)
    info = TiffHeaderReader("fixture://soil", old.fetch).parse()

    class FreshClient(BytesClient):
        def __init__(self):
            super().__init__(new_raw)
            self.source_identity = None

        def fetch(self, start, end):
            result = super().fetch(start, end)
            self.source_identity = (len(new_raw), "fixture-v2", None)
            return result

    client = FreshClient()
    monkeypatch.setattr(mod, "RangeClient", lambda url: client)
    with pytest.raises(ValueError, match="header.*identity"):
        mod.extract_conservative("fixture://soil", info, [0, 1], [60, 0],
                                 cache_dir=tmp_path, source_version="same-id", block_rows=1,
                                 client=None if default_client else client)
    assert client.requests_made == 1  # identity probe only, no pixels
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("provide_fresh_client", [False, True])
def test_api_primes_identity_before_resuming_http_checkpoints(tmp_path, provide_fresh_client):
    import socketserver
    import threading
    from range_client import RangeClient
    from test_synthetic import RangeHTTPHandler

    raw, info = fixture([[100], [100]])
    new_raw, _ = fixture([[200], [200]])
    assert len(raw) == len(new_raw)  # length alone cannot distinguish revision

    class Handler(RangeHTTPHandler):
        file_bytes = raw
        revision = '"v1"'

        def end_headers(self):
            self.send_header("ETag", type(self).revision)
            super().end_headers()

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/soil"
        def extract():
            header_client = RangeClient(url)
            current_info = TiffHeaderReader(url, header_client.fetch).parse()
            return implementation().extract_conservative(
                url, current_info, [0, 1], [60, 0], cache_dir=tmp_path,
                source_version="same-archive-id", block_rows=1,
                client=RangeClient(url) if provide_fresh_client else None)
        try:
            old = extract()
            Handler.file_bytes, Handler.revision = new_raw, '"v2"'
            new = extract()
            np.testing.assert_allclose(new["density"], [[200]])
            assert old["cache_identity"] != new["cache_identity"]
            assert new["source"]["http_identity"][1] == '"v2"'
            assert new["cache_hits"] == 0
            assert extract()["cache_hits"] == 2
        finally:
            server.shutdown()


def test_cli_failure_has_no_final_product_and_pilot_is_labelled(tmp_path):
    # Serve a real tagged TIFF; route the existing local fixture helper's bytes.
    from test_synthetic import RangeHTTPHandler
    import socketserver
    import threading

    raw, _ = fixture([[100, np.nan], [200, np.nan]])
    class Handler(RangeHTTPHandler):
        file_bytes = raw
        requests = 0

        def do_GET(self):
            type(self).requests += 1
            super().do_GET()

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/fixture"
        cmd = [sys.executable, str(Path(__file__).with_name("eth_raster.py")), "conservative", url,
               "--source-version", "test-v1", "--citation", "https://example.org/dataset",
               "--out-width", "1", "--out-height", "1", "--west", "0", "--east", "2",
               "--north", "60", "--south", "0", "--cache-dir", str(tmp_path / "cache"),
               "--out-dir", str(tmp_path / "output"), "--tag", "pilot",
               "--pilot-start-row", "0", "--pilot-rows", "1",
               "--request-interval", "0", "--rate-limit-cooldown", "0"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert Handler.requests <= 3, "batch header reads to avoid archive rate limiting"
            meta = json.loads((tmp_path / "output/pilot_meta.json").read_text())
            assert meta["complete"] is False and meta["row_coverage_actual"] == 0.5
            assert meta["provenance"]["sampling_depth"] == "unknown"
            with np.load(tmp_path / "output/pilot_grid.npz") as grid:
                assert set(grid.files) == {"density", "valid_model_fraction", "processed_fraction", "lon_edges", "lat_edges"}
            # Existing final products must never be silently overwritten.
            again = subprocess.run(cmd, capture_output=True, text=True)
            assert again.returncode != 0
            bad = cmd.copy()
            bad[bad.index("--tag") + 1] = "bad"
            bad[bad.index("--pilot-rows") + 1] = "100"
            failed = subprocess.run(bad, capture_output=True, text=True)
            assert failed.returncode != 0
            assert not (tmp_path / "output/bad_meta.json").exists()
            assert not (tmp_path / "output/bad_grid.npz").exists()
        finally:
            server.shutdown()
