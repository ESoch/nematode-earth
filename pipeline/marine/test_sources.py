from marine.sources import discover_pangaea, discover_wei
import pytest
from marine import sources


def test_download_and_licence_must_come_from_landing_page():
    html = '''<meta name="description" content="Measured nematodes">
    <link rel="license" href="https://creativecommons.org/licenses/by/3.0/">
    <link rel="item" href="?format=textfile" type="text/tab-separated-values">'''
    source = discover_pangaea(html, "https://doi.pangaea.de/10.1594/PANGAEA.873272")
    assert source["download_url"] == "https://doi.pangaea.de/10.1594/PANGAEA.873272?format=textfile"
    assert source["license_url"] == "https://creativecommons.org/licenses/by/3.0/"


def test_wei_attachment_discovered_and_html_entities_decoded():
    html = '''<a href="article/file?id=10.1371/journal.pone.0015323.s001&amp;type=supplementary">File S1</a>
    <a href="https://creativecommons.org/licenses/by/4.0/">License</a>'''
    source = discover_wei(html, "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0015323")
    assert source["download_url"] == "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0015323.s001&type=supplementary"


def test_wei_records_unversioned_article_licence_without_inventing_version():
    html = '''<a href="article/file?type=supplementary&amp;id=10.1371/journal.pone.0015323.s001">S1</a>
    <p>Creative Commons Attribution License</p><a href="/plosone/s/licenses-and-copyright">License</a>'''
    source = discover_wei(html, "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0015323")
    assert source["license_url"] == "https://journals.plos.org/plosone/s/licenses-and-copyright"
    assert source["license_version"] is None


def test_download_bounds_and_hash_verification(tmp_path):
    # The response double substitutes only network IO; real persistence and checks run.
    class Response:
        url = "https://example.org/data"
        headers = {"Content-Type": "text/plain"}
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield b"abc"; yield b"def"
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Session:
        def get(self, url, **kwargs): return Response()
    assert hasattr(sources, "download"), "bounded, checksummed downloads not implemented"
    path = tmp_path / "data.txt"
    with pytest.raises(ValueError, match="limit"):
        sources.download(Session(), Response.url, path, limit=5)
    assert not path.exists()
    receipt = sources.download(Session(), Response.url, path, limit=6)
    assert receipt["sha256"] == "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
    assert sources.verify_download(path, receipt) == b"abcdef"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        sources.verify_download(path, receipt)
