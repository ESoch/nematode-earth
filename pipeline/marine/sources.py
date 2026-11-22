"""Discover source downloads and licences from publisher landing pages."""
from html.parser import HTMLParser
from urllib.parse import urljoin
from pathlib import Path
import hashlib
from datetime import datetime, timezone


class Landing(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.links, self.meta = [], {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"a", "link"} and "href" in attrs:
            self.links.append(attrs)
        if tag == "meta" and "name" in attrs:
            self.meta[attrs["name"]] = attrs.get("content", "")


def discover_pangaea(html, landing_url):
    page = Landing(html)
    downloads = [a["href"] for a in page.links
                 if a.get("type") == "text/tab-separated-values"]
    licenses = [a["href"] for a in page.links if a.get("rel") == "license"]
    if len(set(downloads)) != 1 or not licenses:
        raise ValueError("missing or ambiguous published download/license")
    return dict(landing_url=landing_url, download_url=urljoin(landing_url, downloads[0]),
                license_url=licenses[0], citation=page.meta.get("description"))


def discover_wei(html, landing_url):
    page = Landing(html)
    downloads = [a["href"] for a in page.links if "article/file?" in a["href"]
                 and "journal.pone.0015323.s001" in a["href"]]
    licenses = [a["href"] for a in page.links if "creativecommons.org/licenses/" in a["href"]]
    if not licenses and "Creative Commons Attribution License" in html:
        licenses = [a["href"] for a in page.links if "licenses-and-copyright" in a["href"]]
    if len(set(downloads)) != 1 or not licenses:
        raise ValueError("missing or ambiguous Wei supplement/license")
    return dict(landing_url=landing_url, download_url=urljoin(landing_url, downloads[0]),
                license_url=urljoin(landing_url, licenses[0]), license_version=None,
                license_statement="Creative Commons Attribution; article does not specify version",
                citation="Wei et al. (2010), Global Patterns and Predictions of Seafloor Biomass Using Random Forests; File S1, doi:10.1371/journal.pone.0015323.s001")


def download(session, url, path, *, limit=35_000_000):
    """Bound decoded payload before persisting; every source is small enough for RAM."""
    chunks, size = [], 0
    with session.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > limit:
                raise ValueError(f"download exceeds byte limit: {url}")
            chunks.append(chunk)
        data = b"".join(chunks)
        receipt = dict(requested_url=url, resolved_url=response.url, bytes=size,
                       sha256=hashlib.sha256(data).hexdigest(),
                       retrieved_at=datetime.now(timezone.utc).isoformat(),
                       content_type=response.headers.get("Content-Type"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return receipt


def verify_download(path, receipt):
    data = Path(path).read_bytes()
    if len(data) != receipt["bytes"] or hashlib.sha256(data).hexdigest() != receipt["sha256"]:
        raise ValueError(f"checksum mismatch: {path}")
    return data
