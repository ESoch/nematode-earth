"""HTTP range-GET client with retries, built for the ETH Research Collection
DSpace 7 backend specifically.

Two things observed about that backend (see .superpowers/session2/pipeline-report.md
for the full evidence) drive the design here:

1. A bare `curl` / default `requests` user-agent gets a 403 from the front
   Apache/WAF layer; a normal browser User-Agent is required.
2. Per-connection throughput is heavily throttled (observed ~70-120 KB/s per
   connection, sometimes 0 -- outright connection stalls/resets). Retries
   with backoff are not optional; they are the normal case, not an edge case.
"""

from __future__ import annotations

import time
import re
import math
import logging
from email.utils import parsedate_to_datetime

import requests

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class RangeClient:
    def __init__(
        self,
        url: str,
        *,
        max_retries: int = 12,
        timeout: float = 40.0,
        backoff_base: float = 1.5,
        backoff_cap: float = 20.0,
        max_response_bytes: int = 32 * 1024 * 1024,
        min_request_interval: float = 0.0,
        rate_limit_cooldown: float = 60.0,
    ):
        if any(not math.isfinite(v) or v < 0 for v in (min_request_interval, rate_limit_cooldown)):
            raise ValueError("Request pacing and cooldown must be finite and non-negative")
        self.url = url
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.max_response_bytes = max_response_bytes
        self.source_identity = None
        self.min_request_interval = min_request_interval
        self.rate_limit_cooldown = rate_limit_cooldown
        self._next_request_at = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = _UA
        self.bytes_fetched = 0
        self.requests_made = 0
        self.retries_used = 0

    def fetch(self, start: int, end_inclusive: int) -> bytes:
        """GET bytes [start, end_inclusive] (HTTP Range semantics)."""
        expected = end_inclusive - start + 1
        if start < 0 or expected <= 0 or expected > self.max_response_bytes:
            raise ValueError("Range exceeds bounded response size or has invalid endpoints")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            wait = self._next_request_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next_request_at = time.monotonic() + self.min_request_interval
            self.requests_made += 1
            try:
                with self._session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end_inclusive}", "Accept-Encoding": "identity"},
                    timeout=self.timeout,
                    stream=True,
                ) as resp:
                    if resp.status_code in (429, 503):
                        cooldown = self.rate_limit_cooldown if resp.status_code == 429 else 0.0
                        retry_after = resp.headers.get("Retry-After", "")
                        try:
                            seconds = (float(retry_after) if retry_after.isdigit()
                                       else parsedate_to_datetime(retry_after).timestamp() - time.time())
                            if math.isfinite(seconds):
                                cooldown = max(cooldown, seconds)
                        except (ValueError, TypeError, OverflowError):
                            pass
                        # Retry-After is a server minimum, never truncated to
                        # the normal exponential-backoff cap. Keep this deadline
                        # even if the caller exhausts retries and calls again.
                        self._next_request_at = max(self._next_request_at, time.monotonic() + cooldown)
                        logging.getLogger(__name__).warning("HTTP %s: cooldown %.1fs", resp.status_code, cooldown)
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", resp.headers.get("Content-Range", ""))
                    if resp.status_code != 206 or not match:
                        raise RuntimeError(f"Expected 206 with a numeric Content-Range; got HTTP {resp.status_code}")
                    a, b, total = map(int, match.groups())
                    if (a, b) != (start, end_inclusive) or total <= b:
                        raise RuntimeError("Content-Range does not match requested bytes")
                    if resp.headers.get("Content-Encoding", "identity") != "identity":
                        raise RuntimeError("Encoded range responses are unsupported")
                    length = resp.headers.get("Content-Length")
                    if length is not None and length != str(expected):
                        raise RuntimeError("Content-Length does not match requested bytes")
                    identity = (total, resp.headers.get("ETag"), resp.headers.get("Last-Modified"))
                    if self.source_identity is not None and identity != self.source_identity:
                        raise RuntimeError("Source identity changed during ranged download")
                    data = bytearray()
                    for chunk in resp.iter_content(chunk_size=min(65536, expected + 1)):
                        if len(data) + len(chunk) > expected:
                            raise RuntimeError("Oversized range response")
                        data.extend(chunk)
                    if len(data) != expected:
                        raise RuntimeError(f"short read: got {len(data)} bytes, expected {expected}")
                    self.source_identity = identity
                self.bytes_fetched += len(data)
                return bytes(data)
            except (requests.RequestException, RuntimeError) as exc:
                last_exc = exc
                self.retries_used += 1
                # Connection resets happen routinely against this server; a
                # fresh connection sometimes recovers where retrying on the
                # same one won't.
                self._session.close()
                self._session = requests.Session()
                self._session.headers["User-Agent"] = _UA
                if attempt < self.max_retries - 1:
                    delay = min(self.backoff_base * (2**attempt), self.backoff_cap)
                    self._next_request_at = max(self._next_request_at, time.monotonic() + delay)
        raise RuntimeError(
            f"range fetch failed after {self.max_retries} attempts "
            f"({start}-{end_inclusive}): {last_exc}"
        )

    def fetch_header_probe(self) -> requests.Response:
        """HEAD request, mainly to confirm accept-ranges / content-length behavior."""
        return self._session.head(self.url, timeout=self.timeout)
