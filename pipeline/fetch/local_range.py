"""Bounded, read-only ranges from a downloaded TIFF; no HTTP or raw copies.

The entire file is hashed once before checkpoint selection. Stat identity is
checked around each read and before CLI publication to reject concurrent edits.
The SHA-256 identifies local content, not a verified publisher checksum.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class LocalRangeClient:
    def __init__(self, path, *, max_response_bytes=32 * 1024 * 1024):
        self.path = Path(path).resolve(strict=True)
        if not self.path.is_file() or max_response_bytes < 1:
            raise ValueError("A regular local file and positive range bound are required")
        self.max_response_bytes = max_response_bytes
        self._initial = _fingerprint(self.path.stat())
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            if _fingerprint(os.fstat(stream.fileno())) != self._initial:
                raise RuntimeError("Local source changed before hashing")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.verify_unchanged()
        self.sha256 = digest.hexdigest()
        self.source_identity = (self._initial[2], "sha256:" + self.sha256, None)
        self.bytes_fetched = self.reads_made = 0
        # These fields remain strictly HTTP counters, including in CLI metadata.
        self.requests_made = self.retries_used = 0

    def verify_unchanged(self):
        try:
            current = _fingerprint(self.path.stat())
        except OSError as error:
            raise RuntimeError("Local source changed or disappeared") from error
        if current != self._initial:
            raise RuntimeError("Local source changed during extraction")

    def fetch(self, start, end_inclusive):
        length = end_inclusive - start + 1
        if start < 0 or length <= 0 or length > self.max_response_bytes:
            raise ValueError("Range exceeds bounded read size or has invalid endpoints")
        if end_inclusive >= self.source_identity[0]:
            raise ValueError("Requested range is outside local file")
        self.verify_unchanged()
        with self.path.open("rb") as stream:
            if _fingerprint(os.fstat(stream.fileno())) != self._initial:
                raise RuntimeError("Local source changed before read")
            stream.seek(start)
            data = stream.read(length)
            if _fingerprint(os.fstat(stream.fileno())) != self._initial:
                raise RuntimeError("Local source changed during read")
        self.verify_unchanged()
        if len(data) != length:
            raise RuntimeError("Short local file read")
        self.bytes_fetched += length
        self.reads_made += 1
        return data
