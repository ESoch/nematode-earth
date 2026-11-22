"""Minimal TIFF / BigTIFF header parser that works entirely over HTTP range
requests -- never downloads the pixel data itself.

Handles both classic TIFF (32-bit offsets, magic 42) and BigTIFF (64-bit
offsets, magic 43), either byte order. Only the tags we actually need for
row-addressable, uncompressed rasters are decoded:

  256  ImageWidth
  257  ImageLength
  258  BitsPerSample
  259  Compression
  277  SamplesPerPixel
  278  RowsPerStrip
  273  StripOffsets
  279  StripByteCounts
  339  SampleFormat            (1=uint, 2=int, 3=float)
  33550 ModelPixelScaleTag
  33922 ModelTiepointTag

This is deliberately not a general-purpose TIFF reader: it assumes (and
verifies) an uncompressed, single-image, striped layout, which is what the
ETH rasters use.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_OFFSETS = 273
TAG_STRIP_BYTE_COUNTS = 279
TAG_SAMPLE_FORMAT = 339
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922

# TIFF field type -> (struct code, size in bytes)
_TYPE_INFO = {
    1: ("B", 1),  # BYTE
    2: ("B", 1),  # ASCII (treated as bytes)
    3: ("H", 2),  # SHORT
    4: ("I", 4),  # LONG
    5: ("II", 8),  # RATIONAL (2x LONG) -- handled specially
    11: ("f", 4),  # FLOAT
    12: ("d", 8),  # DOUBLE
    16: ("Q", 8),  # LONG8 (BigTIFF)
    17: ("q", 8),  # SLONG8 (BigTIFF)
    18: ("Q", 8),  # IFD8 (BigTIFF)
}


@dataclass
class IfdEntry:
    tag: int
    type_: int
    count: int
    value_bytes: bytes  # raw inline value/offset field, still needs resolving
    is_offset: bool  # True if value_bytes holds an offset, not the data


@dataclass
class TiffInfo:
    url: str
    byte_order: str  # "<" or ">"
    big_tiff: bool
    ifd_offset: int
    width: int
    height: int
    bits_per_sample: int
    sample_format: int  # 1=uint 2=int 3=float
    samples_per_pixel: int
    compression: int
    rows_per_strip: int
    strip_offsets: list[int]
    strip_byte_counts: list[int]
    model_pixel_scale: tuple[float, float, float] | None
    model_tiepoint: tuple[float, ...] | None
    raw_entries: dict[int, IfdEntry] = field(default_factory=dict)
    raster_type: int | None = None  # GeoKey 1025: 1=area, 2=point; never guess
    geographic_crs: int | None = None
    model_type: int | None = None
    nodata: float | None = None
    source_identity: tuple | None = None  # HTTP identity of the parsed header

    @property
    def row_stride(self) -> int:
        """Bytes per fully addressable row, derived from strip 0/1 spacing."""
        return self.strip_offsets[1] - self.strip_offsets[0]

    @property
    def dtype_str(self) -> str:
        kind = {1: "u", 2: "i", 3: "f"}[self.sample_format]
        return f"{kind}{self.bits_per_sample // 8}"


def _np_dtype(byte_order: str, dtype_str: str) -> str:
    order = "<" if byte_order == "<" else ">"
    return f"{order}{dtype_str}"


class TiffHeaderReader:
    """Parses a TIFF/BigTIFF header via a caller-supplied range-fetch function.

    `fetch(start, end_inclusive) -> bytes` is expected to perform an HTTP
    Range GET for [start, end] inclusive, matching HTTP semantics.
    """

    def __init__(self, url: str, fetch, *, source_identity=None):
        self.url = url
        self._fetch = fetch
        self._source_identity = source_identity

    def parse(self) -> TiffInfo:
        header = self._fetch(0, 15)  # enough for either header flavor
        owner = getattr(self._fetch, "__self__", None)
        identity = self._source_identity
        if identity is None:
            identity = getattr(owner, "source_identity", None)
        identity = tuple(identity) if identity is not None else None
        bo_mark = header[0:2]
        if bo_mark == b"II":
            byte_order = "<"
        elif bo_mark == b"MM":
            byte_order = ">"
        else:
            raise ValueError(f"Not a TIFF: bad byte-order mark {bo_mark!r}")

        magic = struct.unpack(byte_order + "H", header[2:4])[0]
        if magic == 42:
            big_tiff = False
            ifd_offset = struct.unpack(byte_order + "I", header[4:8])[0]
        elif magic == 43:
            big_tiff = True
            offset_size, always_zero = struct.unpack(byte_order + "HH", header[4:8])
            if offset_size != 8:
                raise ValueError(f"Unexpected BigTIFF offset size {offset_size}")
            ifd_offset = struct.unpack(byte_order + "Q", header[8:16])[0]
        else:
            raise ValueError(f"Not a TIFF: bad magic {magic}")

        entries, next_ifd = self._read_ifd(byte_order, big_tiff, ifd_offset)

        def scalar(tag: int, default=None):
            e = entries.get(tag)
            if e is None:
                return default
            return self._resolve_scalar(byte_order, e)

        def array(tag: int):
            e = entries.get(tag)
            if e is None:
                return []
            return self._resolve_array(byte_order, e)

        width = scalar(TAG_IMAGE_WIDTH)
        height = scalar(TAG_IMAGE_LENGTH)
        bits_per_sample = scalar(TAG_BITS_PER_SAMPLE, 8)
        sample_format = scalar(TAG_SAMPLE_FORMAT, 1)
        samples_per_pixel = scalar(TAG_SAMPLES_PER_PIXEL, 1)
        compression = scalar(TAG_COMPRESSION, 1)
        rows_per_strip = scalar(TAG_ROWS_PER_STRIP, height)

        strip_offsets = array(TAG_STRIP_OFFSETS)
        strip_byte_counts = array(TAG_STRIP_BYTE_COUNTS)

        pixel_scale = array(TAG_MODEL_PIXEL_SCALE)
        tiepoint = array(TAG_MODEL_TIEPOINT)
        key_directory = array(34735)
        keys = {}
        if key_directory:
            if len(key_directory) != 4 + 4 * key_directory[3]:
                raise ValueError("Malformed GeoKeyDirectoryTag")
            for i in range(4, len(key_directory), 4):
                key, location, count, value = key_directory[i:i + 4]
                if location == 0 and count == 1:
                    keys[key] = value
        nodata_bytes = array(42113)
        nodata = float(bytes(nodata_bytes).rstrip(b"\x00").decode("ascii")) if nodata_bytes else None

        if owner is not None and getattr(owner, "source_identity", None) != identity:
            raise ValueError("Source identity changed while parsing TIFF header")

        return TiffInfo(
            url=self.url,
            byte_order=byte_order,
            big_tiff=big_tiff,
            ifd_offset=ifd_offset,
            width=width,
            height=height,
            bits_per_sample=bits_per_sample,
            sample_format=sample_format,
            samples_per_pixel=samples_per_pixel,
            compression=compression,
            rows_per_strip=rows_per_strip,
            strip_offsets=strip_offsets,
            strip_byte_counts=strip_byte_counts,
            model_pixel_scale=tuple(pixel_scale) if pixel_scale else None,
            model_tiepoint=tuple(tiepoint) if tiepoint else None,
            raw_entries=entries,
            raster_type=keys.get(1025),
            geographic_crs=keys.get(2048),
            model_type=keys.get(1024),
            nodata=nodata,
            source_identity=identity,
        )

    def _read_ifd(self, byte_order: str, big_tiff: bool, offset: int):
        if big_tiff:
            count_bytes = self._fetch(offset, offset + 7)
            n_entries = struct.unpack(byte_order + "Q", count_bytes)[0]
            entry_size = 20
            entries_start = offset + 8
        else:
            count_bytes = self._fetch(offset, offset + 1)
            n_entries = struct.unpack(byte_order + "H", count_bytes)[0]
            entry_size = 12
            entries_start = offset + 2

        block_len = n_entries * entry_size
        block = self._fetch(entries_start, entries_start + block_len - 1)

        entries: dict[int, IfdEntry] = {}
        for i in range(n_entries):
            chunk = block[i * entry_size : (i + 1) * entry_size]
            if big_tiff:
                tag, type_ = struct.unpack(byte_order + "HH", chunk[0:4])
                count = struct.unpack(byte_order + "Q", chunk[4:12])[0]
                value_field = chunk[12:20]
                _, type_size = _TYPE_INFO.get(type_, ("B", 1))
                is_offset = count * type_size > 8
            else:
                tag, type_ = struct.unpack(byte_order + "HH", chunk[0:4])
                count = struct.unpack(byte_order + "I", chunk[4:8])[0]
                value_field = chunk[8:12]
                _, type_size = _TYPE_INFO.get(type_, ("B", 1))
                is_offset = count * type_size > 4
            entries[tag] = IfdEntry(tag, type_, count, value_field, is_offset)

        next_off_bytes_len = 8 if big_tiff else 4
        next_off_raw = self._fetch(
            entries_start + block_len,
            entries_start + block_len + next_off_bytes_len - 1,
        )
        fmt = "Q" if big_tiff else "I"
        next_ifd = struct.unpack(byte_order + fmt, next_off_raw)[0]
        return entries, next_ifd

    def _resolve_scalar(self, byte_order: str, e: IfdEntry):
        values = self._resolve_array(byte_order, e)
        return values[0] if values else None

    def _resolve_array(self, byte_order: str, e: IfdEntry):
        fmt_char, size = _TYPE_INFO.get(e.type_, ("B", 1))
        total_bytes = e.count * size

        if e.is_offset:
            # value_field holds an 8- or 4-byte offset to the real data.
            if len(e.value_bytes) == 8:
                off = struct.unpack(byte_order + "Q", e.value_bytes)[0]
            else:
                off = struct.unpack(byte_order + "I", e.value_bytes)[0]
            raw = self._fetch(off, off + total_bytes - 1)
        else:
            raw = e.value_bytes[:total_bytes]

        if e.type_ == 5:  # RATIONAL: pairs of uint32 -> float
            n = e.count
            vals = struct.unpack(byte_order + "II" * n, raw)
            return [vals[2 * i] / vals[2 * i + 1] for i in range(n)]

        vals = struct.unpack(byte_order + fmt_char * e.count, raw)
        return list(vals)
