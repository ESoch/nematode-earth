#!/usr/bin/env python3
"""Bake ETOPO 2022 bedrock elevation into cube maps, same machinery as
`bake_cube.py` (density): both projections (plain gnomonic + EAC), same
face-embedding table, same geodetic<->geocentric conversion, same
RGBA16Float raw-face-plus-JSON-sidecar format -- so the app can use one
sampler path for both fields, per the task brief.

Differs from the density bake in the one place it has to: the ENCODING.
Elevation is signed (Mariana Trench ~-10,752 m to Everest-region ~8,157 m
in this 60"-averaged grid) and there is no ocean/no-data population to
floor away (see fetch/etopo.py -- the -99999 sentinel is declared but
never actually present, confirmed by an exhaustive scan) -- so R is raw
signed meters, stored directly as float16, with no log encoding and no
floor. "Do not clamp negatives" per the brief: the seafloor's negative
elevation is exactly the content this bake exists to carry.

THREE resolutions (coarse first-paint tier through finer streamed tiers),
same "do not upsample past source" discipline as the density bake -- see
the resolution check in main(). Bakes BOTH projections at each tier.

G channel: a real, high-resolution land-fraction mask (see
`compute_land_mask`), registered pixel-for-pixel with R via the same
box-average + bilinear machinery. This exists because the density cube's
land mask (app/public/data/cube_plain_128, A channel) is baked from a
512x256 equirect extraction -- ~78 km/texel, eight times coarser than
this atlas's 9.77 km/texel -- and visibly smears the coastline when used
next to the finer relief. Bed elevation's OWN sign is not a substitute:
probing the baked 1024 atlas directly found West Antarctica would open up
as a hole in the continent under a naive `bed >= 0` rule --

    probe                          bed elev (baked, 1024)   bed>=0 verdict
    Antarctica (Vostok)            -245 m                   ocean -- WRONG
    Byrd Subglacial Basin          -913 m                   ocean -- WRONG
    Antarctica (Dome A)            +1916 m                  land -- ok
    Greenland summit               +209 m                   land -- ok

-- because bed elevation is bedrock, and ice sheets routinely sit on
bedrock below sea level. The fix needs the ETOPO *surface* raster
(ice-top where glaciated) as well as bed; see `compute_land_mask`.

Usage: uv run build/bake_elevation_cube.py [--face-sizes 64,128,256]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from eac import (
    FACE_COUNT,
    FACE_NAMES,
    direction_to_face_uv,
    direction_to_lonlat,
    face_direction,
    lonlat_to_direction,
    texel_solid_angles,
)
from raster import bilinear_sample
from wgs84 import R_MEAN, geocentric_to_geodetic_lat, geodetic_to_geocentric_lat

sys.path.insert(0, str(Path(__file__).parent.parent / "fetch"))
from etopo import (
    DEFAULT_SURFACE_CACHE,
    SURFACE_SOURCE_URL,
    box_average_decimate,
    load_elevation,
    verify_no_nodata,
)

PROJECTIONS = ["plain", "eac"]
SOURCE_HEIGHT, SOURCE_WIDTH = 10800, 21600  # ETOPO 60" grid
SOURCE_LON0, SOURCE_LAT0 = -180.0, 90.0
SOURCE_DLON = 360.0 / SOURCE_WIDTH
SOURCE_DLAT = 180.0 / SOURCE_HEIGHT

# Ice density over seawater density -- a PHYSICAL CONSTANT (rho_ice ~= 917
# kg/m^3, rho_water ~= 1000 kg/m^3), not a tuned threshold. See `is_grounded`.
# A first version of this rule used a bare ice-thickness test
# (surface-bed > 1.0) instead of a real flotation test; that shipped, and
# was wrong -- floating sea ice and ice shelves also have real thickness
# sitting on deep water, so it classified open ocean under floating ice as
# land. Visible in the render as a hard-edged rotated rectangle around
# Greenland (the BedMachine Greenland source mosaic's polar-stereographic
# extent, reprojected onto the sphere) and, less visibly, a broad
# contaminated band around the Antarctic ice shelves. Do NOT go back to a
# bare thickness test -- use the physical constant below, not a retuned one.
RHO_ICE_OVER_RHO_WATER = 0.917


def is_grounded(bed_grid: np.ndarray, surface_grid: np.ndarray) -> np.ndarray:
    """Archimedes flotation test: is an ice column of thickness
    T = surface - bed resting on bedrock (grounded), or floating on the
    ocean?

    A floating column of thickness T displaces a draft (submerged depth
    below the waterline) of `(rho_ice/rho_water) * T` -- the standard
    iceberg-flotation relation. The ice is grounded exactly when that draft
    would reach past the seafloor, i.e. when the seafloor is shallower than
    the draft:

        bed + (rho_ice/rho_water) * T > 0

    Strictly greater than, so the flotation boundary itself is not counted
    as grounded (matches the open-ocean case T=0, bed<0, which must always
    read not-grounded).

    See the RHO_ICE_OVER_RHO_WATER docstring above for why this replaced a
    bare thickness test."""
    thickness = surface_grid - bed_grid
    return (bed_grid + RHO_ICE_OVER_RHO_WATER * thickness) > 0.0


def compute_land_mask(bed_grid: np.ndarray, surface_grid: np.ndarray) -> np.ndarray:
    """Boolean per-source-cell land mask:

        is_grounded(bed_elev, surface_elev)

    NOT `surface_elev >= 0 OR is_grounded(...)` -- an earlier version of
    this rule had that extra `surface_elev >= 0` clause (added to make the
    ordinary no-ice dry-land case trivially true), and it was ALSO wrong,
    independently of the bare-thickness bug `is_grounded` itself fixed:
    floating ice shelves have positive freeboard by construction (an ice
    shelf is thick enough that ~(1 - rho_ice/rho_water) ~= 8% of it pokes
    above the waterline), so `surface_elev >= 0` is true almost everywhere
    on a floating shelf REGARDLESS of grounding, and that clause alone kept
    misclassifying real ocean under real floating ice shelves as land even
    after `is_grounded` shipped. Measured directly on this ETOPO grid:
    ~1,000,000 of 233,280,000 source cells (~0.43% of Earth's surface,
    concentrated at 60-85 S -- the Antarctic ice shelves, e.g. Ross,
    Ronne-Filchner, Amery, Larsen C -- with a smaller amount at 75-85 N)
    were land ONLY because of that clause, with bed a clearly-oceanic
    -100..-3000 m and real ice thickness (mean ~850 m) underneath.

    `is_grounded` alone is not just simpler but MORE correct: for the
    ordinary no-ice case (surface==bed, T=0) it reduces exactly to
    `bed_elev > 0`, which is what the removed clause was providing anyway
    -- so dropping it changes nothing for dry land and fixes ice shelves.

    Bed elevation's sign alone is NOT a land mask either (see module
    docstring's probe table): ETOPO's bed raster is signed bedrock
    elevation, which for an ice sheet sitting on ground below sea level
    (e.g. the Byrd Subglacial Basin, -913 m) reads as ocean even though
    the ice on top is clearly land -- `is_grounded` (using `surface_elev`
    too) is what resolves this.

    Returns a boolean array, same shape as the inputs. `box_average_decimate`
    of this array (cast to float) is exactly the area-weighted-fraction G
    channel the app needs: a box mean of 0/1 values IS the fraction of 1s
    in that block.

    KNOWN, ACCEPTED LIMITATION: endorheic below-datum basins (Death Valley
    -86 m, Turpan, the Dead Sea) read as water under this rule. They are
    misclassified at ANY resolution, not merely under-sampled: with no ice
    the rule reduces exactly to bed > 0, so a below-datum basin reads as
    water however finely it is sampled. Being sub-texel only hid the
    consequence, and no longer does -- this note was written when 1024
    (9.77 km/texel) was the finest tier; at 2048 (4.886 km/texel) Death
    Valley's ~15 km spans about three texels. The alternative --
    rasterizing Natural Earth coastline vectors -- costs a geo-stack
    dependency this pipeline deliberately does not have. Documented in
    cube_meta.json's channel doc too, so this isn't "fixed" by accident."""
    return is_grounded(bed_grid, surface_grid)


def bake_elevation_face(
    face: int,
    size: int,
    elev_grid: np.ndarray,
    mask_grid: np.ndarray,
    projection: str,
):
    """Returns rgba_uint16 [size,size,4]. No polar gap, no floor -- ETOPO
    covers the full sphere and every extracted pixel is real data.

    `elev_grid` is the per-tier INTERMEDIATE decimated grid built in
    main() (not the raw 21600x10800 source), so its own dlon/dlat --
    derived from ITS shape, not the source constants -- is what
    bilinear_sample must use. Using the source's finer SOURCE_DLON/DLAT
    here while indexing this coarser grid was an earlier bug: row/col
    fractional indices came out far too large, got clamped/wrapped, and
    every face ended up repeatedly sampling a small corner of the grid --
    visible as baked min/max collapsing to a ~500 m band instead of the
    true -10752..8157 m range, and volume integrals off by 2-3 orders of
    magnitude. Caught by the mass-conservation gate, not by inspection.

    `mask_grid` (the land-fraction field, see `compute_land_mask`) MUST be
    the same shape as `elev_grid` -- both are built by `box_average_decimate`
    over the SAME inter_h/inter_w for a given tier -- so that sampling both
    through the identical (lon,lat)->(row,col) math below keeps R and G
    pixel-for-pixel registered."""
    if mask_grid.shape != elev_grid.shape:
        raise ValueError(
            f"mask_grid shape {mask_grid.shape} != elev_grid shape "
            f"{elev_grid.shape} -- both must come from the same "
            f"box_average_decimate call for R/G to stay registered"
        )
    inter_h, inter_w = elev_grid.shape
    dlon = 360.0 / inter_w
    dlat = 180.0 / inter_h

    idx = (np.arange(size) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(idx, idx)
    x, y, z = face_direction(face, u, v, projection)
    norm = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm

    lon_rad, lat_gc_rad = direction_to_lonlat(x, y, z)
    lat_gd_rad = np.vectorize(geocentric_to_geodetic_lat)(lat_gc_rad)
    lon_deg = np.degrees(lon_rad)
    lat_gd_deg = np.clip(np.degrees(lat_gd_rad), -90.0, 90.0)

    elevation = bilinear_sample(
        elev_grid,
        lon_deg,
        lat_gd_deg,
        SOURCE_LON0,
        SOURCE_LAT0,
        dlon,
        dlat,
    )
    land_fraction = bilinear_sample(
        mask_grid,
        lon_deg,
        lat_gd_deg,
        SOURCE_LON0,
        SOURCE_LAT0,
        dlon,
        dlat,
    )
    # Mathematically already in [0,1] (box means of 0/1 values, bilinearly
    # blended -- a convex combination of in-range values stays in range),
    # but clip anyway as cheap insurance against float roundoff, matching
    # the density bake's same defensive clip on its land-fraction channel.
    land_fraction = np.clip(land_fraction, 0.0, 1.0)

    rgba = np.zeros((size, size, 4), dtype=np.float16)
    rgba[..., 0] = elevation.astype(
        np.float16
    )  # R: signed elevation, metres, unclamped
    rgba[..., 1] = land_fraction.astype(np.float16)  # G: land fraction (0..1)
    rgba[..., 2] = np.float16(0.0)  # B: reserved
    rgba[..., 3] = np.float16(1.0)  # A: validity (always 1 -- no nodata in source)
    return rgba.view(np.uint16)


# Elevation thresholds (m) for the hypsometric-curve comparison -- area
# fraction of the sphere at or below each threshold. Spans trench to
# summit with finer spacing near sea level, where most of the area-weighted
# action is.
HYPSOMETRY_THRESHOLDS = [
    -10000,
    -6000,
    -4000,
    -3000,
    -2000,
    -1000,
    -200,
    0,
    200,
    1000,
    2000,
    4000,
    6000,
    8000,
]


def _row_areas_m2(out_h: int, out_w: int) -> np.ndarray:
    """Exact spherical lat-band area per row (m^2), shape (out_h,), full
    +/-90 coverage (no clipping, unlike the density raster). Broadcast
    across columns by the caller (equirect columns are equal-area within
    a row)."""
    lat_edges = SOURCE_LAT0 - np.linspace(0, 180.0, out_h + 1)
    band_area = (
        R_MEAN**2
        * math.radians(360.0)
        * (np.sin(np.radians(lat_edges[:-1])) - np.sin(np.radians(lat_edges[1:])))
    )
    return band_area / out_w  # per-pixel area, one value per row


def elevation_stats(elevation: np.ndarray, area_m2: np.ndarray) -> dict:
    """Multiple independent checks on an area-weighted elevation field,
    given as flat parallel arrays. A single signed volume-integral ratio
    is a weak gate here: land (+) and ocean (-) volumes are within an
    order of magnitude of each other, so a large absolute error can hide
    behind near-total cancellation and still report a ratio near 1 (or,
    as happened with a real bug this session, a small error in AREA
    weighting made the ratio wildly wrong even though sign was right --
    a signed-only check gave one number to be fooled by; these don't):

      - volume_integral: signed, m^3 (the original check -- kept, but
        not relied on alone)
      - abs_volume_integral: integral of |elevation|*area -- no
        cancellation, behaves like the density gate (large number, small
        errors perturb it only slightly)
      - land_volume / ocean_volume: same signed integral, split by
        elevation>0 vs <=0 -- each is single-signed, so each alone
        behaves like the density gate too
      - mean_elevation: volume_integral / total_area -- compare to the
        textbook ~-2440 m global figure
      - land_area_fraction: sanity check against the textbook ~29%
      - hypsometry: area fraction at/below each of HYPSOMETRY_THRESHOLDS

    Note: `land_area_fraction` here is derived from R (elevation > 0), a
    SEPARATE, weaker signal than the G-channel land mask (`land_mask_stats`
    below) -- this one is a self-consistency sanity check on elevation
    alone (kept for backward compatibility with the pre-G-channel bake),
    not the gate that binds; `assert_land_mask_valid` is."""
    total_area = float(area_m2.sum())
    volume = float((elevation * area_m2).sum())
    abs_volume = float((np.abs(elevation) * area_m2).sum())
    land = elevation > 0
    land_volume = float((elevation[land] * area_m2[land]).sum())
    ocean_volume = float((elevation[~land] * area_m2[~land]).sum())
    land_area_fraction = float(area_m2[land].sum() / total_area)
    hypsometry = {}
    for t in HYPSOMETRY_THRESHOLDS:
        hypsometry[str(t)] = float(area_m2[elevation <= t].sum() / total_area)
    return {
        "total_area_m2": total_area,
        "volume_integral_m3": volume,
        "abs_volume_integral_m3": abs_volume,
        "land_volume_m3": land_volume,
        "ocean_volume_m3": ocean_volume,
        "mean_elevation_m": volume / total_area,
        "land_area_fraction": land_area_fraction,
        "hypsometry_area_fraction_at_or_below": hypsometry,
    }


def volume_integral(elev_grid: np.ndarray, out_h: int, out_w: int) -> float:
    """Backward-compatible single-number signed integral (m^3) -- kept for
    the existing tests/callers; `source_stats`/`baked_stats` below are the
    real (multi-check) validation now used in main()."""
    row_area = _row_areas_m2(out_h, out_w)
    return float((elev_grid * row_area[:, None]).sum())


def source_stats(elev_grid: np.ndarray) -> dict:
    out_h, out_w = elev_grid.shape
    row_area = _row_areas_m2(out_h, out_w)
    area = np.broadcast_to(row_area[:, None], elev_grid.shape)
    return elevation_stats(elev_grid.ravel(), area.ravel())


def baked_volume_integral(
    faces_rgba: list[np.ndarray], face_size: int, projection: str
) -> float:
    """Backward-compatible single-number signed integral (m^3)."""
    solid_angles = texel_solid_angles(face_size, projection)
    total = 0.0
    for rgba_u16 in faces_rgba:
        r_u16 = rgba_u16[..., 0]
        elevation = r_u16.view(np.float16).astype(np.float64)
        total += float((elevation * solid_angles).sum()) * R_MEAN**2
    return total


def baked_stats(faces_rgba: list[np.ndarray], face_size: int, projection: str) -> dict:
    solid_angles = texel_solid_angles(face_size, projection)
    area = solid_angles * R_MEAN**2  # m^2 per texel
    all_elev, all_area = [], []
    for rgba_u16 in faces_rgba:
        r_u16 = rgba_u16[..., 0]
        elevation = r_u16.view(np.float16).astype(np.float64)
        all_elev.append(elevation.ravel())
        all_area.append(area.ravel())
    return elevation_stats(np.concatenate(all_elev), np.concatenate(all_area))


# ---------------------------------------------------------------------------
# Land-mask validation gate: hard-fails the bake (raises), does not warn.
# ---------------------------------------------------------------------------

# Accepted Earth land fraction ~0.292; ETOPO's +/-90 coverage needs no polar
# clipping correction here (unlike the density cube's +/-88 clip).
LAND_FRACTION_MIN = 0.285
LAND_FRACTION_MAX = 0.299

# (name, lat_deg, lon_deg, expected) -- probe points from the task brief,
# plus three floating-ice regression probes added after two related bugs
# (see is_grounded's and compute_land_mask's docstrings): a bare
# thickness test that fired on any floating ice, and (after fixing that) a
# `surface>=0` clause that independently kept floating ICE SHELVES (which
# have positive freeboard) misclassified. Antarctica/Greenland probes
# exercise grounded ice; the Baffin Bay / Arctic / Ross Ice Shelf probes
# exercise FLOATING ice, which must NOT read as land. Mid-Atlantic point
# chosen well clear of any coast (equatorial, deep open water).
LAND_MASK_PROBES = [
    ("Vostok", -78.46, 106.84, "land"),
    ("Byrd Subglacial Basin", -80.0, -120.0, "land"),
    ("Dome A", -80.37, 77.35, "land"),
    ("Greenland summit", 72.58, -38.46, "land"),
    ("Amazon/Manaus", -3.1, -60.0, "land"),
    ("Mid-Pacific abyssal", 0.0, -140.0, "ocean"),
    ("Mid-Atlantic", 0.0, -30.0, "ocean"),
    # Deep water (bed ~-1300..-1450 m) under floating sea ice/thin shelf
    # ice within the BedMachine Greenland mosaic extent -- the exact class
    # of texel the bare-thickness rule got wrong (measured ~0.98 "land"
    # under the old rule at this point in the shipped 1024 atlas).
    ("Baffin Bay", 70.0, -61.0, "ocean"),
    # Deep central Arctic Ocean basin (bed ~-2000..-4500 m in a >=1.5deg
    # neighborhood, confirmed clean of any land/grounded ice at that
    # radius) north of Greenland -- far enough from any coastline/
    # grounding line to read near-zero even at the coarsest (64) tier,
    # unlike a point picked for being exactly on a contaminated texel in
    # the fine atlas, which is by construction close to a real coastline
    # and legitimately blends at coarse resolution (see the brief: coastal
    # texels are SUPPOSED to blend, not read exactly 0 or 1).
    ("Arctic Ocean (north of Greenland)", 86.0, -40.0, "ocean"),
    # Deep interior of the Ross Ice Shelf, the world's largest floating
    # ice shelf (bed ~-700..-900 m nearby, real ice on top, real positive
    # freeboard). A `surface>=0 OR is_grounded` rule reads points like this
    # as land (freeboard alone satisfies the first clause). Picked, and
    # confirmed against the actual bake at every tier 64/128/256/1024, to
    # sit in a >=3x12 deg neighborhood that's ~99% not-grounded -- deep
    # shelf interior, not a local pinning point/ice rise/island (Ross
    # Island, Roosevelt Island etc. really are grounded and really are
    # inside the shelf's extent, so a probe near one would legitimately
    # blend, per the brief's "coastal texels blend" rule -- that is not
    # what this probe is checking).
    ("Ross Ice Shelf", -79.1, 178.0, "ocean"),
]


def geodetic_probe_to_face_uv(
    lat_deg: float, lon_deg: float, projection: str
) -> tuple[int, float, float]:
    """A cube-map direction is inherently geocentric (asin(z) has no
    ellipsoid concept), so a GEODETIC probe latitude (as given in the task
    brief / real-world coordinates) must be converted to geocentric BEFORE
    building the direction vector -- the inverse of what `bake_elevation_face`
    does when it takes a cube direction back to geodetic lat for sampling
    the source raster. Getting this backwards is a known trap in this repo
    (an earlier GHSL polar check compared `abs(lat) > grid_lat_min` where
    the bound was signed negative, silently zeroing an entire bake): the
    geocentric latitude must sit slightly EQUATORWARD of the geodetic one
    in the northern hemisphere, not the reverse -- see
    test_probe_conversion_moves_equatorward_in_northern_hemisphere."""
    lat_gd_rad = math.radians(lat_deg)
    lat_gc_rad = geodetic_to_geocentric_lat(lat_gd_rad)
    lon_rad = math.radians(lon_deg)
    x, y, z = lonlat_to_direction(lon_rad, lat_gc_rad)
    return direction_to_face_uv(float(x), float(y), float(z), projection)


def _sample_channel_nearest(
    faces_rgba: list[np.ndarray],
    face_size: int,
    face: int,
    u: float,
    v: float,
    channel: int,
) -> float:
    """Nearest-texel lookup (not bilinear) -- the probe check only needs
    "is this baked texel land/ocean", and nearest avoids blending across a
    coastline for probes that are deliberately deep in one regime or the
    other."""
    px = int(np.clip(round((u + 1) / 2 * face_size - 0.5), 0, face_size - 1))
    py = int(np.clip(round((v + 1) / 2 * face_size - 0.5), 0, face_size - 1))
    plane = faces_rgba[face][..., channel].view(np.float16)
    return float(plane[py, px])


def probe_land_mask(
    faces_rgba: list[np.ndarray],
    face_size: int,
    projection: str,
    probes: list[tuple[str, float, float, str]] = LAND_MASK_PROBES,
) -> dict:
    results = {}
    for name, lat, lon, expect in probes:
        face, u, v = geodetic_probe_to_face_uv(lat, lon, projection)
        g = _sample_channel_nearest(faces_rgba, face_size, face, u, v, channel=1)
        results[name] = {
            "lat": lat,
            "lon": lon,
            "expect": expect,
            "face": face,
            "u": u,
            "v": v,
            "g": g,
        }
    return results


def land_mask_stats(
    faces_rgba: list[np.ndarray], face_size: int, projection: str
) -> dict:
    """Global area-weighted land fraction from the G channel. Uses raw
    solid angle (not scaled by R_MEAN^2) since this is a ratio -- the R^2
    factor cancels."""
    solid_angles = texel_solid_angles(face_size, projection)
    total_area = 0.0
    land_area = 0.0
    for rgba_u16 in faces_rgba:
        g = rgba_u16[..., 1].view(np.float16).astype(np.float64)
        total_area += float(solid_angles.sum())
        land_area += float((g * solid_angles).sum())
    return {
        "global_land_fraction": land_area / total_area,
        "accepted_range": [LAND_FRACTION_MIN, LAND_FRACTION_MAX],
    }


def assert_land_mask_valid(
    faces_rgba: list[np.ndarray], face_size: int, projection: str
) -> dict:
    """Hard-fail gate: raises AssertionError (not a printed warning) if the
    baked G channel doesn't look like a real land mask. Called from
    bake_one for every tier/projection actually baked, so a bad bake stops
    the pipeline rather than shipping silently-wrong data."""
    stats = land_mask_stats(faces_rgba, face_size, projection)
    frac = stats["global_land_fraction"]
    assert LAND_FRACTION_MIN <= frac <= LAND_FRACTION_MAX, (
        f"[{projection} size={face_size}] global land fraction {frac:.4f} "
        f"outside accepted [{LAND_FRACTION_MIN}, {LAND_FRACTION_MAX}] "
        f"(Earth ~0.292) -- G channel does not look like a real land mask"
    )
    probes = probe_land_mask(faces_rgba, face_size, projection)
    for name, lat, lon, expect in LAND_MASK_PROBES:
        g = probes[name]["g"]
        if expect == "land":
            assert g >= 0.99, (
                f"[{projection} size={face_size}] probe {name!r} "
                f"({lat},{lon}) expected land (G>=0.99), got G={g:.4f}"
            )
        else:
            assert g <= 0.01, (
                f"[{projection} size={face_size}] probe {name!r} "
                f"({lat},{lon}) expected ocean (G<=0.01), got G={g:.4f}"
            )
    stats["probes"] = probes
    return stats


def bake_one(
    projection: str,
    size: int,
    elev_grid: np.ndarray,
    mask_grid: np.ndarray,
    out_dir: Path,
    src_stats: dict,
    nodata_audit: dict,
    surface_nodata_audit: dict,
) -> dict:
    faces_rgba = []
    for face in range(FACE_COUNT):
        rgba_u16 = bake_elevation_face(face, size, elev_grid, mask_grid, projection)
        faces_rgba.append(rgba_u16)
        print(
            f"  [{projection} size={size}] face {face} ({FACE_NAMES[face]}) baked",
            file=sys.stderr,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for face, rgba_u16 in enumerate(faces_rgba):
        raw = rgba_u16.astype("<u2").tobytes()
        (out_dir / f"face_{face}.bin").write_bytes(raw)
        total_bytes += len(raw)

    bk_stats = baked_stats(faces_rgba, size, projection)
    # Hard gate -- raises AssertionError (stopping the bake) if the G
    # channel doesn't look like a real land mask. Must run AFTER faces are
    # written is fine (files get overwritten on the next successful run),
    # but it must run before returning "success" metadata.
    land_mask_val = assert_land_mask_valid(faces_rgba, size, projection)
    km_per_texel = (90.0 / size) * (math.pi / 180.0) * R_MEAN / 1000.0

    meta = {
        "field": "elevation",
        "units": "metres",
        "projection": projection,
        "format": "RGBA16Float (uint16 little-endian bit patterns), row-major, "
        "index = (px + py*face_size)*4 + channel",
        "face_size": size,
        "km_per_texel": km_per_texel,
        "face_order": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        "face_convention": "matches THREE.CubeTexture images order; texel (px,py) "
        "-> u,v=((idx+0.5)/size)*2-1 in [-1,1]; row 0 = v=-1 "
        "(flipY=false, matches app/src/field/bakeCube.ts)",
        "geodetic_to_geocentric": {
            "applied": True,
            "wgs84_flattening": 1 / 298.257223563,
        },
        "channels": {
            "R": "signed elevation in metres, raw (no log encoding, no clamping of "
            "negatives -- seafloor depth is real content, not a defect)",
            "G": "area-weighted fraction (0..1) of contributing ETOPO 60-arcsec "
            "source cells satisfying is_grounded(bed_elev, surface_elev) -- a "
            "real land-fraction mask, matching the app's existing landFraction "
            "semantics (continuous, clamp01-blended, not boolean). Registered "
            "pixel-for-pixel with R via the same box-average+bilinear "
            "machinery, so a given texel's R and G always come from the same "
            "source cells. NOT derived from R: bed elevation's sign alone "
            "misclassifies ice sheets sitting on below-sea-level bedrock (e.g. "
            "West Antarctica) as ocean -- see module docstring for the probe "
            "table that demonstrated this. is_grounded is an Archimedes "
            "flotation test (bed + (rho_ice/rho_water)*(surface-bed) > 0, "
            "rho_ice/rho_water=0.917) applied ALONE, with no extra "
            "surface_elev>=0 clause -- two earlier versions of this rule were "
            "each wrong in a different way and both shipped before this one: "
            "(1) a bare ice-thickness test (surface-bed>1.0) fired on any "
            "floating ice regardless of depth, misclassifying open ocean "
            "under floating sea ice as land (the Greenland/Baffin Bay hard-"
            "edged-rectangle bug); (2) replacing it with is_grounded but "
            "keeping a `surface_elev>=0 OR` clause still misclassified "
            "floating ICE SHELVES as land, because a floating shelf has "
            "positive freeboard by construction (~8% of its thickness pokes "
            "above the waterline) -- measured ~1,000,000 of 233,280,000 "
            "source cells (~0.43% of Earth, concentrated at 60-85S: Ross, "
            "Ronne-Filchner, Amery, Larsen C) wrongly land under (2) alone. "
            "is_grounded with no OR clause fixes both: for the ordinary "
            "no-ice case it reduces exactly to bed_elev>0, so dry land is "
            "unaffected. Do not add back either a thickness-only test or a "
            "surface>=0 clause. KNOWN LIMITATION (do not 'fix' by adjusting "
            "the rule): sub-texel endorheic below-datum basins (Death Valley "
            "-86 m, Turpan, the Dead Sea) read as water -- each is far "
            "smaller than a source cell at every tier shipped here (Death "
            "Valley ~15 km wide vs 9.77 km/texel), and rasterizing coastline "
            "vectors to fix it would add a geo-stack dependency this "
            "pipeline deliberately does not have.",
            "B": "reserved (0)",
            "A": 'validity: always 1.0 -- ETOPO 2022 60" declares a -99999 nodata '
            "sentinel that is never actually present (see nodata_audit below, "
            "exhaustively verified over all 233,280,000 source pixels)",
        },
        "nodata_audit": nodata_audit,
        "surface_nodata_audit": surface_nodata_audit,
        "source": {
            "url": "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/"
            "60s_bed_elev_gtif/ETOPO_2022_v1_60s_N90W180_bed.tif",
            "resolution": "60 arc-second (1/60 deg), 21600x10800, full +/-90 coverage",
            "stats": src_stats,
        },
        "surface_source": {
            "url": SURFACE_SOURCE_URL,
            "resolution": "60 arc-second (1/60 deg), 21600x10800, full +/-90 coverage",
            "note": "used only to compute the G-channel land mask (compute_land_mask) "
            "-- not baked into any channel directly.",
        },
        "validation": {
            "baked_stats": bk_stats,
            "ratio_volume_integral_baked_to_source": bk_stats["volume_integral_m3"]
            / src_stats["volume_integral_m3"],
            "ratio_abs_volume_integral_baked_to_source": bk_stats[
                "abs_volume_integral_m3"
            ]
            / src_stats["abs_volume_integral_m3"],
            "ratio_land_volume_baked_to_source": bk_stats["land_volume_m3"]
            / src_stats["land_volume_m3"],
            "ratio_ocean_volume_baked_to_source": bk_stats["ocean_volume_m3"]
            / src_stats["ocean_volume_m3"],
            "note": "a single signed volume-integral ratio is a weak check here (land "
            "and ocean volumes are within an order of magnitude of each other "
            "and partially cancel) -- abs/land/ocean ratios don't share that "
            "cancellation risk and should all be checked, not just the first one",
            "land_mask": land_mask_val,
        },
        "bytes_per_face": total_bytes // FACE_COUNT,
        "total_bytes": total_bytes,
    }
    with open(out_dir / "cube_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--face-sizes", type=str, default="64,128,256")
    ap.add_argument("--out-root", type=Path, default=Path("../app/public/data"))
    ap.add_argument(
        "--projections",
        type=str,
        default=",".join(PROJECTIONS),
        help="Comma-separated subset of {plain,eac} to bake at every requested "
        "face size (default: both). High tiers are plain-only per the app's "
        "measured EAC-vs-plain parity -- worst-case texel improvement 1.27x, "
        "identical RMS -- which doesn't justify EAC's extra runtime ALU cost "
        "at large face sizes.",
    )
    args = ap.parse_args()
    face_sizes = [int(s) for s in args.face_sizes.split(",")]
    projections = [p.strip() for p in args.projections.split(",")]
    for p in projections:
        if p not in PROJECTIONS:
            ap.error(
                f"--projections: unknown projection {p!r} (must be one of {PROJECTIONS})"
            )

    full = load_elevation()
    nodata_audit = verify_no_nodata(full)
    print(f"bed nodata audit: {nodata_audit}", file=sys.stderr)

    full_surface = load_elevation(DEFAULT_SURFACE_CACHE, SURFACE_SOURCE_URL)
    surface_nodata_audit = verify_no_nodata(full_surface)
    print(f"surface nodata audit: {surface_nodata_audit}", file=sys.stderr)

    land_mask_full = compute_land_mask(full, full_surface).astype(np.float64)
    print(
        f"full-resolution land mask: mean (unweighted) fraction "
        f"{land_mask_full.mean():.4f}",
        file=sys.stderr,
    )

    src_km_per_px = SOURCE_DLON * (math.pi / 180.0) * R_MEAN / 1000.0
    print(
        f"source resolution: {src_km_per_px:.3f} km/px at equator (60 arc-second)",
        file=sys.stderr,
    )

    results = {}
    for size in face_sizes:
        face_km_per_texel = (90.0 / size) * (math.pi / 180.0) * R_MEAN / 1000.0
        if face_km_per_texel < src_km_per_px:
            print(
                f"WARNING: face_size={size} ({face_km_per_texel:.2f} km/texel) is finer "
                f"than the source's {src_km_per_px:.2f} km/px -- would invent detail.",
                file=sys.stderr,
            )

        # Extract this tier's equirect intermediate at a modest oversampling
        # of the target cube resolution (2x linear) so the cube's bilinear
        # sample draws on a properly box-averaged field, not a second
        # nearest/bilinear pass directly off 21600x10800 -- consistent with
        # "do not upsample past source" since the intermediate itself is
        # still far coarser than the 60" native grid at every tier here.
        inter_h = max(64, min(SOURCE_HEIGHT, size * 4))
        inter_w = inter_h * 2
        elev_grid = box_average_decimate(full, inter_h, inter_w)
        # Same decimation, same inter_h/inter_w, applied to the boolean land
        # mask instead of raw elevation -- a box mean of 0/1 values is
        # exactly the area-weighted fraction of land cells in that block,
        # and reusing the identical box_average_decimate call keeps this
        # grid registered with elev_grid cell-for-cell.
        mask_grid = box_average_decimate(land_mask_full, inter_h, inter_w)
        src_stats = source_stats(elev_grid)
        print(
            f"size={size}: intermediate grid {inter_w}x{inter_h}, "
            f"source volume_integral={src_stats['volume_integral_m3']:.4e} m^3, "
            f"mean_elevation={src_stats['mean_elevation_m']:.1f} m, "
            f"land_area_fraction={src_stats['land_area_fraction']:.4f}, "
            f"mask_grid mean={mask_grid.mean():.4f}",
            file=sys.stderr,
        )

        for projection in projections:
            out_dir = args.out_root / f"elevation_{projection}_{size}"
            results[f"{projection}_{size}"] = bake_one(
                projection,
                size,
                elev_grid,
                mask_grid,
                out_dir,
                src_stats,
                nodata_audit,
                surface_nodata_audit,
            )

    total_bytes_all = sum(r["total_bytes"] for r in results.values())
    print("\n=== elevation bake summary ===")
    print(
        json.dumps(
            {
                "face_sizes": face_sizes,
                "source_km_per_px_equator": src_km_per_px,
                "total_bytes_all_tiers_both_projections": total_bytes_all,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
