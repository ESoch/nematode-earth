#!/usr/bin/env python3
"""Bake `pipeline/out/total_persqm_density_512x256.npy` (+ land-fraction
companion) into cube maps the app can load directly as `THREE.CubeTexture`'s
six `DataTexture` faces.

Bakes BOTH projections from the same source, identical in every other
respect, per the team-lead direction to measure the EAC warp's runtime cost
against its quality gain rather than assume it's worth it (spec S4.4 prices
the warp at 30-60 ALU/sample x 128 samples/pixel = ~4ms, against a plain
N=2048 cube matching EAC's worst-case texel at N~1450 for zero warp ALU):

  - PLAIN (gnomonic): `app/src/field/bakeCube.ts`'s existing projection.
  - EAC (equi-angular): the arctan warp, flattens texel solid-angle
    variation from 5.196:1 to (in principle) exactly sqrt(2):1.

Output: `app/public/data/cube_{plain,eac}_<size>/face_{0..5}.bin` (raw
RGBA16Float, little-endian uint16, row-major,
`(px + py*size)*4 + channel`) plus a `cube_meta.json` per projection
documenting the layout, WITH a `projection` field naming which is which.

Usage: uv run build/bake_cube.py [--face-size 128]
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
    direction_to_lonlat,
    face_direction,
    texel_solid_angles,
)
from raster import bilinear_sample
from wgs84 import R_MEAN, geocentric_to_geodetic_lat

PUBLISHED_TOTAL = 4.4e20
PROJECTIONS = ["plain", "eac"]

# Natural-log encoding, matching `bakeCube.ts::encodeLogTau` exactly
# (`Math.log` / `Math.exp`). The synthetic field is always positive so the
# TS code never needed a floor; our real density is exactly 0 over ocean
# (see pipeline-report.md's NaN->0 convention), and log(0) = -inf would
# poison bilinear filtering of a "filterable everywhere" R16Float texture
# (any weighted blend including -inf is -inf) -- the exact same class of
# bug as raw NaN reaching the shader. DENSITY_FLOOR sidesteps it: it is far
# below the real data's observed minimum (~3.3e4 even on this coarse,
# heavily-averaged grid; the true per-pixel minimum is lower still, but
# still many orders of magnitude above 1.0), so log(1.0)=0.0 decodes to a
# value indistinguishable from "no nematodes" for any land pixel that
# actually has data, while remaining finite and filter-safe.
DENSITY_FLOOR = 1.0


def bake_face(
    face: int,
    size: int,
    density_grid: np.ndarray,
    land_grid: np.ndarray,
    lon0: float,
    lat0: float,
    dlon: float,
    dlat: float,
    polar_limit_deg: float,
    projection: str,
    g_grid: np.ndarray | None = None,
    b_grid: np.ndarray | None = None,
):
    """Returns (rgba_uint16 [size,size,4], n_polar_gap_texels).

    `g_grid` (structure fraction) and `b_grid` (bootstrap CV, already
    un-diluted -- see uncertainty.land_mean_from_diluted) are OPTIONAL and
    default to None, preserving the original reserved-zero behaviour bit
    for bit when omitted (see test_bake.py, which calls this function
    without them and must keep passing unmodified). Both are expected on
    the SAME (lon0, lat0, dlon, dlat) grid as density_grid/land_grid --
    true for the real data because the group rasters share R's exact
    43201x21121 tiepoint, and the bootstrap CV raster's 43200x21120 native
    grid, resampled via the same box-averaging extract_grid already uses
    for R, lands on the identical (lon0=-180, lat0=88, dlon=360/out_w,
    dlat=176/out_h) mapping -- see rebake_gb.py's docstring for why no
    separate coordinate params are needed here."""
    idx = (np.arange(size) + 0.5) / size * 2 - 1  # texel centers, matches bakeCube.ts
    u, v = np.meshgrid(idx, idx)  # v varies down rows (py), u across columns (px)
    x, y, z = face_direction(face, u, v, projection)
    norm = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm

    lon_rad, lat_gc_rad = direction_to_lonlat(x, y, z)
    # Geocentric -> geodetic BEFORE sampling the source raster: the raster's
    # rows are geodetic latitude bands (EPSG:4326), but a cube-map direction
    # vector is inherently geocentric (asin(z) has no ellipsoid concept).
    # Skipping this step displaces content by up to 21.4 km at 45.1 deg lat
    # -- see wgs84.py / the design spec section 4.4.
    lat_gd_rad = np.vectorize(geocentric_to_geodetic_lat)(lat_gc_rad)
    lon_deg = np.degrees(lon_rad)
    lat_gd_deg = np.degrees(lat_gd_rad)

    # No source data exists beyond +/-88 deg geodetic (the raster is
    # deliberately clipped there, not at +/-90). Rather than extrapolating,
    # texels whose geodetic latitude falls in that polar gap are explicit
    # no-data: density=0, land_fraction=0. This keeps the bake honest about
    # what the source actually claims, and keeps the mass-conservation
    # check clean (no invented area contributing invented mass).
    polar_gap = np.abs(lat_gd_deg) > polar_limit_deg
    lat_sample = np.clip(lat_gd_deg, -polar_limit_deg, polar_limit_deg)

    density = bilinear_sample(density_grid, lon_deg, lat_sample, lon0, lat0, dlon, dlat)
    land = bilinear_sample(land_grid, lon_deg, lat_sample, lon0, lat0, dlon, dlat)
    density = np.where(polar_gap, 0.0, np.maximum(density, 0.0))
    land = np.where(polar_gap, 0.0, np.clip(land, 0.0, 1.0))

    log_density = np.log(np.maximum(density, DENSITY_FLOOR)).astype(np.float32)
    land = land.astype(np.float32)

    # G/B: sampled through the identical bilinear_sample + polar-gap path
    # as R/A above, so a texel's G/B always comes from the same (lon,lat)
    # as its R/A -- pixel-for-pixel registration falls out of reusing the
    # same lon_deg/lat_sample/polar_gap rather than being asserted after
    # the fact. Ocean/no-data (g_grid or b_grid == 0 there, by
    # construction -- see functional_group.structure_fraction and
    # uncertainty.land_mean_from_diluted) floors to 0, same convention as
    # A. When the caller omits g_grid/b_grid this reduces to the exact
    # original reserved-zero constant (see bake_face's docstring).
    if g_grid is None:
        structure_fraction = np.zeros((size, size), dtype=np.float32)
    else:
        g = bilinear_sample(g_grid, lon_deg, lat_sample, lon0, lat0, dlon, dlat)
        structure_fraction = np.where(polar_gap, 0.0, np.clip(g, 0.0, 1.0)).astype(
            np.float32
        )
    if b_grid is None:
        bootstrap_cv = np.zeros((size, size), dtype=np.float32)
    else:
        b = bilinear_sample(b_grid, lon_deg, lat_sample, lon0, lat0, dlon, dlat)
        bootstrap_cv = np.where(polar_gap, 0.0, np.maximum(b, 0.0)).astype(np.float32)

    rgba = np.zeros((size, size, 4), dtype=np.float16)
    rgba[..., 0] = log_density.astype(np.float16)  # R: log density
    rgba[..., 1] = structure_fraction.astype(np.float16)  # G: structure fraction
    rgba[..., 2] = bootstrap_cv.astype(np.float16)  # B: bootstrap CV
    rgba[..., 3] = land.astype(np.float16)  # A: land fraction / validity

    return rgba.view(np.uint16), int(polar_gap.sum())


def validate_bake(
    faces_rgba: list[np.ndarray], face_size: int, projection: str
) -> dict:
    """Re-integrate the baked (quantized, decoded) cube over the sphere and
    compare to the published total. Uses the EXACT per-texel solid angle
    for `projection` (see eac.rect_solid_angle), not an approximation, and
    decodes from the ACTUAL stored half-float bit pattern -- so this
    catches quantization loss as well as reprojection loss.

    Also reports the exact "floor contamination": every texel that decodes
    to exactly DENSITY_FLOOR is one where the pre-encode density was 0
    (ocean, within-domain) or the texel fell in the polar no-data gap --
    log(DENSITY_FLOOR) and its float16 round-trip are both exact, so this
    detection is exact, not a mask threaded through from bake_face. Ocean
    covers most of the sphere, so this is the dominant source of any
    baked-vs-source integral drift, not the polar gap (which is a tiny
    area near +/-90 deg)."""
    solid_angles = texel_solid_angles(face_size, projection)  # same for every face
    total = 0.0
    contamination = 0.0
    n_floored = 0
    n_floored_with_land = 0  # would indicate a bug: floored=no-data but land>0
    max_land_among_floored = 0.0
    for rgba_u16 in faces_rgba:
        r_u16 = rgba_u16[..., 0]
        a_u16 = rgba_u16[..., 3]
        log_density = r_u16.view(np.float16).astype(np.float64)
        land = a_u16.view(np.float16).astype(np.float64)
        density = np.exp(log_density)
        total += float((density * solid_angles).sum()) * R_MEAN**2
        floored = density == DENSITY_FLOOR
        n_floored += int(floored.sum())
        contamination += (
            float((DENSITY_FLOOR * solid_angles[floored]).sum()) * R_MEAN**2
        )
        n_floored_with_land += int((floored & (land > 0)).sum())
        if floored.any():
            max_land_among_floored = max(
                max_land_among_floored, float(land[floored].max())
            )
    return {
        "integral": total,
        "floor_contamination": contamination,
        "n_floored_texels": n_floored,
        # Audit: every floored texel MUST have land_fraction==0 (ocean,
        # within-domain, or the polar no-data gap -- see bake_face). If this
        # were ever nonzero it would mean real land got clamped to the
        # floor, not ocean -- a genuine bug, not the expected 70%-ish figure.
        "n_floored_with_land_fraction_gt_0": n_floored_with_land,
        "max_land_fraction_among_floored_texels": max_land_among_floored,
    }


def ground_texel_stats(face_size: int, projection: str) -> dict:
    """Worst-case and RMS ground texel size (km), plus the raw solid-angle
    ratio max/min -- the number the EAC-vs-plain trade is actually argued
    over (spec claims EAC flattens 5.196:1 down to exactly sqrt(2):1).
    "Ground texel size" here is the equivalent-square-side length
    sqrt(solid_angle)*R, a standard linear stand-in for an (irregular,
    non-square) texel's angular footprint."""
    sa = texel_solid_angles(face_size, projection)
    ground_km = np.sqrt(sa) * R_MEAN / 1000.0
    return {
        "worst_case_km": float(ground_km.max()),
        "best_case_km": float(ground_km.min()),
        "rms_km": float(np.sqrt((ground_km**2).mean())),
        "solid_angle_ratio_max_over_min": float(sa.max() / sa.min()),
    }


def bake_projection(
    projection: str,
    size: int,
    density_grid: np.ndarray,
    land_grid: np.ndarray,
    lon0: float,
    lat0: float,
    dlon: float,
    dlat: float,
    polar_limit_deg: float,
    out_dir: Path,
    source_meta: dict,
) -> dict:
    faces_rgba = []
    total_polar_gap = 0
    for face in range(FACE_COUNT):
        rgba_u16, n_gap = bake_face(
            face,
            size,
            density_grid,
            land_grid,
            lon0,
            lat0,
            dlon,
            dlat,
            polar_limit_deg,
            projection,
        )
        faces_rgba.append(rgba_u16)
        total_polar_gap += n_gap
        print(
            f"  [{projection}] face {face} ({FACE_NAMES[face]}): {n_gap} polar-gap texels",
            file=sys.stderr,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for face, rgba_u16 in enumerate(faces_rgba):
        raw = rgba_u16.astype("<u2").tobytes()
        (out_dir / f"face_{face}.bin").write_bytes(raw)
        total_bytes += len(raw)

    val = validate_bake(faces_rgba, size, projection)
    stats = ground_texel_stats(size, projection)
    source_integral = source_meta["area_weighted_integral"]

    projection_doc = {
        "plain": "plain gnomonic: raw linear NDC (u,v) fed directly into the "
        "standard cube face embedding, matching app/src/field/bakeCube.ts's "
        "EXISTING (already-shipped) bake exactly",
        "eac": "equi-angular cubemap (EAC): u,v pre-warped via tan(t*pi/4) before "
        "the same standard cube face embedding",
    }[projection]

    meta = {
        "projection": projection,
        "projection_description": projection_doc,
        "format": "RGBA16Float (uint16 little-endian bit patterns), row-major, "
        "index = (px + py*face_size)*4 + channel",
        "face_size": size,
        "face_order": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        "face_convention": "matches THREE.CubeTexture images order; texel (px,py) "
        "-> u,v=((idx+0.5)/size)*2-1 in [-1,1]; row 0 = v=-1 "
        "(flipY=false, matches app/src/field/bakeCube.ts)",
        "geodetic_to_geocentric": {
            "applied": True,
            "wgs84_flattening": 1 / 298.257223563,
            "note": "cube direction (geocentric) converted to geodetic latitude "
            "before sampling the EPSG:4326 source raster; see build/wgs84.py",
        },
        "channels": {
            "R": "natural-log density: log(max(density_per_sqm, floor)); "
            f"floor={DENSITY_FLOOR}; decode with exp(R)",
            "G": "reserved (0) -- functional-group channel, not yet populated",
            "B": "reserved (0) -- provenance channel, not yet populated",
            "A": "land fraction (0..1), unmodified from source extraction",
        },
        "polar_gap": {
            "limit_deg": polar_limit_deg,
            "note": "no source data beyond +/-88 deg geodetic; texels landing there "
            "are explicit no-data (R decodes to floor, A=0), not extrapolated",
            "total_texels_affected": total_polar_gap,
        },
        "texel_ground_stats": stats,
        "source": {
            "density_npy": "out/total_persqm_density_512x256.npy",
            "landfrac_npy": "out/total_persqm_landfrac_512x256.npy",
            "source_row_coverage_actual": source_meta["row_coverage_actual"],
            "source_area_weighted_integral": source_integral,
        },
        "validation": {
            "baked_cube_integral": val["integral"],
            "published_total": PUBLISHED_TOTAL,
            "ratio_baked_to_source": val["integral"] / source_integral,
            "ratio_baked_to_published": val["integral"] / PUBLISHED_TOTAL,
            "floor_contamination_exact": val["floor_contamination"],
            "floor_contamination_fraction_of_published": val["floor_contamination"]
            / PUBLISHED_TOTAL,
            "n_floored_texels": val["n_floored_texels"],
            "n_texels_total": FACE_COUNT * size * size,
            "floored_texel_audit": {
                "note": "~70% of texels decode to the DENSITY_FLOOR placeholder "
                "(log(1.0)) rather than a real measurement. That is expected, "
                "not a bug: Earth is ~70% ocean, the source raster is NaN "
                "there (converted to density=0 at extraction time), and this "
                "bake floors density=0 to avoid storing log(0)=-inf. Confirmed "
                "below by cross-checking the A (land_fraction) channel: every "
                "floored texel has land_fraction exactly 0 -- if this were "
                "ever >0 it would mean real land got clamped to the floor, "
                "which would be a genuine bug.",
                "n_floored_with_land_fraction_gt_0": val[
                    "n_floored_with_land_fraction_gt_0"
                ],
                "max_land_fraction_among_floored_texels": val[
                    "max_land_fraction_among_floored_texels"
                ],
                "confirmed_ocean_or_polar_gap_only": val[
                    "n_floored_with_land_fraction_gt_0"
                ]
                == 0,
            },
        },
        "bytes_per_face": total_bytes // FACE_COUNT,
        "total_bytes": total_bytes,
    }
    with open(out_dir / "cube_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--face-size", type=int, default=128)
    ap.add_argument(
        "--density-npy", type=Path, default=Path("out/total_persqm_density_512x256.npy")
    )
    ap.add_argument(
        "--landfrac-npy",
        type=Path,
        default=Path("out/total_persqm_landfrac_512x256.npy"),
    )
    ap.add_argument(
        "--source-meta", type=Path, default=Path("out/total_persqm_meta_512x256.json")
    )
    ap.add_argument("--out-root", type=Path, default=Path("../app/public/data"))
    args = ap.parse_args()

    density_grid = np.load(args.density_npy).astype(np.float64)
    land_grid = np.load(args.landfrac_npy).astype(np.float64)
    with open(args.source_meta) as f:
        source_meta = json.load(f)

    out_h, out_w = density_grid.shape
    assert (out_h, out_w) == (source_meta["out_height"], source_meta["out_width"])
    lat0 = source_meta["tiff"]["model_tiepoint"][4]  # 88.0
    lon0 = source_meta["tiff"]["model_tiepoint"][3]  # -180.0
    dlon = 360.0 / out_w
    dlat = (2 * lat0) / out_h  # 176 / out_h
    polar_limit_deg = lat0  # data claims nothing beyond this

    size = args.face_size
    src_deg_per_px = 360.0 / out_w
    src_km_per_px = src_deg_per_px * (math.pi / 180.0) * R_MEAN / 1000.0
    if (90.0 / size) * (math.pi / 180.0) * R_MEAN / 1000.0 < src_km_per_px * 0.9:
        print(
            f"WARNING: face_size={size} is finer than the source's {src_km_per_px:.1f} "
            f"km/px -- this would invent detail. Reduce --face-size or extract at "
            f"higher resolution first.",
            file=sys.stderr,
        )

    results = {}
    for projection in PROJECTIONS:
        out_dir = args.out_root / f"cube_{projection}_{size}"
        results[projection] = bake_projection(
            projection,
            size,
            density_grid,
            land_grid,
            lon0,
            lat0,
            dlon,
            dlat,
            polar_limit_deg,
            out_dir,
            source_meta,
        )

    print("\n=== bake summary: both projections ===")
    print(
        json.dumps(
            {
                "face_size": size,
                "source_km_per_px_equator": src_km_per_px,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
