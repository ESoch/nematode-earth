#!/usr/bin/env python3
"""Bake GHSL built-up surface (fraction, 0..1) into cube maps, same shape
as `bake_cube.py` (density) and `bake_elevation_cube.py` (ETOPO): both
projections, RGBA16Float raw faces + JSON sidecar with `units` and
`km_per_texel` populated, `field`/`projection` naming.

Why this channel matters twice: the SAME built-up fraction drives glow in
the "matter" (city-lights) pass and suppression in the veil (nematode
density) pass -- one wrong number breaks both halves together, which is
a deliberately checkable property, not a coincidence. `impl-realdata`
consumes this cube for the matter pass; this module owns the format so
both consumers read the same asset instead of two incompatible ones.

Encoding: R = built-up fraction, RAW (no log -- unlike density, this
field is naturally bounded [0,1] and mostly near 0, so log-encoding would
just waste precision on the common case). G/B reserved. A = validity
(always 1.0 -- see fetch/ghsl.py: texels beyond GHSL's actual coverage,
a strip near each pole, are treated as u=0 -- genuinely no built-up area
there in reality, not missing data -- so validity stays 1).

Resolution: GHSL's 30" native data resolves real geographic FORM (river
valleys, coastlines, deserts) that is the entire point of this layer per
the task brief, so it gets more resolution than density's 128-face
default -- three tiers, 128/256/512, both projections. See main() for the
exact byte-cost report; the caller decides whether 512 (24 MB combined
across both projections) is worth shipping given it streams rather than
gating first paint.

Validation: built-up area is strictly positive (unlike elevation), so a
plain ratio gate is appropriate here -- total built-up area (m^2)
conserved between the source decimated grid and the baked, quantized,
re-integrated cube.
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).parent.parent / "fetch"))
from ghsl import (
    decimate_built_up_fraction,
    load_ghsl,
    total_built_up_area_m2,
)

PROJECTIONS = ["plain", "eac"]


def bake_ghsl_face(
    face: int,
    size: int,
    frac_grid: np.ndarray,
    grid_lon0: float,
    grid_lat0: float,
    grid_dlon: float,
    grid_dlat: float,
    grid_lat_min: float,
    projection: str,
):
    idx = (np.arange(size) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(idx, idx)
    x, y, z = face_direction(face, u, v, projection)
    norm = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm

    lon_rad, lat_gc_rad = direction_to_lonlat(x, y, z)
    lat_gd_rad = np.vectorize(geocentric_to_geodetic_lat)(lat_gc_rad)
    lon_deg = np.degrees(lon_rad)
    lat_gd_deg = np.degrees(lat_gd_rad)

    # grid_lat_min is the SIGNED southern edge of actual coverage (e.g.
    # -89.1, not a symmetric +/- limit like the density raster's clip),
    # and grid_lat0 is the (also signed, positive) northern edge -- an
    # earlier version compared abs(lat) > grid_lat_min, which is true for
    # EVERY latitude once grid_lat_min is negative, zeroing the entire
    # bake. The actual coverage window is [grid_lat_min, grid_lat0].
    beyond_coverage = (lat_gd_deg > grid_lat0) | (lat_gd_deg < grid_lat_min)
    lat_sample = np.clip(lat_gd_deg, grid_lat_min, grid_lat0)
    frac = bilinear_sample(
        frac_grid, lon_deg, lat_sample, grid_lon0, grid_lat0, grid_dlon, grid_dlat
    )
    frac = np.where(beyond_coverage, 0.0, np.clip(frac, 0.0, 1.0))

    rgba = np.zeros((size, size, 4), dtype=np.float16)
    rgba[..., 0] = frac.astype(np.float16)  # R: built-up fraction, raw
    rgba[..., 1] = np.float16(0.0)  # G: reserved
    rgba[..., 2] = np.float16(0.0)  # B: reserved
    rgba[..., 3] = np.float16(1.0)  # A: validity (always 1; see module docstring)
    return rgba.view(np.uint16), int(beyond_coverage.sum())


def baked_total_built_up_area(
    faces_rgba: list[np.ndarray], face_size: int, projection: str
) -> float:
    solid_angles = texel_solid_angles(face_size, projection)
    total = 0.0
    for rgba_u16 in faces_rgba:
        frac = rgba_u16[..., 0].view(np.float16).astype(np.float64)
        total += float((frac * solid_angles).sum()) * R_MEAN**2
    return total


# Real shipped urban_plain_512's ratio_baked_to_source is 0.9989 -- the
# quantization/reprojection loss here is small (built-up fraction is
# raw-encoded, not log-encoded, so float16 rounding is the dominant error
# source, not a floor/log artifact). +/-10% is generous relative to that
# but still catches the failure modes that matter: a wholesale zeroing
# (ratio -> 0) or a runaway overcount, not a false alarm on ordinary
# quantization noise at a new face size.
RATIO_MIN, RATIO_MAX = 0.9, 1.1

# GHSL's real polar gap is a thin sliver near each pole (0.0066% of
# texels at size=512 -- see the shipped urban_plain_512/cube_meta.json).
# 2% is a loose but real ceiling: comfortably above any legitimate
# coverage-gap size at any face size this pipeline ships, while still
# catching "something ate way more of the globe than just the poles."
BEYOND_COVERAGE_FRACTION_MAX = 0.02


def assert_built_up_area_valid(
    ratio_baked_to_source: float,
    baked_total_area_m2: float,
    n_beyond: int,
    face_size: int,
) -> dict:
    """Hard-fail gate (raises AssertionError, not a printed warning) for
    the baked GHSL built-up-fraction cube. Exists because the bake was
    ONCE ALREADY silently zeroed by a sign-handling bug in
    `bake_ghsl_face` (see its docstring: an earlier version compared
    `abs(lat) > grid_lat_min` against a signed, negative grid_lat_min,
    which is true everywhere once the bound is negative) and at the time
    there was no assertion to catch it -- `ratio_baked_to_source` and
    `texels_beyond_coverage` were written to cube_meta.json for a human
    to notice, never gated. This closes that gap before baking a finer
    tier makes a silent regression more expensive to catch by inspection
    (more texels to eyeball, less chance anyone actually looks).

    Three independent checks, each targeting a different way "silently
    zeroed" could still slip past the others:
      1. `baked_total_area_m2 > 0` -- the most direct signature of a
         wholesale zeroing, and independent of the ratio (a ratio-only
         gate could pass a degenerate 0/0 case).
      2. `n_beyond` is not the whole cube and not an implausibly large
         fraction of it -- the exact shape of the historical bug
         (`abs(lat) > grid_lat_min` true for every texel), generalized
         to also catch a partial version of the same class of mistake.
      3. `ratio_baked_to_source` inside [RATIO_MIN, RATIO_MAX] -- the
         general-purpose mass-conservation check already computed by the
         caller, now actually enforced rather than merely recorded.
    """
    n_texels_total = FACE_COUNT * face_size * face_size
    beyond_fraction = n_beyond / n_texels_total

    assert baked_total_area_m2 > 0, (
        f"baked total built-up area is {baked_total_area_m2} (not positive) -- "
        "this is the signature of a wholesale-zeroing bug, not legitimate data"
    )
    assert n_beyond < n_texels_total, (
        f"ALL {n_texels_total} texels are flagged beyond coverage at face_size="
        f"{face_size} -- this is exactly the historical sign-handling bug "
        "(abs(lat) > grid_lat_min compared against a signed, negative bound) "
        "zeroing the whole bake, not a real GHSL coverage gap"
    )
    assert beyond_fraction <= BEYOND_COVERAGE_FRACTION_MAX, (
        f"{beyond_fraction:.4%} of texels are beyond coverage at face_size="
        f"{face_size}, exceeding the {BEYOND_COVERAGE_FRACTION_MAX:.0%} ceiling -- "
        "GHSL's real polar gap is a thin sliver near each pole; this many "
        "texels excluded means something ate more of the globe than just that"
    )
    assert RATIO_MIN <= ratio_baked_to_source <= RATIO_MAX, (
        f"baked/source built-up-area ratio {ratio_baked_to_source:.4f} at "
        f"face_size={face_size} is outside the accepted [{RATIO_MIN}, {RATIO_MAX}] "
        "band -- mass is not being conserved through the bake"
    )

    return {
        "passed": True,
        "ratio_baked_to_source": ratio_baked_to_source,
        "beyond_coverage_fraction": beyond_fraction,
        "accepted_ratio_range": [RATIO_MIN, RATIO_MAX],
        "accepted_beyond_coverage_fraction_max": BEYOND_COVERAGE_FRACTION_MAX,
    }


def bake_one(
    projection: str,
    size: int,
    frac_grid: np.ndarray,
    grid_lon0: float,
    grid_lat0: float,
    grid_dlon: float,
    grid_dlat: float,
    grid_lat_min: float,
    out_dir: Path,
    source_total_area: float,
) -> dict:
    faces_rgba, total_beyond = [], 0
    for face in range(FACE_COUNT):
        rgba_u16, n_beyond = bake_ghsl_face(
            face,
            size,
            frac_grid,
            grid_lon0,
            grid_lat0,
            grid_dlon,
            grid_dlat,
            grid_lat_min,
            projection,
        )
        faces_rgba.append(rgba_u16)
        total_beyond += n_beyond
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

    baked_total_area = baked_total_built_up_area(faces_rgba, size, projection)
    ratio_baked_to_source = baked_total_area / source_total_area
    km_per_texel = (90.0 / size) * (math.pi / 180.0) * R_MEAN / 1000.0

    # Hard gate -- raises AssertionError (stopping the bake, before any
    # cube_meta.json claiming success is written) if the bake looks like
    # the historical wholesale-zeroing bug or otherwise fails mass
    # conservation. See assert_built_up_area_valid's docstring.
    gate_result = assert_built_up_area_valid(
        ratio_baked_to_source, baked_total_area, total_beyond, size
    )

    meta = {
        "field": "built_up_fraction",
        "units": "fraction (0-1, dimensionless)",
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
            "R": "built-up fraction, raw (0..1, no log encoding -- values are "
            "naturally bounded and mostly near 0)",
            "G": "reserved (0)",
            "B": "reserved (0)",
            "A": "validity: always 1.0 -- texels beyond GHSL's actual polar coverage "
            "are treated as genuinely 0 built-up (real: no cities in Antarctica), "
            "not missing data",
        },
        "coverage": {
            "lat_limit_deg": grid_lat_min,
            "texels_beyond_coverage": total_beyond,
            "note": "GHSL clips a bit short of the poles (tiepoint lat0=89.0996, not "
            "90.0); texels beyond are set to u=0, not extrapolated",
        },
        "source": {
            "url": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss/"
            "V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip",
            "resolution": "30 arc-second (~0.93 km/px equator), EPSG:4326, epoch 2020",
            "source_total_built_up_area_m2": source_total_area,
        },
        "validation": {
            "baked_total_built_up_area_m2": baked_total_area,
            "ratio_baked_to_source": ratio_baked_to_source,
            "wholesale_zeroing_gate": gate_result,
        },
        "bytes_per_face": total_bytes // FACE_COUNT,
        "total_bytes": total_bytes,
    }
    with open(out_dir / "cube_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--face-sizes", type=str, default="128,256,512")
    ap.add_argument("--out-root", type=Path, default=Path("../app/public/data"))
    ap.add_argument(
        "--projections",
        type=str,
        default=",".join(PROJECTIONS),
        help="Comma-separated subset of {plain,eac} to bake at every requested "
        "face size (default: both). Only 'plain' is shipped to app/public/data "
        "today (see bake_cube.py/bake_elevation_cube.py's own plain-only "
        "shipping at their larger tiers) -- pass --projections plain to bake "
        "only what's actually shipped, e.g. for a new, larger tier.",
    )
    args = ap.parse_args()
    face_sizes = [int(s) for s in args.face_sizes.split(",")]
    projections = [p.strip() for p in args.projections.split(",")]
    for p in projections:
        if p not in PROJECTIONS:
            ap.error(
                f"--projections: unknown projection {p!r} (must be one of {PROJECTIONS})"
            )

    print("loading GHSL (full array, ~3.7 GB) ...", file=sys.stderr)
    ghsl_area, g_lon0, g_lat0, g_dlon, g_dlat = load_ghsl()
    ghsl_h, ghsl_w = ghsl_area.shape
    grid_lat_min = g_lat0 - ghsl_h * g_dlat  # southern edge of actual coverage
    src_km_per_px = g_dlon * (math.pi / 180.0) * R_MEAN / 1000.0
    print(
        f"GHSL native: {ghsl_w}x{ghsl_h}, {src_km_per_px:.3f} km/px equator, "
        f"coverage +/-{min(g_lat0, -grid_lat_min):.2f} deg",
        file=sys.stderr,
    )

    results = {}
    total_bytes_all = 0
    for size in face_sizes:
        km_per_texel = (90.0 / size) * (math.pi / 180.0) * R_MEAN / 1000.0
        if km_per_texel < src_km_per_px:
            print(
                f"WARNING: face_size={size} ({km_per_texel:.2f} km/texel) is finer "
                f"than the source's {src_km_per_px:.2f} km/px.",
                file=sys.stderr,
            )

        inter_h = max(128, min(ghsl_h, size * 3))
        inter_w = inter_h * 2
        frac_grid = decimate_built_up_fraction(
            ghsl_area, g_lon0, g_lat0, g_dlon, g_dlat, inter_h, inter_w
        )
        grid_dlon = 360.0 / inter_w
        # The intermediate grid spans exactly the source's own actual row
        # extent (g_lat0 down to grid_lat_min), partitioned evenly across
        # inter_h rows -- decimate_built_up_fraction does this partition
        # via linspace(0, in_h, out_h+1), so the output grid's own dlat is
        # the source's total lat span divided by inter_h, not the
        # source's native dlat.
        grid_lat0 = g_lat0
        grid_dlat = (g_lat0 - grid_lat_min) / inter_h
        source_total_area = total_built_up_area_m2(
            frac_grid, grid_lat0, grid_dlat * inter_h, 360.0
        )
        print(
            f"size={size}: intermediate grid {inter_w}x{inter_h}, "
            f"source_total_area={source_total_area:.4e} m^2",
            file=sys.stderr,
        )

        for projection in projections:
            out_dir = args.out_root / f"urban_{projection}_{size}"
            meta = bake_one(
                projection,
                size,
                frac_grid,
                grid_lon0=g_lon0,
                grid_lat0=grid_lat0,
                grid_dlon=grid_dlon,
                grid_dlat=grid_dlat,
                grid_lat_min=grid_lat_min,
                out_dir=out_dir,
                source_total_area=source_total_area,
            )
            results[f"{projection}_{size}"] = meta
            total_bytes_all += meta["total_bytes"]

    print("\n=== GHSL urban cube bake summary ===")
    print(
        json.dumps(
            {
                "face_sizes": face_sizes,
                "source_km_per_px_equator": src_km_per_px,
                "total_bytes_all_tiers_both_projections": total_bytes_all,
                "results": {
                    k: {
                        "ratio_baked_to_source": v["validation"][
                            "ratio_baked_to_source"
                        ],
                        "bytes_per_face": v["bytes_per_face"],
                        "total_bytes": v["total_bytes"],
                        "km_per_texel": v["km_per_texel"],
                    }
                    for k, v in results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
