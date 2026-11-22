#!/usr/bin/env python3
"""Fill `cube_plain_128`'s G (structure fraction) and B (bootstrap CV)
channels from real ETH Research Collection rasters, per the team-lead's
binding channel ruling.

This is a ONE-SHOT, targeted rebake -- not a re-run of `bake_cube.py`'s
full dual-projection pipeline. It touches only `app/public/data/cube_plain_128`
(the only cube shipped there; `cube_eac_128` was never baked to app/public
by an earlier session and is out of scope here), reusing `bake_face`,
`validate_bake`, and `ground_texel_stats` from `bake_cube.py` UNCHANGED --
R and A are computed by the exact same code path as before, and this
script's own hard-fail gate proves their bytes did not move (see
`verify_r_a_byte_identical`).

Inputs (already fetched to `pipeline/out/` by `fetch_g_b_sources.sh`, via
the existing, unmodified `fetch/eth_raster.py extract` CLI -- same
2.42% stratified row-coverage, full column-coverage approach as the
already-shipped R/A extraction):

  - doi:10.3929/ethz-b-000354394 (item 6f6b1997-f1ba-4c85-8fc6-a7d68dc71234):
    Bacterivores/Fungivores/Herbivores/Omnivores/Predators_per100gSoil.tif
    -- five independently-modelled functional-group rasters, 43201x21121,
    float32, SAME tiepoint/pixel-scale as the density raster already baked.
  - doi:10.3929/ethz-b-000354035 (item 73e5c336-f3bd-46a7-990e-150ac581bc28,
    the record `total_persqm`'s source raster also lives in): also fetched
    `TotalNumber_per100g.tif` (to check the summation trap in the SAME
    units, no per-m2 conversion factor needed) and
    `TotalNumber_Bootstrap_CoefVar.tif` -- 43200x21120, float64, SAME
    tiepoint/pixel-scale as the main grid but ONE FEWER native row/column.
    Deliberately NOT `TotalNumber_Ensemble_CoefVar.tif` -- see the ruling.

**Why no separate lon0/lat0/dlon/dlat is needed for the bootstrap grid.**
Both real headers were verified live (`eth_raster.py verify`): identical
ModelTiepointTag (lon0=-180, lat0=88) and identical ModelPixelScaleTag
(1/120 deg) on both the 43201x21121 density grid and the 43200x21120
bootstrap grid. `eth_raster.extract_grid` partitions a raster's OWN native
pixel-index range into `out_width`/`out_height` bands, using that
raster's own `info.width`/`info.height` -- so calling it separately on
each file (which `fetch_g_b_sources.sh` already does) produces a 512x256
grid correctly registered to the SAME (lon0=-180, lat0=88, dlon=360/512,
dlat=176/256) physical mapping bake_cube.py already uses for R/A, with NO
extra coordinate bookkeeping required here. This is exact for the
bootstrap grid (43200 cols * 1/120 deg/col = 360 deg exactly, 21120 rows *
1/120 = 176 deg exactly -- no leftover fraction of a pixel), whereas it is
a already-accepted tiny (~0.008 deg, one native pixel) approximation for
the shipped 43201x21121 density grid -- see bake_cube.py's main(). Both
land on the same output mapping either way. The measured effect of
extracting each raster independently at only 2.42% row coverage is a
land_frac cross-check mean abs diff of 1.6e-5 and max 0.37 in 3 of 131,072
cells (all coastal, irregular-coastline rows where the two independently-
stratified samples picked different native rows) -- see
`cross_check_land_frac` below.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from bake_cube import DENSITY_FLOOR, bake_face, ground_texel_stats, validate_bake
from eac import FACE_COUNT, FACE_NAMES
from functional_group import (
    GROUP_NAMES,
    global_structure_fraction,
    global_summation_ratio,
    group_sum,
    structure_fraction,
    summation_ratio,
)
from uncertainty import assert_bootstrap_cv_valid, land_mean_from_diluted
from wgs84 import R_MEAN

OUT_DIR = Path(__file__).parent.parent / "out"
CUBE_DIR = (
    Path(__file__).parent.parent.parent / "app" / "public" / "data" / "cube_plain_128"
)
FACE_SIZE = 128
PROJECTION = "plain"

PUBLISHED_STRUCTURE_FRACTION = 0.134  # Table 1: 8.9% omnivores + 4.5% predators
STRUCTURE_FRACTION_REL_TOL = 0.10  # 10% relative -- see rebake report for why

# Provenance: bitstream URLs actually fetched (resolved live via the DSpace
# REST API against the two DOIs -- see module docstring).
GROUP_SOURCE = {
    "doi": "10.3929/ethz-b-000354394",
    "item_uuid": "6f6b1997-f1ba-4c85-8fc6-a7d68dc71234",
    "files": {
        "bacterivores": "6394a9df-fe27-4bef-b712-f4f3aeb2189f",
        "fungivores": "2eb513f5-dbd5-4dd8-becd-ae0699e6840a",
        "herbivores": "c4de346b-d8ae-4b02-a955-52ddb0222f4e",
        "omnivores": "ed1a9f1e-1e45-4224-8969-3236fa1b3365",
        "predators": "62f5d623-90a7-43dc-a87f-c84f89fb9210",
    },
}
BOOTSTRAP_SOURCE = {
    "doi": "10.3929/ethz-b-000354035",
    "item_uuid": "73e5c336-f3bd-46a7-990e-150ac581bc28",
    "bitstream_uuid": "137ecc0c-2673-40fd-a9db-a4b4839bde33",
    "file": "TotalNumber_Bootstrap_CoefVar.tif",
}
TOTAL_PER100G_SOURCE = {
    "doi": "10.3929/ethz-b-000354035",
    "item_uuid": "73e5c336-f3bd-46a7-990e-150ac581bc28",
    "bitstream_uuid": "9c6f6a0d-a7cd-4261-aa1c-48924fb53374",
    "file": "TotalNumber_per100g.tif",
}
NOT_USED_ENSEMBLE_COEFVAR = {
    "bitstream_uuid": "106718a3-d351-4907-a22c-405de8fa19e3",
    "file": "TotalNumber_Ensemble_CoefVar.tif",
    "reason": "seed-to-seed jitter of a single model run, not real bootstrap "
    "uncertainty -- shares the convenient 43201x21121 grid but would read as "
    "'confidence' while measuring nothing of the kind. See the channel ruling.",
}


def _load(tag: str) -> tuple[np.ndarray, np.ndarray, dict]:
    density = np.load(OUT_DIR / f"{tag}_density_512x256.npy").astype(np.float64)
    land = np.load(OUT_DIR / f"{tag}_landfrac_512x256.npy").astype(np.float64)
    with open(OUT_DIR / f"{tag}_meta_512x256.json") as f:
        meta = json.load(f)
    return density, land, meta


def row_area_weights(out_h: int, out_w: int, lat0: float) -> np.ndarray:
    """Exact spherical lat-band area per output cell (m^2), matching
    eth_raster.area_weighted_integral's formula, broadcast to the full
    grid shape (equirect columns are equal-area within a row)."""
    lat_edges = lat0 - np.linspace(0, 2 * lat0, out_h + 1)
    band_area = (
        R_MEAN**2
        * math.radians(360.0)
        * (np.sin(np.radians(lat_edges[:-1])) - np.sin(np.radians(lat_edges[1:])))
    )
    row_area = band_area / out_w
    return np.broadcast_to(row_area[:, None], (out_h, out_w)).copy()


def cross_check_land_frac(land_a: np.ndarray, land_b: np.ndarray, label: str) -> dict:
    """Diagnostic (not itself a hard gate for the group files, since their
    land_frac is EXPECTED to be bit-identical to R's -- same 43201x21121
    native grid, same deterministic row-selection algorithm; that identity
    IS asserted as a hard gate elsewhere). For the bootstrap grid (a
    genuinely different native size), this quantifies how much the
    independent 2.42% stratified sampling on a differently-sized grid
    moved the coastline -- see module docstring."""
    diff = np.abs(land_a - land_b)
    return {
        "label": label,
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "n_cells_diff_gt_0.05": int((diff > 0.05).sum()),
        "n_cells_diff_gt_0.3": int((diff > 0.3).sum()),
        "total_cells": int(diff.size),
    }


def verify_r_a_byte_identical(new_faces: list[np.ndarray]) -> dict:
    """Load the CURRENTLY-COMMITTED face_N.bin bytes straight from git
    (not from the working tree, in case this script already overwrote
    them) and compare their R (channel 0) and A (channel 3) uint16 arrays
    against the freshly baked ones. Hard-fails if either differs by even
    one bit -- this is the literal 'verify with git diff, not by
    re-deriving' the task requires."""
    results = {}
    all_match = True
    for face in range(FACE_COUNT):
        rel = f"app/public/data/cube_plain_128/face_{face}.bin"
        old_bytes = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=Path(__file__).parent.parent.parent,
            check=True,
            capture_output=True,
        ).stdout
        old = np.frombuffer(old_bytes, dtype="<u2").reshape(FACE_SIZE, FACE_SIZE, 4)
        new = new_faces[face]
        r_match = bool(np.array_equal(old[..., 0], new[..., 0]))
        a_match = bool(np.array_equal(old[..., 3], new[..., 3]))
        results[f"face_{face}"] = {
            "R_byte_identical": r_match,
            "A_byte_identical": a_match,
        }
        all_match = all_match and r_match and a_match
    results["all_faces_r_and_a_byte_identical"] = all_match
    assert all_match, f"R and/or A channel changed vs the committed bake: {results}"
    return results


def main():
    # --- Load the five functional-group grids + TotalNumber_per100g -----
    groups = {}
    for name in GROUP_NAMES:
        d, land, meta = _load(name)
        groups[name] = {"density": d, "land": land, "meta": meta}

    # Hard requirement, not a diagnostic: the five group files + the total
    # file share the EXACT same 43201x21121 native grid and the same
    # deterministic (info.height, out_height, row_coverage) row-selection
    # as R's own total_persqm extraction, so their land_frac grids MUST be
    # bit-identical to each other and to R's -- any difference would mean
    # a mismatched out_width/out_height/row_coverage argument snuck in.
    total_d, total_land, total_meta = _load("totalnumber_per100g")
    r_density, r_land, r_meta = _load("total_persqm")
    reference_land = groups["bacterivores"]["land"]
    all_land_grids = [groups[n]["land"] for n in GROUP_NAMES] + [total_land]
    for g in all_land_grids:
        assert np.array_equal(g, reference_land), (
            "group/total-per100g land_frac grids must be bit-identical "
            "(same native grid, same deterministic row selection) -- one "
            "of the extractions used a different out_width/out_height/"
            "row_coverage than the others"
        )
    assert np.array_equal(reference_land, r_land), (
        "group land_frac must be bit-identical to R's (total_persqm) "
        "land_frac -- same 43201x21121 native grid and extraction params"
    )

    sum_five = group_sum(
        groups["bacterivores"]["density"],
        groups["fungivores"]["density"],
        groups["herbivores"]["density"],
        groups["omnivores"]["density"],
        groups["predators"]["density"],
    )
    g_grid, n_clipped, max_clip_extent = structure_fraction(
        groups["omnivores"]["density"], groups["predators"]["density"], sum_five
    )
    assert np.all((g_grid >= 0.0) & (g_grid <= 1.0)), "G escaped [0,1] after clipping"

    out_h, out_w = g_grid.shape
    lat0 = r_meta["tiff"]["model_tiepoint"][4]
    area = row_area_weights(out_h, out_w, lat0)

    global_g = global_structure_fraction(
        groups["omnivores"]["density"], groups["predators"]["density"], sum_five, area
    )
    rel_err = (
        abs(global_g - PUBLISHED_STRUCTURE_FRACTION) / PUBLISHED_STRUCTURE_FRACTION
    )
    assert rel_err < STRUCTURE_FRACTION_REL_TOL, (
        f"global structure fraction {global_g:.6f} ({global_g * 100:.2f}%) is "
        f"{rel_err * 100:.1f}% relative off Table 1's published 13.4% -- exceeds "
        f"the {STRUCTURE_FRACTION_REL_TOL * 100:.0f}% hard-fail tolerance"
    )

    # The summation trap: does sum_five reconcile to the independently
    # modelled TotalNumber_per100g? (It does not -- see report.)
    per_pixel_ratio = summation_ratio(sum_five, total_d)
    land_ratio_mask = total_d > 0
    global_sum_ratio = global_summation_ratio(sum_five, total_d, area)

    # --- Bootstrap CV --------------------------------------------------
    boot_raw, boot_land, boot_meta = _load("bootstrap_coefvar")
    b_grid = land_mean_from_diluted(boot_raw, boot_land)
    b_stats = assert_bootstrap_cv_valid(b_grid, boot_land, boot_raw)
    land_frac_cross_check = cross_check_land_frac(boot_land, r_land, "bootstrap_vs_R")
    assert land_frac_cross_check["mean_abs_diff"] < 0.02, (
        "bootstrap CV land_frac diverges from R's land_frac by more than "
        "expected sampling noise on average -- likely a grid-registration bug"
    )
    assert land_frac_cross_check["max_abs_diff"] < 0.5, (
        "a single cell's bootstrap-vs-R land_frac diverged by more than a "
        "plausible coastal-sampling-noise amount -- check for an off-by-one "
        "row/column between the two extractions"
    )

    # --- Rebake app/public/data/cube_plain_128 --------------------------
    dlon = 360.0 / out_w
    dlat = (2 * lat0) / out_h
    lon0 = r_meta["tiff"]["model_tiepoint"][3]
    polar_limit_deg = lat0

    faces_rgba = []
    total_polar_gap = 0
    for face in range(FACE_COUNT):
        rgba_u16, n_gap = bake_face(
            face,
            FACE_SIZE,
            r_density,
            r_land,
            lon0,
            lat0,
            dlon,
            dlat,
            polar_limit_deg,
            PROJECTION,
            g_grid=g_grid,
            b_grid=b_grid,
        )
        faces_rgba.append(rgba_u16)
        total_polar_gap += n_gap
        print(
            f"  face {face} ({FACE_NAMES[face]}): {n_gap} polar-gap texels",
            file=sys.stderr,
        )

    # Verify R/A unchanged BEFORE overwriting the working-tree files (the
    # comparison target is `git show HEAD:...`, unaffected either way, but
    # doing this first means a failure leaves the shipped files untouched).
    byte_check = verify_r_a_byte_identical(faces_rgba)

    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for face, rgba_u16 in enumerate(faces_rgba):
        raw = rgba_u16.astype("<u2").tobytes()
        (CUBE_DIR / f"face_{face}.bin").write_bytes(raw)
        total_bytes += len(raw)

    val = validate_bake(faces_rgba, FACE_SIZE, PROJECTION)
    stats = ground_texel_stats(FACE_SIZE, PROJECTION)

    with open(CUBE_DIR / "cube_meta.json") as f:
        meta = json.load(f)

    meta["channels"] = {
        "R": "natural-log density: log(max(density_per_sqm, floor)); "
        f"floor={DENSITY_FLOOR}; decode with exp(R). Unit: nematodes/m^2. "
        "Unchanged by this update -- see validation.r_a_byte_identical.",
        "G": "structure fraction: (omnivores + predators) / (sum of the five "
        "functional-group per100gSoil grids), in [0,1]. Denominator is the SUM "
        "OF THE FIVE GROUPS, not TotalNumber -- the two do NOT reconcile on "
        "real data (see validation.summation_trap): the five groups were "
        "modelled independently of TotalNumber_per100g.tif and sum to it with "
        "a global area-weighted ratio of "
        f"{global_sum_ratio:.4f} (per-pixel ratio ranges "
        f"{float(np.nanmin(per_pixel_ratio[land_ratio_mask])):.3f} to "
        f"{float(np.nanmax(per_pixel_ratio[land_ratio_mask])):.3f}). Normalising "
        "by TotalNumber instead would silently encode that model disagreement "
        "as ecology -- see the channel ruling. Provenance: ETH Research "
        f"Collection doi:{GROUP_SOURCE['doi']} (item {GROUP_SOURCE['item_uuid']}), "
        f"files {', '.join(f'{n}_per100gSoil.tif' for n in GROUP_NAMES)}.",
        "B": "bootstrap coefficient of variation (per-pixel model uncertainty), "
        "un-diluted to the land-only mean (see uncertainty.land_mean_from_diluted "
        "-- extract_grid's box average treats ocean as 0 for mass conservation, "
        "which is correct for a density but would silently understate "
        "uncertainty at coastal cells for a per-soil-sample statistic like CV). "
        "Dimensionless (std/mean of the 100-iteration bootstrap distribution). "
        f"Deliberately NOT {NOT_USED_ENSEMBLE_COEFVAR['file']} -- "
        f"{NOT_USED_ENSEMBLE_COEFVAR['reason']} Provenance: ETH Research "
        f"Collection doi:{BOOTSTRAP_SOURCE['doi']} (item {BOOTSTRAP_SOURCE['item_uuid']}), "
        f"file {BOOTSTRAP_SOURCE['file']}, on a 43200x21120 native grid (one "
        "fewer row/column than the main 43201x21121 density grid, same "
        "tiepoint/pixel-scale) -- resampled via the same box-averaging "
        "extract_grid already uses for R, landing on the identical output "
        "mapping with no separate coordinate transform (see rebake_gb.py's "
        "module docstring for why that is exact for this grid, not an "
        "approximation).",
        "A": "land fraction (0..1), unmodified from source extraction. "
        "Unchanged by this update -- see validation.r_a_byte_identical.",
    }

    meta["validation"]["r_a_byte_identical"] = byte_check
    meta["validation"]["structure_fraction"] = {
        "global_area_and_abundance_weighted": global_g,
        "published_table1": PUBLISHED_STRUCTURE_FRACTION,
        "relative_error": rel_err,
        "hard_fail_relative_tolerance": STRUCTURE_FRACTION_REL_TOL,
        "n_texels_clipped_to_0_1": n_clipped,
        "max_clip_extent": max_clip_extent,
        "g_min_over_land": float(g_grid[sum_five > 0].min()),
        "g_max_over_land": float(g_grid[sum_five > 0].max()),
    }
    meta["validation"]["summation_trap"] = {
        "note": "do the five functional-group grids sum to the independently "
        "modelled TotalNumber_per100g grid? NO -- see per_pixel_ratio_* below. "
        "G is normalised by sum_five (the five groups), not TotalNumber, "
        "per the channel ruling.",
        "global_area_weighted_sum_five_over_total": global_sum_ratio,
        "per_pixel_ratio_min": float(np.nanmin(per_pixel_ratio[land_ratio_mask])),
        "per_pixel_ratio_median": float(np.nanmedian(per_pixel_ratio[land_ratio_mask])),
        "per_pixel_ratio_max": float(np.nanmax(per_pixel_ratio[land_ratio_mask])),
    }
    meta["validation"]["bootstrap_cv"] = {
        **b_stats,
        "raw_diluted_percentiles_from_extraction": boot_meta["percentiles"],
        "land_frac_cross_check_vs_R": land_frac_cross_check,
    }
    meta["source"]["functional_groups"] = GROUP_SOURCE
    meta["source"]["total_per100g"] = TOTAL_PER100G_SOURCE
    meta["source"]["bootstrap_coefvar"] = BOOTSTRAP_SOURCE
    meta["source"]["not_used_ensemble_coefvar"] = NOT_USED_ENSEMBLE_COEFVAR
    meta["bytes_per_face"] = total_bytes // FACE_COUNT
    meta["total_bytes"] = total_bytes

    with open(CUBE_DIR / "cube_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    report = {
        "global_structure_fraction": global_g,
        "global_structure_fraction_pct": global_g * 100,
        "published_pct": PUBLISHED_STRUCTURE_FRACTION * 100,
        "relative_error_pct": rel_err * 100,
        "groups_sum_to_total": False,
        "global_area_weighted_sum_five_over_total": global_sum_ratio,
        "n_g_texels_clipped": n_clipped,
        "bootstrap_cv_stats": b_stats,
        "land_frac_cross_check": land_frac_cross_check,
        "r_a_byte_identical": byte_check["all_faces_r_and_a_byte_identical"],
        "baked_density_integral": val["integral"],
        "total_polar_gap_texels": total_polar_gap,
    }
    print("\n=== rebake_gb.py: final report ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
