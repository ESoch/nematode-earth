"""The density cube's B channel: bootstrap coefficient of variation.

Per the channel ruling, B carries real per-pixel model uncertainty from
`TotalNumber_Bootstrap_CoefVar.tif` -- 100 stratified bootstrap iterations
of the same random-forest model, 209M pixels classified 100 times each,
yielding a per-pixel 95%-CI-derived coefficient of variation. This is
deliberately NOT `TotalNumber_Ensemble_CoefVar.tif` (seed-to-seed jitter of
a single model run -- convenient because it shares the main 43201x21121
grid, but it would read to a viewer as "confidence" while measuring
nothing of the kind; see the ruling).

**Un-dilution, not un-modified reuse of the density convention.**
`eth_raster.extract_grid`'s box average treats no-data (ocean, or any
pixel where the model didn't run) as 0 and averages over ALL cells in the
box -- correct for a density, because density is defined per unit AREA of
pixel and diluting a coastal box with ocean=0 IS the physically right
thing to do (see `eth_raster.py`'s own docstring on this). A coefficient
of variation is not an areal quantity; it is defined PER SOIL SAMPLE.
Diluting it toward 0 at a coastal box would silently understate
uncertainty exactly where the coastline makes the model least reliable --
the opposite of what a "confidence" channel should do. `land_mean_from_diluted`
divides the dilution back out.
"""

from __future__ import annotations

import numpy as np

UNDILUTION_EPS = 1e-9


def land_mean_from_diluted(raw_avg: np.ndarray, land_frac: np.ndarray) -> np.ndarray:
    """Recover the mean value over the LAND-ONLY portion of each output
    cell from `extract_grid`'s mass-conserving (NaN->0, average-over-all-
    cells) box average. `land_frac == 0` (pure ocean / no-data) decodes to
    exactly 0.0, matching the R/A/G no-data convention -- never a 0/0 NaN
    or inf."""
    safe_frac = np.where(land_frac > 0, land_frac, 1.0)
    undiluted = raw_avg / safe_frac
    return np.where(land_frac > 0, undiluted, 0.0)


def bootstrap_cv_stats(b: np.ndarray, land_frac: np.ndarray) -> dict:
    """Basic descriptive stats over land cells (land_frac > 0) -- used by
    both the hard-fail gate and the bake's metadata report."""
    land = b[land_frac > 0]
    if land.size == 0:
        return {
            "n_land_cells": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "n_land_cells": int(land.size),
        "min": float(land.min()),
        "max": float(land.max()),
        "mean": float(land.mean()),
        "median": float(np.median(land)),
    }


def assert_bootstrap_cv_valid(
    b: np.ndarray, land_frac: np.ndarray, raw_avg: np.ndarray
) -> dict:
    """Hard-fail gate (raises AssertionError, not a printed warning) for
    the un-diluted B channel:

      1. Non-negative everywhere land_frac > 0 -- a coefficient of
         variation is std/mean of a non-negative bootstrap distribution
         and cannot be negative. This is a mathematical requirement, not
         a data-dependent threshold.
      2. Finite everywhere land_frac > 0 -- no NaN/inf reaching the
         shader.
      3. Never smaller than the raw diluted box average at any land cell
         (dividing by land_frac in (0,1] can only hold steady or grow the
         magnitude) -- the mechanical signature that un-dilution actually
         ran, and ran the right direction. Catches both "forgot to
         un-dilute" (b == raw_avg exactly, only passes at land_frac==1)
         and "un-diluted backwards" (multiplied instead of divided, which
         SHRINKS values away from land_frac==1).

    Returns the descriptive-stats dict (`bootstrap_cv_stats`) for the
    caller to fold into `cube_meta.json`."""
    land_mask = land_frac > 0
    land_vals = b[land_mask]

    # Finite check first: NaN fails EVERY comparison (NaN >= 0 is False),
    # so checking non-negativity first would misreport a NaN as "0
    # negative cells" instead of naming the real problem.
    assert np.all(np.isfinite(land_vals)), (
        f"bootstrap CV must be finite everywhere land_frac>0 -- found "
        f"{int((~np.isfinite(land_vals)).sum())} non-finite cell(s)"
    )
    assert np.all(land_vals >= 0.0), (
        f"bootstrap CV must be non-negative everywhere land_frac>0 -- found "
        f"{int((land_vals < 0.0).sum())} negative cell(s), min={land_vals.min():.6g}"
    )
    shrank = b[land_mask] < (raw_avg[land_mask] - UNDILUTION_EPS)
    assert not np.any(shrank), (
        f"un-dilution must never shrink a land cell's value below the raw "
        f"diluted box average -- found {int(shrank.sum())} cell(s) where it "
        f"did, meaning the un-dilution step was skipped or ran backwards"
    )

    return bootstrap_cv_stats(b, land_frac)
