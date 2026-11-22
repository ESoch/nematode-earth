#!/usr/bin/env python3
"""Calibrate the urban-suppression multiplier m(u) = 1 + a*u + b*u^2
(spec section 6.3: m_data(u) = 1 - 1.70u + 0.75u^2, flagged as resting on
"a source that could not be located") against the actual nematode density
raster and GHSL built-up surface, per the task brief: "calibrate against
data we already have... since you have both rasters."

ADAPTATION FROM THE LITERAL "~20 CITIES" METHODOLOGY -- and why:

The brief's (and the spec's) suggested method is "sample nematode density
inside versus outside built-up areas across ~20 cities." That needs
nematode density resolved at roughly city scale (a few km) so "inside"
and "outside" a city are actually distinguishable pixels. The nematode
density this pipeline has extracted so far (pipeline/out/
total_persqm_density_512x256.npy) is ~78 km/px (session 2's stratified
2.42%-row-coverage extraction) -- an entire large metropolitan area plus
its rural surroundings falls inside ONE pixel at that resolution. Doing
"inside vs outside" city-by-city literally would require new targeted
high-resolution range-fetches against the ETH source for ~20 city
windows, against a server with demonstrated (if survivable) flakiness
(see pipeline-report.md).

Given time budget, this script instead runs a REGRESSION at the
resolution actually available: every one of the 512x256 coarse density
pixels gets a built-up FRACTION u by aggregating GHSL's much finer
(30 arc-second, ~860 m/px) built-up-area grid within that coarse pixel's
footprint, and density is regressed against u. Raw global regression would
be badly confounded by biome/latitude (density varies enormously by
climate zone independent of urbanization, and cities aren't uniformly
distributed across biomes), so this bins by latitude band first and
fits within-band suppression ratios against a within-band rural (u~0)
baseline before pooling -- controlling for the dominant confound the
naive version would miss.

This exploratory fit compares model pixels at about 78 km resolution.
For the twenty-city comparison using native-resolution windows, see
city_calibration.py and the city comparison in the documentation index.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tifffile

GHSL_PATH = Path("cache/ghsl/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif")
DENSITY_NPY = Path("out/total_persqm_density_512x256.npy")
LANDFRAC_NPY = Path("out/total_persqm_landfrac_512x256.npy")
DENSITY_META = Path("out/total_persqm_meta_512x256.json")
R_MEAN = 6_371_000.0

SPEC_A, SPEC_B = -1.70, 0.75  # the unverified constants being checked


def pixel_area_m2(lat_top_deg: float, lat_bot_deg: float, dlon_deg: float) -> float:
    band_area = (
        R_MEAN**2
        * math.radians(360.0)
        * (math.sin(math.radians(lat_top_deg)) - math.sin(math.radians(lat_bot_deg)))
    )
    return band_area * (dlon_deg / 360.0)


def compute_built_up_fraction_grid(
    ghsl: np.ndarray,
    ghsl_lon0: float,
    ghsl_lat0: float,
    ghsl_dlon: float,
    ghsl_dlat: float,
    out_h: int,
    out_w: int,
    lon0: float,
    lat0: float,
    dlon: float,
    dlat: float,
) -> np.ndarray:
    """For each of the out_h x out_w coarse cells, sum GHSL built-up area
    (m^2) within its lon/lat footprint and divide by the cell's own
    physical area to get a 0..1 fraction."""
    ghsl_h, ghsl_w = ghsl.shape
    u = np.zeros((out_h, out_w), dtype=np.float64)
    for r in range(out_h):
        lat_top = lat0 - r * dlat
        lat_bot = lat0 - (r + 1) * dlat
        g_r0 = int(round((ghsl_lat0 - lat_top) / ghsl_dlat))
        g_r1 = int(round((ghsl_lat0 - lat_bot) / ghsl_dlat))
        g_r0, g_r1 = max(0, g_r0), min(ghsl_h, g_r1)
        if g_r1 <= g_r0:
            continue
        cell_area = pixel_area_m2(lat_top, lat_bot, dlon)
        for c in range(out_w):
            lon_l = lon0 + c * dlon
            lon_r = lon0 + (c + 1) * dlon
            g_c0 = int(round((lon_l - ghsl_lon0) / ghsl_dlon))
            g_c1 = int(round((lon_r - ghsl_lon0) / ghsl_dlon))
            g_c0, g_c1 = max(0, g_c0), min(ghsl_w, g_c1)
            if g_c1 <= g_c0:
                continue
            built_up_m2 = float(ghsl[g_r0:g_r1, g_c0:g_c1].sum(dtype=np.float64))
            u[r, c] = min(1.0, built_up_m2 / cell_area)
    return u


def fit_quadratic_through_one(
    u: np.ndarray, m: np.ndarray
) -> tuple[float, float, float]:
    """Least-squares fit of m = 1 + a*u + b*u^2 (intercept fixed at 1, by
    construction -- m(0)=1 is not a free parameter, it's the definition
    of "no suppression"). Returns (a, b, r_squared)."""
    y = m - 1.0
    X = np.column_stack([u, u**2])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    pred = 1.0 + a * u + b * u**2
    ss_res = float(np.sum((m - pred) ** 2))
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def main():
    with open(DENSITY_META) as f:
        meta = json.load(f)
    density = np.load(DENSITY_NPY).astype(np.float64)
    land = np.load(LANDFRAC_NPY).astype(np.float64)
    out_h, out_w = density.shape
    lon0, lat0 = meta["tiff"]["model_tiepoint"][3], meta["tiff"]["model_tiepoint"][4]
    dlon = 360.0 / out_w
    dlat = (2 * lat0) / out_h

    print(f"loading GHSL from {GHSL_PATH} ...")
    with tifffile.TiffFile(str(GHSL_PATH)) as tf:
        page = tf.pages[0]
        tags = {t.name: t.value for t in page.tags}
    ghsl = tifffile.imread(str(GHSL_PATH)).astype(np.float64)
    g_scale = tags["ModelPixelScaleTag"]
    g_tie = tags["ModelTiepointTag"]
    ghsl_dlon, ghsl_dlat = g_scale[0], g_scale[1]
    ghsl_lon0, ghsl_lat0 = g_tie[3], g_tie[4]
    print(
        f"GHSL shape={ghsl.shape} lon0={ghsl_lon0:.4f} lat0={ghsl_lat0:.4f} "
        f"dlon={ghsl_dlon:.6f} dlat={ghsl_dlat:.6f}"
    )

    u = compute_built_up_fraction_grid(
        ghsl,
        ghsl_lon0,
        ghsl_lat0,
        ghsl_dlon,
        ghsl_dlat,
        out_h,
        out_w,
        lon0,
        lat0,
        dlon,
        dlat,
    )

    # Restrict to solid land pixels (land_fraction > 0.5) to avoid coastal
    # dilution noise; latitude-band the rest to control the biome confound.
    land_mask = land > 0.5
    lat_centers = lat0 - (np.arange(out_h) + 0.5) * dlat
    lat_grid = np.broadcast_to(lat_centers[:, None], (out_h, out_w))
    band_width = 10.0
    bands = np.floor((lat_grid + 90) / band_width).astype(int)

    pooled_u, pooled_m, band_reports = [], [], []
    for b in np.unique(bands[land_mask]):
        sel = land_mask & (bands == b)
        if sel.sum() < 8:
            continue
        u_b, d_b = u[sel], density[sel]
        rural = d_b[u_b < 0.01]
        if rural.size < 3 or rural.mean() <= 0:
            continue
        baseline = rural.mean()
        m_b = d_b / baseline
        pooled_u.append(u_b)
        pooled_m.append(m_b)
        band_reports.append(
            {
                "lat_band": f"{b * band_width - 90:.0f} to {(b + 1) * band_width - 90:.0f}",
                "n_pixels": int(sel.sum()),
                "n_rural": int(rural.size),
                "rural_baseline_density": float(baseline),
            }
        )

    all_u = np.concatenate(pooled_u)
    all_m = np.concatenate(pooled_m)
    # Robustness: clip suppression ratio outliers (>99.5th pctile) before
    # fitting -- a handful of pixels with tiny rural samples produce wild
    # baseline estimates otherwise.
    lo, hi = np.percentile(all_m, [0.5, 99.5])
    keep = (all_m >= lo) & (all_m <= hi)
    all_u, all_m = all_u[keep], all_m[keep]

    a_fit, b_fit, r2 = fit_quadratic_through_one(all_u, all_m)

    result = {
        "n_bands_used": len(band_reports),
        "n_pooled_samples": int(all_u.size),
        "fitted_a": a_fit,
        "fitted_b": b_fit,
        "r_squared": r2,
        "spec_a": SPEC_A,
        "spec_b": SPEC_B,
        "m_fitted_at_u1": 1 + a_fit + b_fit,
        "m_spec_at_u1": 1 + SPEC_A + SPEC_B,
        "correlation_u_vs_m": float(np.corrcoef(all_u, all_m)[0, 1]),
        "band_reports": band_reports,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
