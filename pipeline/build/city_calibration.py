#!/usr/bin/env python3
"""City-scale urban-suppression calibration: extract native-resolution
(~1 km) nematode density windows around ~20 real cities from the live ETH
raster, pair each with the matching GHSL built-up-fraction window (same
lat/lon footprint, resampled onto the density grid), and fit
m(u) = 1 + a*u + b*u^2 against genuine intra-city urban/rural contrast.

Controls, per the task brief:
  - Latitude / biome confound: each city's rural baseline is its OWN
    low-u (u<0.02) pixels WITHIN THE SAME WINDOW, not a global or
    cross-city reference -- so the fit measures "how much does density
    drop going from this city's rural surroundings to its built-up core",
    pooled across cities, never "how does city A compare to city B."
  - Coastal dilution: ocean pixels (density NaN in the native raster) are
    excluded entirely from both the rural baseline and the u-vs-m sample,
    for every city, port or not.

Run: uv run build/city_calibration.py
"""

from __future__ import annotations

import json
import math

import numpy as np
import tifffile
from city_windows import (
    CITIES,
    WINDOW_DEG,
    city_bbox,
    fetch_density_header,
    fetch_density_window,
)
from urban_calibration import GHSL_PATH, fit_quadratic_through_one

SPEC_A, SPEC_B = -1.70, 0.75
N_BOOTSTRAP = 2000
RNG_SEED = 42


def load_ghsl_geo():
    with tifffile.TiffFile(str(GHSL_PATH)) as tf:
        tags = {t.name: t.value for t in tf.pages[0].tags}
    g_scale = tags["ModelPixelScaleTag"]
    g_tie = tags["ModelTiepointTag"]
    return g_tie[3], g_tie[4], g_scale[0], g_scale[1]  # lon0, lat0, dlon, dlat


def ghsl_window_fraction(
    ghsl_mm: np.memmap,
    ghsl_shape,
    g_lon0,
    g_lat0,
    g_dlon,
    g_dlat,
    lat_n: float,
    lat_s: float,
    lon_w: float,
    lon_e: float,
    out_rows: int,
    out_cols: int,
) -> np.ndarray:
    """Built-up fraction on the OUTPUT (density) grid, for the box
    [lat_s,lat_n] x [lon_w,lon_e], by nearest-neighbor lookup into the
    GHSL array (both grids are ~30 arc-second, so nearest-neighbor at
    this scale introduces sub-pixel alignment error only, not a
    resolution mismatch)."""
    ghsl_h, ghsl_w = ghsl_shape
    lat_edges = np.linspace(lat_n, lat_s, out_rows + 1)
    lon_edges = np.linspace(lon_w, lon_e, out_cols + 1)
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    g_row = np.clip(np.round((g_lat0 - lat_c) / g_dlat).astype(int), 0, ghsl_h - 1)
    g_col = np.clip(np.round((lon_c - g_lon0) / g_dlon).astype(int), 0, ghsl_w - 1)
    built_m2 = ghsl_mm[np.ix_(g_row, g_col)].astype(np.float64)

    # Physical pixel area (m^2) at each output row's latitude, for the
    # fraction denominator.
    R_MEAN = 6_371_000.0
    dlat_out = (lat_n - lat_s) / out_rows
    dlon_out = (lon_e - lon_w) / out_cols
    band_area = (
        R_MEAN**2
        * math.radians(dlon_out)
        * (np.sin(np.radians(lat_edges[:-1])) - np.sin(np.radians(lat_edges[1:])))
    )
    pixel_area = band_area[:, None]  # one value per row, broadcast over columns
    return np.clip(built_m2 / pixel_area, 0.0, 1.0)


def bootstrap_ci(u: np.ndarray, m: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(u)
    a_samples, b_samples = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b, _ = fit_quadratic_through_one(u[idx], m[idx])
        a_samples.append(a)
        b_samples.append(b)
    a_samples, b_samples = np.array(a_samples), np.array(b_samples)
    return {
        "a_ci95": [
            float(np.percentile(a_samples, 2.5)),
            float(np.percentile(a_samples, 97.5)),
        ],
        "b_ci95": [
            float(np.percentile(b_samples, 2.5)),
            float(np.percentile(b_samples, 97.5)),
        ],
    }


def main():
    print("fetching density header...")
    client, info = fetch_density_header()

    ghsl_lon0, ghsl_lat0, ghsl_dlon, ghsl_dlat = load_ghsl_geo()
    print(f"GHSL geo: lon0={ghsl_lon0:.4f} lat0={ghsl_lat0:.4f} dlon={ghsl_dlon:.6f}")
    # GHSL is LZW-compressed and tiled, so it isn't memory-mappable;
    # tifffile.imread decodes it fully (~3.7 GB as uint32) -- already
    # proven to work in urban_calibration.py's global regression.
    ghsl = tifffile.imread(str(GHSL_PATH))
    print(f"GHSL array shape: {ghsl.shape}")

    per_city = {}
    all_u, all_m = [], []

    for name, (lat, lon) in CITIES.items():
        row0, row1, col0, col1 = city_bbox(lat, lon, info)
        density = fetch_density_window(client, info, row0, row1, col0, col1)
        half = WINDOW_DEG / 2
        lat_n, lat_s = lat + half, lat - half
        lon_w, lon_e = lon - half, lon + half
        u = ghsl_window_fraction(
            ghsl,
            ghsl.shape,
            ghsl_lon0,
            ghsl_lat0,
            ghsl_dlon,
            ghsl_dlat,
            lat_n,
            lat_s,
            lon_w,
            lon_e,
            density.shape[0],
            density.shape[1],
        )

        land_mask = ~np.isnan(density)  # coastal dilution control: drop ocean entirely
        d, uu = density[land_mask], u[land_mask]
        rural = d[uu < 0.02]
        n_land = int(land_mask.sum())
        if rural.size < 20 or rural.mean() <= 0:
            print(
                f"  {name}: SKIPPED (only {rural.size} rural land pixels, "
                f"insufficient for a local baseline)"
            )
            per_city[name] = {
                "n_land_pixels": n_land,
                "n_rural": int(rural.size),
                "skipped": True,
            }
            continue
        baseline = float(rural.mean())
        m = d / baseline

        per_city[name] = {
            "lat": lat,
            "lon": lon,
            "n_land_pixels": n_land,
            "n_rural_pixels": int(rural.size),
            "rural_baseline_density": baseline,
            "max_u_observed": float(uu.max()),
            "mean_u_observed": float(uu.mean()),
            "correlation_u_vs_m": float(np.corrcoef(uu, m)[0, 1])
            if uu.std() > 0
            else None,
        }
        all_u.append(uu)
        all_m.append(m)
        print(
            f"  {name}: {n_land} land px, {rural.size} rural, baseline={baseline:.3e}, "
            f"max_u={uu.max():.3f}, corr={per_city[name]['correlation_u_vs_m']}"
        )

    all_u = np.concatenate(all_u)
    all_m = np.concatenate(all_m)
    # Same outlier guard as the coarse regression: drop the extreme 1%
    # of suppression-ratio values (thin-tailed baseline noise), not the
    # u values themselves.
    lo, hi = np.percentile(all_m, [0.5, 99.5])
    keep = (all_m >= lo) & (all_m <= hi)
    all_u, all_m = all_u[keep], all_m[keep]

    a_fit, b_fit, r2 = fit_quadratic_through_one(all_u, all_m)
    ci = bootstrap_ci(all_u, all_m, N_BOOTSTRAP, RNG_SEED)

    result = {
        "window_deg": WINDOW_DEG,
        "n_cities_used": sum(1 for c in per_city.values() if not c.get("skipped")),
        "n_cities_total": len(CITIES),
        "n_pooled_samples": int(all_u.size),
        "u_range_observed": [float(all_u.min()), float(all_u.max())],
        "fitted_a": a_fit,
        "a_ci95": ci["a_ci95"],
        "fitted_b": b_fit,
        "b_ci95": ci["b_ci95"],
        "r_squared": r2,
        "spec_a": SPEC_A,
        "spec_b": SPEC_B,
        "m_fitted_at_u1": 1 + a_fit + b_fit,
        "correlation_u_vs_m_pooled": float(np.corrcoef(all_u, all_m)[0, 1]),
        "per_city": per_city,
    }
    print("\n=== city-scale calibration result ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
