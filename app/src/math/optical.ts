import { WGS84 } from './wgs84'

/**
 * Assumed abundance-weighted mean projected area for the rendering model.
 *
 * Design rationale: 0.6426 * E[L*D_max], E[L*D_max] = 2.31e4 um^2, with
 * a chosen 30% adult / 70% juvenile weighting and reference to van den
 * Hoogen 2019 Table 1 biomass. Biomass does not uniquely determine area;
 * no reproducible calibration or statistical CI is supplied. This is a
 * display-model assumption, not a measured global silhouette distribution.
 */
export const E_A = 1.49e-8

export const N_A_LAND = 2.95e6
export const N_A_ICEFREE = 3.4e6

/**
 * tau = E[A] * N_A. Depends ONLY on the column integral, not on the profile
 * shape, so the per-m^2 raster maps directly to nadir optical depth with no
 * depth model in the shader loop. Spec section 4.2.
 */
export function tauNadir(nA: number): number {
  return E_A * nA
}

/**
 * Limb/nadir optical-depth ratio for an exponential profile: Q = sqrt(2*pi*R/Hs).
 *
 * Set by the PROFILE SCALE HEIGHT, not by the geometric bound. The tangent ray
 * traverses the shell in both directions, which is why this is 2x the one-sided
 * Chapman asymptote ch(x, pi/2) = sqrt(pi*x/2). Spec section 7.4.
 *
 * The uniform-slab formula 2*sqrt(2R/H + 1) is a DIFFERENT quantity and must
 * not be substituted here.
 */
export function limbRatio(Hs: number, R: number = WGS84.R_MEAN): number {
  return Math.sqrt((2 * Math.PI * R) / Hs)
}

export function tauLimb(tauN: number, Hs: number): number {
  return tauN * limbRatio(Hs)
}

/**
 * Largest nadir depth that keeps tau_limb at or below 4 (the readable ceiling).
 * Without this cap the descent ramp drives tau_limb from 1.7 to 35.
 * Spec section 7.3.
 */
export function saturationCapTau(Hs: number): number {
  return 4 / limbRatio(Hs)
}

/**
 * Peak extinction such that the truncated exponential integrates to exactly
 * tau_nadir over [0, Hmax].
 */
export function sigma0(tauN: number, Hs: number, Hmax: number): number {
  return tauN / (Hs * (1 - Math.exp(-Hmax / Hs)))
}

/** Silhouette coverage for Poisson-placed particles. Exact, not an approximation. */
export function arealCoverage(tau: number): number {
  return 1 - Math.exp(-tau)
}
