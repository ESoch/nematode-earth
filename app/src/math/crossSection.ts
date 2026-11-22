import { E_A } from './optical'

/**
 * Per-instance cross-sections are DRAWN FROM THE DISTRIBUTION, not set to the
 * mean (spec 7.7): A is heavily right-skewed with mean/median = 2.02.
 *
 * A regime-2 instance is a super-particle standing for `omega` real animals,
 * so the quantity that matters is the distribution of the SUM of omega draws,
 * not of one draw. This module models that sum as a log-normal matched on its
 * first two moments. That choice is exact at omega = 1 (sigma_omega^2 reduces
 * to the single-animal ln-variance identically) and has the correct mean and
 * variance at every omega, while staying strictly positive and right-skewed -
 * none of which a naive Gaussian or a mean-valued constant does.
 *
 * WHERE IT BITES: omega = sigma_t*s^3/E[A]. In the shipped regime-2 band
 * (10 km down to ~5 m) omega is 1e6 and up, so the draw is a sub-percent
 * correction. It approaches 1 only at cell sizes near 4 cm, i.e. viewing
 * distances near 5 m - exactly the regime 2->3 boundary of spec 7.8. The
 * instruction is therefore implemented correctly at both ends rather than
 * being decorative at one of them.
 */
export const MEAN_OVER_MEDIAN = 2.02

/** sigma^2 of ln A. mean/median = exp(sigma^2/2) => sigma^2 = 2*ln(2.02). */
export const LN_SIGMA2 = 2 * Math.log(MEAN_OVER_MEDIAN)

/** Var(A)/E[A]^2 = e^(sigma^2) - 1 = (mean/median)^2 - 1. */
export const REL_VAR = MEAN_OVER_MEDIAN * MEAN_OVER_MEDIAN - 1

export const MEDIAN_A = E_A / MEAN_OVER_MEDIAN

/** ln-variance of the aggregate of `omega` iid draws, moment-matched. */
export function aggregateLogSigma2(omega: number): number {
  return Math.log1p(REL_VAR / omega)
}

export function aggregateMean(omega: number): number {
  return omega * E_A
}

export function aggregateVariance(omega: number): number {
  return omega * E_A * E_A * REL_VAR
}

/** Var/E^2 of one instance's flux, the noise inflation term for math/lod.ts. */
export function relVarPerInstance(omega: number): number {
  return REL_VAR / omega
}

/** Standard normal from two uniforms. Cheap enough to mirror in TSL. */
export function boxMullerNormal(u1: number, u2: number): number {
  const r = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-7)))
  return r * Math.cos(2 * Math.PI * u2)
}

/** Inverse-CDF draw of the aggregate projected area, in m^2. */
export function drawAggregateArea(omega: number, u1: number, u2: number): number {
  const s2 = aggregateLogSigma2(omega)
  const mu = Math.log(aggregateMean(omega)) - s2 / 2
  return Math.exp(mu + Math.sqrt(s2) * boxMullerNormal(u1, u2))
}
