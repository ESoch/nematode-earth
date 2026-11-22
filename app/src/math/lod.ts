/**
 * Regime-2 LOD arithmetic. Pure, no three.js.
 *
 * Every constant here is derived, not invented. The spec's `N_pix ~ 2.20*d^2`
 * (section 7.8) is exactly `N_A / f_px^2` at 1080p with a 50 degree vertical
 * FOV: 2.95e6 / 1158.03^2 = 2.2001. The spec's 1.93 instances/px, 28x lift,
 * ~55 effective samples and 72%/14% noise all fall out of the two functions
 * below with B = 4e6 at 1920x1080.
 *
 * NOT DEAD CODE, BUT NOT WIRED IN EITHER: `fullPopulationPerPixel` /
 * `instancesPerPixel` / `unityPixelDistance` model a WHOLE-SCREEN Poisson
 * placement (budget spread as an average density over every visible
 * pixel) - the noise-budget argument spec 7.7 uses to decide the budget
 * closes at all. `regimes/Particles.ts`'s actual `emitKernel` places
 * particles with a different scheme entirely: a world-space volumetric
 * lattice confined to a small window around the camera. Nothing in
 * `Particles.ts` calls any function in this file except `sigmaPixAt`/
 * `cellSizeForBudget` - `fullPopulationPerPixel` and friends are not
 * currently exercised by the render path at all. That is deliberate per
 * `Particles.ts`'s own "SIMPLIFIED" doc comment, not an oversight, but a
 * future reader who assumes this file is wired into what is on screen
 * would be wrong - see the matching note in `Particles.ts`.
 */

const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi)
const smooth = (u: number): number => u * u * (3 - 2 * u)

/** Pixels subtended by one radian at frame centre. */
export function pixelsPerRadian(heightPx: number, fovYDeg: number): number {
  return (heightPx / 2) / Math.tan((fovYDeg * Math.PI) / 360)
}

/** Full-population instances in one pixel's cone at nadir: N_A * (d/f_px)^2. */
export function fullPopulationPerPixel(d: number, nA: number, fPx: number): number {
  const k = d / fPx
  return nA * k * k
}

/** Distance at which the column under one pixel holds exactly one animal. */
export function unityPixelDistance(nA: number, fPx: number): number {
  return fPx / Math.sqrt(nA)
}

export function instancesPerPixel(population: number, budget: number, nPixels: number): number {
  return Math.min(population, budget) / nPixels
}

/**
 * σ_pix scheduling constants (spec 7.7). Splat widening is the PRIMARY
 * mechanism the budget closes against, not a fallback: once population
 * exceeds the draw budget, instances-per-pixel pins at `B/n_pixels`
 * (1.929/px at 1080p, B=4e6) regardless of how much richer the true field
 * is - that is 72% Poisson noise at every altitude, unwidened. Widening to
 * sigma_pix=3 lifts effective samples ~28x (to ~54.5, 13.5% noise), which is
 * the only reason the budget closes at all. See effectiveSamples/noiseFraction below.
 */
export const SIGMA_PIX_MIN = 0.5
/** Shipped low end of the schedule; the spec's band is 0.5-1 px. */
export const SIGMA_PIX_SHIP_MIN = 0.75
export const SIGMA_PIX_MAX = 3.0
export const SIGMA_PIX_TOP_ALT = 10000
export const SIGMA_PIX_BOT_ALT = 1000

/**
 * Effective samples per pixel under the spec's disc convention: a splat of
 * radius sigma_pix covers pi*sigma^2 pixels, so a lambda-per-pixel Poisson
 * process behaves like pi*sigma^2*lambda samples. This is the convention that
 * reproduces section 7.7's own numbers (28x, ~55, 14%) and is therefore the
 * shipped one. Below one pixel of coverage there is no lift.
 */
export function effectiveSamples(lambda: number, sigmaPix: number): number {
  return lambda * Math.max(1, Math.PI * sigmaPix * sigmaPix)
}

/**
 * The same quantity for a true normalized Gaussian, where 1/integral(k^2) is
 * 4*pi*sigma^2. Exactly 4x the disc figure, i.e. the spec is conservative by
 * a factor of 2 in noise. Reported, never used for scheduling.
 */
export function effectiveSamplesGaussian(lambda: number, sigmaPix: number): number {
  return lambda * Math.max(1, 4 * Math.PI * sigmaPix * sigmaPix)
}

/**
 * Relative brightness noise. `relVarPerInstance` is Var(X)/E[X]^2 of the
 * per-instance flux; it is 0 when every instance carries the mean and
 * 3.0804/omega when cross-sections are drawn from the real distribution
 * (see math/crossSection.ts). Poisson-only is the omega >> 1 limit and is
 * what the spec quotes.
 */
export function noiseFraction(nEff: number, relVarPerInstance = 0): number {
  return Math.sqrt((1 + relVarPerInstance) / nEff)
}

/** Cell edge that puts `lambda` cells in each pixel's cone over `columnLength`. */
export function cellSizeFor(d: number, columnLength: number, lambda: number, fPx: number): number {
  const k = d / fPx
  return Math.cbrt((k * k * columnLength) / lambda)
}

/** Cell edge that fits `budget` cells into `volumeM3`. */
export function cellSizeForBudget(volumeM3: number, budget: number): number {
  return Math.cbrt(volumeM3 / budget)
}

/** Volume of a symmetric perspective frustum truncated at `range`. */
export function frustumVolume(range: number, fovYDeg: number, aspect: number): number {
  const t = Math.tan((fovYDeg * Math.PI) / 360)
  return (4 / 3) * range * range * range * t * t * aspect
}

/**
 * Lattice base cell. The lattice is quantized to octaves of S0 so that a
 * cell's integer index - and therefore its hashed jitter, cross-section and
 * weight - is invariant while the level holds. A continuously varying cell
 * size would move every instance every frame, which is the swimming the
 * brief forbids.
 */
export const S0 = 1 / 1024

export function octaveSplit(sTarget: number): {
  level: number; frac: number; sizeLo: number; sizeHi: number
} {
  const phi = Math.log2(sTarget / S0)
  const level = Math.floor(phi)
  const sizeLo = S0 * Math.pow(2, level)
  return { level, frac: phi - level, sizeLo, sizeHi: sizeLo * 2 }
}

/**
 * sigma_pix against altitude (spec 7.7): ~3 px at the 10 km end, tightening as
 * the handoff completes. Widening is free in energy and fill but spends
 * sharpness, so it is unusable low in regime 2.
 */
export function sigmaPixAt(altM: number): number {
  const lo = Math.log10(SIGMA_PIX_BOT_ALT)
  const hi = Math.log10(SIGMA_PIX_TOP_ALT)
  const u = clamp((Math.log10(Math.max(altM, 1)) - lo) / (hi - lo), 0, 1)
  const s = SIGMA_PIX_SHIP_MIN + (SIGMA_PIX_MAX - SIGMA_PIX_SHIP_MIN) * smooth(u)
  return Math.max(SIGMA_PIX_MIN, s)
}
