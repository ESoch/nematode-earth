/**
 * The Cobb inversion (spec 6.3), CORRECTED per
 * `docs/superpowers/results/2026-08-16-urban-claim-ruling.md`.
 *
 * The original spec's data-end multiplier, `m_data(u) = 1 - 1.70u + 0.75u^2`
 * ("cities as voids"), does not ship. The exploratory calibration against
 * modelled soil abundance found a = -0.0085, nominal pixel-bootstrap 95%
 * CI [-0.035, 0.019], R^2 = -0.0003. That constrained quadratic did not
 * outperform the pooled mean. Its linear coefficient alone is not the
 * whole urban contrast (a + b), and these spatially dependent model pixels
 * are not independent field observations. This does not establish an
 * ecological null or settle Cobb's claim about particular taxa.
 *
 * So the corrected beat: the DATA end of the slider does nothing at all
 * (`dataMultiplier` is the constant 1, not a fitted curve), and only the
 * COBB end adds an authored illustration of his qualitative 1914 claim.
 * The factor of four is ours, not Cobb's. The unmodified data end omits
 * an unsupported extra urban adjustment; it does not assert that the
 * source model contains no information about human land use.
 *
 * `u` here is per-pixel GHSL built-up fraction (0..1) — NOT the slider
 * position. `cobbMix` (also 0..1, the actual slider) blends between the
 * two multiplier FUNCTIONS below; `regimes/Shell.ts`'s `applyCobbMix`
 * mirrors this in TSL and applies it to the veil's rendered density.
 */

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)

/** Authored illustration of Cobb's claim, up to 4x at fully built-up. */
export function cobbMultiplier(builtUpFraction: number): number {
  return 1 + 3 * clamp01(builtUpFraction)
}

/** No additional urban adjustment: retain the source model, not a fitted curve. */
export function dataMultiplier(_builtUpFraction: number): number {
  return 1
}

/** `mix(dataMultiplier(u), cobbMultiplier(u), cobbMix)` — the actual per-pixel density multiplier the slider produces. */
export function densityMultiplier(builtUpFraction: number, cobbMix: number): number {
  const m = clamp01(cobbMix)
  return dataMultiplier(builtUpFraction) * (1 - m) + cobbMultiplier(builtUpFraction) * m
}

/** Exploratory model-to-map fit; uncertainty excludes spatial dependence. */
export const MEASUREMENT_CAPTION = '243,294 modeled pixels · 20 cities · linear coefficient a = −0.0085 (nominal pixel-bootstrap 95% CI −0.035, 0.019) · R² = −0.0003'

/** Regional literature, not a universal urban multiplier; citation in Science & Making. */
export const COMPOSITION_CAPTION = 'Community composition can change without a matching change in total abundance. Gong et al. (2024) reported lower proportions of omnivores and predators in their urban soils. This regional finding does not establish a global urban pattern.'
