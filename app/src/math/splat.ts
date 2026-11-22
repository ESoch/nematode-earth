/**
 * Screen-space splat normalization, in the DISCRETE pixel domain (spec 7.7,
 * Schneegans/Kreskowski/Gerndt, "Smaller than Pixels", EG 2025).
 *
 * A kernel normalized as a continuous function of screen area does not
 * discretely sum to 1 at pixel centres, so energy scintillates as instances
 * drift across pixel boundaries. Two pieces here, and neither is a minimum
 * radius (which the spec calls a cheap proxy, not the fix):
 *
 * 1. `bilinearWeights` - the weighted 2x2 deposit. The four weights are
 *    (1-|dx|)(1-|dy|) at the four nearest pixel centres and they sum to 1
 *    ALGEBRAICALLY, for any subpixel position, with no normalization pass.
 *    The kernel reaches zero exactly at the edge of its 2x2 support, so a
 *    rasterizer tie at the quad boundary carries zero energy either way.
 *
 * 2. `gaussianTaps` - the separable resolve that realizes sigma_pix. Widening
 *    by convolving the deposit buffer is the SAME linear operator as splatting
 *    sigma_pix-wide kernels per instance, so section 7.7's variance arithmetic
 *    is untouched, at 4 fragments per instance instead of (6*sigma)^2:
 *    1.6e7 fragments at B = 4e6 rather than 1.3e9, an 81x reduction. Taps are
 *    normalized over the discrete lattice, not over the continuous Gaussian.
 *
 * `discreteGaussSumClosed` is the Poisson-summation form of the infinite
 * lattice sum, kept because it is the exact normalizer if a future pass ever
 * does splat a wide kernel per instance directly: truncation error after one
 * harmonic is 2*exp(-8*pi^2*sigma^2), which is 5.4e-9 at sigma = 0.5 and
 * 1.4e-2 at sigma = 0.25 - the arithmetic behind lod.ts's SIGMA_PIX_MIN.
 */

export const MAX_TAPS = 21

/**
 * Weights for the four pixel centres nearest a splat at fractional offset
 * (fx, fy) from the lower-left of the four, in [0,1). Order: (0,0), (1,0),
 * (0,1), (1,1).
 */
export function bilinearWeights(fx: number, fy: number): [number, number, number, number] {
  const gx = 1 - fx
  const gy = 1 - fy
  return [gx * gy, fx * gy, gx * fy, fx * fy]
}

/** Unnormalized 1D Gaussian. */
const k1 = (t: number, s: number): number => Math.exp(-(t * t) / (2 * s * s))

/** Discretely normalized separable taps, odd length, radius ceil(3*sigma). */
export function gaussianTaps(sigmaPix: number, maxRadius = (MAX_TAPS - 1) / 2): number[] {
  const r = Math.min(Math.ceil(3 * sigmaPix), maxRadius)
  const raw: number[] = []
  for (let i = -r; i <= r; i++) raw.push(k1(i, sigmaPix))
  const sum = raw.reduce((a, b) => a + b, 0)
  return raw.map((v) => v / sum)
}

/** Brute-force lattice sum, the reference. */
export function discreteGaussSum(frac: number, sigmaPix: number): number {
  let s = 0
  const r = Math.ceil(8 * sigmaPix) + 4
  for (let n = -r; n <= r; n++) s += k1(frac + n, sigmaPix)
  return s
}

/** Poisson summation, first harmonic. */
export function discreteGaussSumClosed(frac: number, sigmaPix: number): number {
  const s = sigmaPix
  const base = Math.sqrt(2 * Math.PI) * s
  return base * (1 + 2 * Math.exp(-2 * Math.PI * Math.PI * s * s) * Math.cos(2 * Math.PI * frac))
}

/**
 * Total energy of one unit-flux instance at continuous pixel coordinate
 * (cx, cy) after the 2x2 deposit and the separable resolve. Must be 1.
 */
export function depositThenBlurEnergy(cx: number, cy: number, taps: number[]): number {
  const ix = Math.floor(cx - 0.5)
  const iy = Math.floor(cy - 0.5)
  const w = bilinearWeights(cx - 0.5 - ix, cy - 0.5 - iy)
  const tapSum = taps.reduce((a, b) => a + b, 0)
  // Separable: the blur multiplies total energy by (sum taps)^2 = 1.
  return (w[0] + w[1] + w[2] + w[3]) * tapSum * tapSum
}
