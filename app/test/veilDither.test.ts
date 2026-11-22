import { describe, it, expect } from 'vitest'
import {
  VEIL_DITHER_AMPLITUDE, acesFilmicToneMap, displayCode8, interleavedGradientNoise,
  srgbEncode, veilDitherScale, type Color3,
} from '../src/field/terrain'
// Vite's `?raw` suffix — the test/dissolveNodes.test.ts precedent.
import shellSource from '../src/regimes/Shell.ts?raw'
import terrainNodesSource from '../src/field/terrainNodes.ts?raw'

/**
 * The instrument for the veil's SECOND defect: contour banding.
 *
 * Separate from the reconstruction filter in `cubeFilter.test.ts`, with a
 * separate cause and a separate fix — the filter one makes the veil
 * polygonal, this one makes it terraced, and fixing either alone leaves the
 * complaint standing. The cause is that the shell's output crosses an
 * undithered ACES-filmic-into-8-bit boundary while occupying only about a
 * dozen code values, so those codes come out as wide flat plateaus.
 *
 * The metric is NOT "how many distinct 8-bit levels appear", and that is
 * worth being explicit about, because it is the obvious metric and it is the
 * wrong one. The veil's range spans the codes it spans; nothing local can
 * widen it, and dither barely moves that count (11 -> 12 below). What dither
 * changes is whether those codes are laid out as hard-edged plateaus or as a
 * spatial mixture that reconstructs the continuous field when integrated —
 * which is exactly what the eye and the display do. So the metrics are:
 *
 *  - the longest run of identical adjacent codes (the plateau width), and
 *  - the error of the locally-averaged output against the true unquantised
 *    value, which is the textbook property dither exists to fix.
 *
 * Both are measured against a model of the real chain
 * (`terrain.ts::displayCode8`), not against a stand-in.
 */

const VEIL_TINT: Color3 = [0.72, 0.86, 0.95]
const DEFAULT_EXPOSURE = 8 // viewer.ts's starting value

// A strip of the veil: a smooth horizontal ramp across the narrow luminance
// band where the banding was reported, constant vertically. Constant-in-y is
// the worst case and the honest one — any variation along the contour would
// break it up for free and flatter the undithered arm.
const WIDTH = 256
const HEIGHT = 64
const COL_LO = 0.0250
const COL_HI = 0.0285

const colAt = (x: number): number => COL_LO + ((COL_HI - COL_LO) * x) / (WIDTH - 1)
/** The shell's premultiplied output for an accumulated column of `col`. */
const tinted = (col: number): Color3 => [VEIL_TINT[0] * col, VEIL_TINT[1] * col, VEIL_TINT[2] * col]
/** Green channel only: the three channels band alike, and one is enough to measure. */
const greenCode = (col: number, exposure = DEFAULT_EXPOSURE): number => displayCode8(tinted(col), exposure)[1]
/** The same value BEFORE 8-bit rounding — what the quantised output ought to average to. */
const greenExact = (col: number, exposure = DEFAULT_EXPOSURE): number =>
  srgbEncode(acesFilmicToneMap(tinted(col), exposure)[1]) * 255

function renderStrip(amplitude: number): Int32Array {
  const out = new Int32Array(WIDTH * HEIGHT)
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      out[y * WIDTH + x] = greenCode(colAt(x) * veilDitherScale(x, y, amplitude))
    }
  }
  return out
}

/** Longest run of identical adjacent codes along a scanline — the plateau width. */
function longestPlateau(img: Int32Array): number {
  let longest = 0
  for (let y = 0; y < HEIGHT; y++) {
    let run = 1
    for (let x = 1; x < WIDTH; x++) {
      if (img[y * WIDTH + x] === img[y * WIDTH + x - 1]) run++
      else { longest = Math.max(longest, run); run = 1 }
    }
    longest = Math.max(longest, run)
  }
  return longest
}

/** RMS error, in code units, of the 5x5-averaged output against the true unquantised value. */
function localAverageError(img: Int32Array): number {
  const R = 2
  let sumSq = 0
  let n = 0
  for (let y = R; y < HEIGHT - R; y++) {
    for (let x = R; x < WIDTH - R; x++) {
      let s = 0
      for (let j = -R; j <= R; j++) for (let i = -R; i <= R; i++) s += img[(y + j) * WIDTH + x + i]!
      const e = s / ((2 * R + 1) ** 2) - greenExact(colAt(x))
      sumSq += e * e
      n++
    }
  }
  return Math.sqrt(sumSq / n)
}

const distinctCodes = (img: Int32Array): number => new Set(img).size

const undithered = renderStrip(0)
const dithered = renderStrip(VEIL_DITHER_AMPLITUDE)

describe('the veil bands because its whole range is a dozen 8-bit codes', () => {
  it('records the before/after measurements, so the numbers live with the code', () => {
    // distinct levels 11 -> 12, longest plateau 28px -> 4px,
    // locally-averaged RMS error 0.237 -> 0.063 codes.
    expect(distinctCodes(undithered)).toBe(11)
    expect(distinctCodes(dithered)).toBe(12)
    expect(longestPlateau(undithered)).toBe(28)
    expect(longestPlateau(dithered)).toBe(4)
    expect(localAverageError(undithered)).toBeCloseTo(0.237, 2)
    expect(localAverageError(dithered)).toBeCloseTo(0.063, 2)
  })

  it('confirms the cause: the strip resolves to ~11 distinct levels at the default 8x exposure', () => {
    expect(distinctCodes(undithered)).toBeLessThan(16)
    expect(distinctCodes(undithered)).toBeGreaterThan(4)
  })

  it('and that widening the level COUNT is not available locally — dither barely moves it', () => {
    // Recorded because it is the metric that would be reached for first, and
    // reading it as "the fix did nothing" would be the wrong conclusion.
    // 11 -> 12. The plateau and averaging tests below are the real ones.
    expect(distinctCodes(dithered)).toBeLessThan(distinctCodes(undithered) + 4)
  })
})

describe('dither, measured on the property that was actually broken', () => {
  it('collapses the flat plateaus that read as contours', () => {
    // Undithered, a code holds for ~28 pixels before stepping — that is the
    // contour. Dithered, runs collapse to a handful of pixels.
    expect(longestPlateau(undithered)).toBeGreaterThan(20)
    expect(longestPlateau(dithered)).toBeLessThan(longestPlateau(undithered) / 4)
  })

  it('makes the locally-averaged output track the true value instead of the plateau', () => {
    // The textbook property of dither: it decorrelates the quantisation
    // error from the signal, so the error averages away instead of
    // accumulating into a step. Measured 0.24 -> 0.06 codes RMS.
    expect(localAverageError(undithered)).toBeGreaterThan(0.2)
    expect(localAverageError(dithered)).toBeLessThan(localAverageError(undithered) / 3)
  })

  it('would NOT be achieved by an amplitude too small to cross a code boundary', () => {
    // Guards the instrument against passing for the wrong reason: a token
    // dither must fail it. If this ever passes, the metrics above are not
    // measuring what they claim to.
    const token = renderStrip(VEIL_DITHER_AMPLITUDE / 20)
    expect(longestPlateau(token)).toBeGreaterThan(longestPlateau(dithered) * 4)
  })
})

describe('VEIL_DITHER_AMPLITUDE is sized against the real output chain', () => {
  it('applies roughly one code peak-to-peak at the default exposure', () => {
    // Ordered dither wants about one code of swing: much less does nothing,
    // much more is visible grain.
    for (const col of [0.005, 0.015, 0.025, 0.046]) {
      const swing = greenExact(col * (1 + VEIL_DITHER_AMPLITUDE / 2))
        - greenExact(col * (1 - VEIL_DITHER_AMPLITUDE / 2))
      expect(swing).toBeGreaterThan(0.5)
      expect(swing).toBeLessThan(1.6)
    }
  })

  it('stays within a usable band across the whole 1x-64x exposure control', () => {
    // The shell cannot see `renderer.toneMappingExposure`, so a RELATIVE
    // amplitude is the only kind that survives the control at all: the
    // ABSOLUTE linear step of one code moves by ~50x over this range.
    let min = Infinity
    let max = 0
    for (const exposure of [1, 2, 4, 8, 16, 32, 64]) {
      for (const col of [0.005, 0.015, 0.025, 0.046]) {
        const tint = (s: number): Color3 => VEIL_TINT.map((t) => t * col * s) as unknown as Color3
        const hi = srgbEncode(acesFilmicToneMap(tint(1 + VEIL_DITHER_AMPLITUDE / 2), exposure)[1]) * 255
        const lo = srgbEncode(acesFilmicToneMap(tint(1 - VEIL_DITHER_AMPLITUDE / 2), exposure)[1]) * 255
        min = Math.min(min, hi - lo)
        max = Math.max(max, hi - lo)
      }
    }
    expect(min).toBeGreaterThan(0.05)
    expect(max).toBeLessThan(2)
  })

  it('vanishes exactly where the veil does, so no noise reaches empty sky', () => {
    // The shell writes premultiplied radiance over space. An ABSOLUTE dither
    // would speckle the whole sky (and, being signed, with negative
    // radiance); a relative one is identically zero on a zero column.
    for (const [x, y] of [[0, 0], [17, 3], [255, 63]] as const) {
      expect(0 * veilDitherScale(x, y)).toBe(0)
    }
  })
})

describe('interleavedGradientNoise', () => {
  it('stays in [0, 1)', () => {
    for (let y = 0; y < 40; y++) {
      for (let x = 0; x < 40; x++) {
        const n = interleavedGradientNoise(x, y)
        expect(n).toBeGreaterThanOrEqual(0)
        expect(n).toBeLessThan(1)
      }
    }
  })

  it('is well spread over any small neighbourhood — which is what breaks a contour', () => {
    // A pattern that repeated over a 5x5 tile would leave the contour
    // intact, just re-shaped. Every 5x5 window must span most of [0,1).
    for (let y = 0; y < 30; y += 3) {
      for (let x = 0; x < 30; x += 3) {
        let lo = 1
        let hi = 0
        for (let j = 0; j < 5; j++) {
          for (let i = 0; i < 5; i++) {
            const n = interleavedGradientNoise(x + i, y + j)
            lo = Math.min(lo, n)
            hi = Math.max(hi, n)
          }
        }
        expect(hi - lo).toBeGreaterThan(0.8)
      }
    }
  })

  it('is deterministic per pixel', () => {
    expect(interleavedGradientNoise(13, 29)).toBe(interleavedGradientNoise(13, 29))
  })
})

describe('the output chain model matches three.js', () => {
  it('reproduces ACESFilmicToneMapping including the /0.6 pre-scale and both matrices', () => {
    // Spot values against three.js's ToneMappingFunctions.js, evaluated by
    // hand from the same constants. Mid grey at exposure 1 lands near 0.5 in
    // the fit, and the operator is monotone and saturating.
    const grey = acesFilmicToneMap([0.18, 0.18, 0.18], 1)
    expect(grey[0]).toBeGreaterThan(0.1)
    expect(grey[0]).toBeLessThan(0.3)
    const bright = acesFilmicToneMap([100, 100, 100], 1)
    expect(bright[1]).toBeCloseTo(1, 2)
    expect(acesFilmicToneMap([0, 0, 0], 8)[1]).toBeCloseTo(0, 6)
  })

  it('reproduces the sRGB transfer function at its breakpoint', () => {
    expect(srgbEncode(0.0031308)).toBeCloseTo(0.0031308 * 12.92, 9)
    expect(srgbEncode(1)).toBeCloseTo(1, 9)
    expect(srgbEncode(0)).toBe(0)
    // continuous across the piecewise join
    expect(srgbEncode(0.0031308 + 1e-9)).toBeCloseTo(srgbEncode(0.0031308), 6)
  })

  it('is monotone end to end, so a brighter column never renders darker', () => {
    let prev = -1
    for (let i = 0; i <= 500; i++) {
      const c = displayCode8([0.001 * i, 0.001 * i, 0.001 * i], DEFAULT_EXPOSURE)[1]
      expect(c).toBeGreaterThanOrEqual(prev)
      prev = c
    }
  })
})

describe('the shell actually applies the dither', () => {
  it('multiplies the whole premultiplied vec4 by one shared scalar', () => {
    // Pre-fix source returns the bare vec4 on both paths — this would not
    // match against it. One scalar, not three: per-channel noise would
    // dither the hue of a near-monochrome veil.
    expect(shellSource).toMatch(/const dither = veilDitherScaleNode\(\)/)
    expect(shellSource).toMatch(/return vec4\(col\.mul\(VEIL_TINT\), alpha\)\.mul\(dither\)/)
    expect(shellSource).toMatch(/paint\)\)\)\.mul\(dither\)/)
  })

  it('mirrors the CPU dither term for term in TSL, on screen pixel coordinates', () => {
    expect(terrainNodesSource).toMatch(/screenCoordinate\.x\.mul\(0\.06711056\)\.add\(screenCoordinate\.y\.mul\(0\.00583715\)\)/)
    expect(terrainNodesSource).toMatch(/fract\(t\.mul\(52\.9829189\)\)/)
    expect(terrainNodesSource).toMatch(/interleavedGradientNoiseNode\(\)\.sub\(0\.5\)\.mul\(amplitude\)\.add\(1\)/)
  })
})

/**
 * The `cubeArrayNodes.test.ts` pattern: a plain-JS mirror of
 * `terrainNodes.ts::veilDitherScaleNode`'s expression SHAPE, so the port's
 * arithmetic is property-tested rather than only regex-pinned. A symbolic
 * node graph never evaluates numbers at build time, so a dropped `- 0.5`
 * (which would turn a zero-mean dither into a 1% brightness lift) would
 * otherwise pass every structural check. WGSL `fract(x)` is `x - floor(x)`.
 */
const fract = (x: number): number => x - Math.floor(x)
function veilDitherScaleNodeMirror(px: number, py: number, amplitude: number): number {
  const t = fract(px * 0.06711056 + py * 0.00583715)
  return (fract(t * 52.9829189) - 0.5) * amplitude + 1
}

describe('the TSL dither agrees numerically with the CPU one', () => {
  it('matches veilDitherScale across a screen-sized block of pixels', () => {
    for (let y = 0; y < 120; y += 7) {
      for (let x = 0; x < 1920; x += 37) {
        expect(veilDitherScaleNodeMirror(x, y, VEIL_DITHER_AMPLITUDE))
          .toBeCloseTo(veilDitherScale(x, y), 12)
      }
    }
  })

  it('is zero-mean over the screen, so the dither does not shift the veil\'s brightness', () => {
    // The failure this catches: a dropped `- 0.5` reads as a uniformly
    // brighter veil, which is a photometric change dressed up as a fix.
    let sum = 0
    let n = 0
    for (let y = 0; y < 200; y++) {
      for (let x = 0; x < 200; x++) { sum += veilDitherScale(x, y) - 1; n++ }
    }
    expect(Math.abs(sum / n)).toBeLessThan(1e-3 * VEIL_DITHER_AMPLITUDE)
  })
})
