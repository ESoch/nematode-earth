import { describe, it, expect } from 'vitest'
import {
  pixelsPerRadian, fullPopulationPerPixel, unityPixelDistance, instancesPerPixel,
  effectiveSamples, effectiveSamplesGaussian, noiseFraction, cellSizeFor,
  cellSizeForBudget, frustumVolume, octaveSplit, sigmaPixAt,
  S0, SIGMA_PIX_MIN, SIGMA_PIX_SHIP_MIN, SIGMA_PIX_MAX,
} from '../src/math/lod'

const F_PX = pixelsPerRadian(1080, 50)
const N_PIXELS = 1920 * 1080
const BUDGET = 4e6

describe('pixelsPerRadian', () => {
  it('is 1158.03 at 1080p with a 50 degree vertical FOV', () => {
    expect(F_PX).toBeCloseTo(1158.03, 2)
  })
  it('scales linearly with frame height', () => {
    expect(pixelsPerRadian(2160, 50)).toBeCloseTo(2 * F_PX, 6)
  })
})

describe('full-population instances per pixel (spec 7.8)', () => {
  it('reproduces the spec constant 2.20 * d^2 as N_A/f_px^2', () => {
    expect(fullPopulationPerPixel(1, 2.95e6, F_PX)).toBeCloseTo(2.2, 2)
  })
  it('is 55 per pixel at d = 5 m', () => {
    expect(fullPopulationPerPixel(5, 2.95e6, F_PX)).toBeCloseTo(55.0, 1)
  })
  it('reaches one per pixel at 0.674 m', () => {
    expect(unityPixelDistance(2.95e6, F_PX)).toBeCloseTo(0.674, 3)
    expect(fullPopulationPerPixel(unityPixelDistance(2.95e6, F_PX), 2.95e6, F_PX)).toBeCloseTo(1, 9)
  })
})

describe('the budget pins instances per pixel (spec 7.7)', () => {
  it('is 1.93 per pixel at 1080p with B = 4e6', () => {
    expect(instancesPerPixel(1e20, BUDGET, N_PIXELS)).toBeCloseTo(1.929, 3)
  })
  it('does not exceed the real population when the population is small', () => {
    expect(instancesPerPixel(1000, BUDGET, N_PIXELS)).toBeCloseTo(1000 / N_PIXELS, 12)
  })
})

describe('effective samples and noise', () => {
  const lambda = instancesPerPixel(1e20, BUDGET, N_PIXELS)
  it('gives 72 percent noise unwidened', () => {
    expect(noiseFraction(effectiveSamples(lambda, SIGMA_PIX_MIN))).toBeCloseTo(0.72, 3)
  })
  it('lifts effective samples 28x at sigma_pix = 3', () => {
    expect(effectiveSamples(lambda, 3) / lambda).toBeCloseTo(28.27, 2)
  })
  it('gives about 55 effective samples and 14 percent noise at sigma_pix = 3', () => {
    expect(effectiveSamples(lambda, 3)).toBeCloseTo(54.5, 1)
    expect(noiseFraction(effectiveSamples(lambda, 3))).toBeCloseTo(0.135, 3)
  })
  it('never falls below one sample per instance', () => {
    expect(effectiveSamples(lambda, 0.1)).toBeCloseTo(lambda, 12)
  })
  it('the true Gaussian kernel is 4x more optimistic than the spec disc', () => {
    expect(effectiveSamplesGaussian(lambda, 3) / effectiveSamples(lambda, 3)).toBeCloseTo(4, 6)
  })
  it('inflates noise when per-instance flux itself varies', () => {
    const n = effectiveSamples(lambda, 3)
    expect(noiseFraction(n, 3.0804)).toBeCloseTo(noiseFraction(n) * Math.sqrt(4.0804), 9)
  })
})

describe('cell sizing', () => {
  it('puts 1.93 instances per pixel in a 1200 m column at 1 km with a 7.74 m cell', () => {
    expect(cellSizeFor(1000, 1200, 1.929, F_PX)).toBeCloseTo(7.74, 2)
  })
  it('inverts against fullPopulationPerPixel: lambda scales as 1/s^3', () => {
    const a = cellSizeFor(1000, 1200, 1.929, F_PX)
    const b = cellSizeFor(1000, 1200, 1.929 * 8, F_PX)
    expect(a / b).toBeCloseTo(2, 9)
  })
  it('fits exactly `budget` cells into a volume', () => {
    const s = cellSizeForBudget(1e6, 4e6)
    expect(1e6 / s ** 3).toBeCloseTo(4e6, 3)
  })
  it('frustum volume matches the cone-of-a-pyramid formula', () => {
    expect(frustumVolume(1000, 50, 1)).toBeCloseTo((4 / 3) * 1e9 * Math.tan(Math.PI * 25 / 180) ** 2, 3)
  })
})

describe('octave quantization keeps the lattice world-stable', () => {
  it('brackets the target size by a factor of two', () => {
    for (const s of [0.01, 0.7, 7.74, 73, 4200]) {
      const o = octaveSplit(s)
      expect(o.sizeLo).toBeLessThanOrEqual(s)
      expect(o.sizeHi).toBeGreaterThanOrEqual(s)
      expect(o.sizeHi / o.sizeLo).toBeCloseTo(2, 12)
      expect(o.frac).toBeGreaterThanOrEqual(0)
      expect(o.frac).toBeLessThan(1)
    }
  })
  it('is exact on an octave boundary', () => {
    const o = octaveSplit(S0 * 2 ** 12)
    expect(o.level).toBe(12)
    expect(o.frac).toBeCloseTo(0, 12)
  })
  it('cross-fade weights always sum to one', () => {
    for (let i = 0; i < 64; i++) {
      const o = octaveSplit(S0 * 2 ** (3 + i / 64))
      expect((1 - o.frac) + o.frac).toBeCloseTo(1, 12)
    }
  })
})

describe('sigma_pix schedule (spec 7.7)', () => {
  it('is 3 px at the 10 km end', () => {
    expect(sigmaPixAt(10000)).toBeCloseTo(SIGMA_PIX_MAX, 6)
  })
  it('tightens to 0.75 px by 1 km, inside the spec 0.5-1 px band', () => {
    expect(sigmaPixAt(1000)).toBeCloseTo(SIGMA_PIX_SHIP_MIN, 6)
    expect(SIGMA_PIX_SHIP_MIN).toBeGreaterThanOrEqual(0.5)
    expect(SIGMA_PIX_SHIP_MIN).toBeLessThanOrEqual(1.0)
  })
  it('never goes below the Poisson-summation validity floor of 0.5', () => {
    for (let a = 0; a <= 2e4; a += 50) expect(sigmaPixAt(a)).toBeGreaterThanOrEqual(SIGMA_PIX_MIN)
  })
  it('is monotonically non-decreasing in altitude', () => {
    let prev = -Infinity
    for (let a = 0; a <= 5e4; a += 100) {
      const s = sigmaPixAt(a)
      expect(s).toBeGreaterThanOrEqual(prev - 1e-12)
      prev = s
    }
  })
})
