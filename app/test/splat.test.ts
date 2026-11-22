import { describe, it, expect } from 'vitest'
import {
  bilinearWeights, gaussianTaps, discreteGaussSum, discreteGaussSumClosed,
  depositThenBlurEnergy, MAX_TAPS,
} from '../src/math/splat'

describe('the 2x2 deposit is normalized in the DISCRETE pixel domain', () => {
  it('sums to exactly 1 at every subpixel position', () => {
    for (let i = 0; i < 64; i++) {
      for (let j = 0; j < 64; j++) {
        const w = bilinearWeights(i / 64, j / 64)
        const sum = w[0] + w[1] + w[2] + w[3]
        expect(Math.abs(sum - 1)).toBeLessThan(1e-15)
      }
    }
  })
  it('is continuous across a pixel boundary - the anti-scintillation property', () => {
    const a = bilinearWeights(0.999999, 0.4)
    const b = bilinearWeights(0.0, 0.4)
    // At the boundary all the energy has migrated to the next pixel, and it
    // did so continuously: the vacated weight is ~0, not a jump.
    expect(a[1]! + a[3]!).toBeCloseTo(b[0]! + b[2]!, 5)
  })
  it('carries zero weight exactly at the edge of its 2x2 support', () => {
    const w = bilinearWeights(0, 0)
    expect(w[1]).toBeCloseTo(0, 15)
    expect(w[2]).toBeCloseTo(0, 15)
    expect(w[3]).toBeCloseTo(0, 15)
    expect(w[0]).toBeCloseTo(1, 15)
  })
})

describe('separable Gaussian resolve taps', () => {
  it.each([[0.5], [0.75], [1], [2], [3]])('sigma=%f taps sum to 1', (s) => {
    const t = gaussianTaps(s)
    expect(t.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
  it('is symmetric and odd length', () => {
    const t = gaussianTaps(2)
    expect(t.length % 2).toBe(1)
    for (let i = 0; i < t.length; i++) expect(t[i]).toBeCloseTo(t[t.length - 1 - i]!, 15)
  })
  it('uses radius ceil(3*sigma) and never exceeds MAX_TAPS', () => {
    expect(gaussianTaps(3).length).toBe(2 * 9 + 1)
    expect(gaussianTaps(3).length).toBeLessThanOrEqual(MAX_TAPS)
    expect(gaussianTaps(20).length).toBe(MAX_TAPS)
  })
})

describe('closed-form discrete lattice sum (Poisson summation)', () => {
  it.each([[0.5], [0.75], [1], [1.5], [3], [4]])(
    'matches brute force at sigma=%f for every subpixel phase', (s) => {
      let worst = 0
      for (let i = 0; i < 128; i++) {
        const f = i / 128
        const rel = Math.abs(discreteGaussSumClosed(f, s) / discreteGaussSum(f, s) - 1)
        worst = Math.max(worst, rel)
      }
      expect(worst).toBeLessThan(1e-6)
    })
  it('shows why 0.5 is the floor: the ripple explodes below it', () => {
    const rip = (s: number): number => 2 * Math.exp(-8 * Math.PI * Math.PI * s * s)
    expect(rip(0.5)).toBeLessThan(1e-8)
    expect(rip(0.25)).toBeGreaterThan(1e-2)
  })
})

describe('deposit then blur conserves energy exactly', () => {
  it.each([[0.75], [1], [3]])(
    'sigma=%f: total energy is 1 for every subpixel offset', (s) => {
      const taps = gaussianTaps(s)
      for (let i = 0; i < 32; i++) {
        for (let j = 0; j < 32; j++) {
          const e = depositThenBlurEnergy(10 + i / 32, 10 + j / 32, taps)
          expect(Math.abs(e - 1)).toBeLessThan(1e-12)
        }
      }
    })
})
