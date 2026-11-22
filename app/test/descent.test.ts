import { describe, it, expect } from 'vitest'
import {
  H_S_MAX, H_S_MIN, K_SHIP, K_MAX, EVEREST_M,
  scaleHeightAt, exaggerationAt, renderedTau, descentAt,
} from '../src/math/descent'
import { tauLimb } from '../src/math/optical'

describe('constants', () => {
  it('ships k = 3.0 with a hard maximum of 3.6, never 10', () => {
    expect(K_SHIP).toBe(3.0)
    expect(K_MAX).toBe(3.6)
    expect(K_MAX).toBeLessThan(H_S_MAX / EVEREST_M)
  })

  it('floors the scale height at 200 m for f32 conditioning', () => {
    expect(H_S_MIN).toBe(200)
  })
})

describe('scaleHeightAt', () => {
  it('reaches the full 32 km at and above 192 km altitude', () => {
    expect(scaleHeightAt(192000)).toBeCloseTo(H_S_MAX, 6)
    expect(scaleHeightAt(1e7)).toBeCloseTo(H_S_MAX, 6)
  })

  it('follows alt/6 in the middle of the range', () => {
    expect(scaleHeightAt(60000)).toBeCloseTo(10000, 6)
  })

  it('never goes below the 200 m floor', () => {
    expect(scaleHeightAt(0)).toBe(H_S_MIN)
    expect(scaleHeightAt(100)).toBe(H_S_MIN)
  })

  it('is monotonically non-decreasing in altitude', () => {
    let prev = -Infinity
    for (let a = 0; a <= 3e5; a += 1000) {
      const h = scaleHeightAt(a)
      expect(h).toBeGreaterThanOrEqual(prev)
      prev = h
    }
  })
})

describe('exaggerationAt', () => {
  it('is 3.6 at orbital altitude, capped by K_MAX', () => {
    expect(exaggerationAt(1e7)).toBeCloseTo(K_MAX, 6)
  })

  it('converges to 1.0 near the surface', () => {
    expect(exaggerationAt(0)).toBeCloseTo(1.0, 6)
  })

  it('never lets EXAGGERATED Everest exceed one e-folding of the shell', () => {
    // Vacuous below the crossover (alt = 6*EVEREST_M = 53094 m), where k is floored
    // at 1.0 and no exaggeration is applied at all.
    for (let a = 0; a <= 5e5; a += 2500) {
      const k = exaggerationAt(a)
      if (k > 1.0) {
        expect(k * EVEREST_M).toBeLessThanOrEqual(scaleHeightAt(a) * 1.000001)
      }
    }
  })

  it('applies no exaggeration at all below the crossover altitude', () => {
    const crossover = 6 * EVEREST_M // 53094 m
    for (let a = 0; a < crossover; a += 1000) {
      expect(exaggerationAt(a)).toBe(1.0)
    }
    expect(exaggerationAt(crossover * 1.5)).toBeGreaterThan(1.0)
  })

  it('rejects the 10x proposal by construction', () => {
    for (let a = 0; a <= 1e7; a += 50000) {
      expect(exaggerationAt(a)).toBeLessThan(10)
    }
  })
})

describe('renderedTau', () => {
  it('passes the global mean through unclamped at full shell', () => {
    expect(renderedTau(0.047, H_S_MAX)).toBeCloseTo(0.047, 6)
  })

  it('clamps the richest pixels only once the shell thins', () => {
    expect(renderedTau(0.19, H_S_MAX)).toBeCloseTo(0.1131, 4)
  })

  it('keeps tau_limb at or below 4 at every altitude', () => {
    for (let a = 1000; a <= 5e5; a += 2000) {
      const Hs = scaleHeightAt(a)
      expect(tauLimb(renderedTau(0.19, Hs), Hs)).toBeLessThanOrEqual(4.0000001)
    }
  })

  it('prevents the naive ramp from driving tau_limb to 35', () => {
    const Hs = scaleHeightAt(10000)
    expect(tauLimb(0.19, Hs)).toBeGreaterThan(10)
    expect(tauLimb(renderedTau(0.19, Hs), Hs)).toBeLessThanOrEqual(4.0000001)
  })
})

describe('descentAt', () => {
  it('bundles a self-consistent state', () => {
    const s = descentAt(192000, 0.047)
    expect(s.Hs).toBeCloseTo(H_S_MAX, 6)
    expect(s.Hmax).toBeCloseTo(6 * H_S_MAX, 6)
    expect(s.tauRendered).toBeCloseTo(0.047, 6)
    expect(s.tauLimb).toBeCloseTo(1.662, 3)
    expect(s.k).toBeCloseTo(K_MAX, 6)
  })

  it('always satisfies Hmax = 6 * Hs so truncation stays under 0.5 percent', () => {
    for (let a = 0; a <= 1e6; a += 5000) {
      const s = descentAt(a, 0.05)
      expect(s.Hmax / s.Hs).toBeCloseTo(6, 9)
    }
  })
})
