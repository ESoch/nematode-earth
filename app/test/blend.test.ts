import { describe, it, expect } from 'vitest'
import {
  REGIME2_TOP_ALT, REGIME2_FULL_ALT, handoffWeight, shellShare, instanceShare,
  windowTaper, partitionAt,
} from '../src/regimes/Blend'
import { saturationCapTau, E_A, sigma0 } from '../src/math/optical'
import { scaleHeightAt } from '../src/math/descent'
import { drawAggregateArea } from '../src/math/crossSection'
import { pixelsPerRadian } from '../src/math/lod'

describe('the regime-2 band is DERIVED from the tau cap, not chosen', () => {
  it('opens exactly where the cap starts discarding the global-mean tau', () => {
    expect(saturationCapTau(scaleHeightAt(REGIME2_TOP_ALT))).toBeCloseTo(0.047, 4)
    expect(REGIME2_TOP_ALT).toBeCloseTo(33160, 0)
  })
  it('completes one decade lower', () => {
    expect(REGIME2_FULL_ALT).toBeCloseTo(REGIME2_TOP_ALT / 10, 6)
  })
})

describe('handoffWeight', () => {
  it('is 0 at and above the top of the band', () => {
    expect(handoffWeight(REGIME2_TOP_ALT)).toBeCloseTo(0, 9)
    expect(handoffWeight(1e7)).toBe(0)
  })
  it('is 1 at and below the bottom of the band', () => {
    expect(handoffWeight(REGIME2_FULL_ALT)).toBeCloseTo(1, 9)
    expect(handoffWeight(0)).toBe(1)
  })
  it('is monotonically non-increasing in altitude', () => {
    let prev = Infinity
    for (let a = 0; a <= 5e4; a += 100) {
      const w = handoffWeight(a)
      expect(w).toBeLessThanOrEqual(prev + 1e-12)
      prev = w
    }
  })
})

describe('the partition conserves energy BY CONSTRUCTION', () => {
  it('shell + instance is exactly 1 at every altitude and tau', () => {
    for (let a = 0; a <= 4e4; a += 250) {
      for (const tau of [0.0024, 0.047, 0.12, 0.19]) {
        const p = partitionAt(a, tau)
        expect(p.shell + p.inst).toBeCloseTo(1, 12)
      }
    }
  })
  it('hands the instances nothing when w=0 and the cap does not bite', () => {
    expect(instanceShare(0, 0.047, 0.047)).toBeCloseTo(0, 12)
  })
  it('hands the instances everything at w=1', () => {
    expect(instanceShare(1, 0.02, 0.19)).toBeCloseTo(1, 12)
  })
  it('hands the instances the energy the cap takes off the shell (spec 7.3)', () => {
    // At 10 km the cap keeps only ~54.9 percent of the global-mean tau.
    const p = partitionAt(10000, 0.047)
    expect(p.r).toBeCloseTo(0.549, 3)
    expect(p.inst).toBeGreaterThan(p.w)
  })
  it('never produces a negative or over-unity share', () => {
    for (let a = 0; a <= 4e4; a += 137) {
      const p = partitionAt(a, 0.19)
      expect(p.shell).toBeGreaterThanOrEqual(0)
      expect(p.inst).toBeGreaterThanOrEqual(0)
      expect(p.inst).toBeLessThanOrEqual(1)
    }
  })
})

describe('windowTaper hides the emission window boundary', () => {
  it('is 1 inside, 0 outside, and C1 at both ends', () => {
    expect(windowTaper(0, 100, 200)).toBe(1)
    expect(windowTaper(100, 100, 200)).toBeCloseTo(1, 9)
    expect(windowTaper(200, 100, 200)).toBeCloseTo(0, 9)
    expect(windowTaper(1e6, 100, 200)).toBe(0)
    expect(windowTaper(150, 100, 200)).toBeCloseTo(0.5, 6)
  })
})

describe('Monte-Carlo: the instance formula reproduces w*tau in expectation', () => {
  it('sums to the analytic instance share within 3 sigma', () => {
    // One pixel column at nadir. Cells of edge s along the ray, each emitting
    // one instance of flux share*L*A_i*f_px^2/d^2. Total must be share*tau*L.
    const fPx = pixelsPerRadian(1080, 50)
    const Hs = 200
    const Hmax = 6 * Hs
    const tau = 0.047
    const s0 = sigma0(tau, Hs, Hmax)
    const share = 0.6
    const L = 1
    const camAlt = 1000
    const s = 2
    let acc = 0
    let u = 0.1234
    const rnd = (): number => { u = (u * 9301 + 0.49297) % 1; return u }
    for (let z = 0; z < Hmax; z += s) {
      const d = camAlt - z
      if (d <= s) break
      const sigma = s0 * Math.exp(-z / Hs)
      const cone = (d / fPx) ** 2
      const cellsAcross = cone / (s * s) // fractional cells in the cone
      const omega = (sigma * s * s * s) / E_A
      const A = drawAggregateArea(omega, rnd(), rnd())
      acc += share * L * A * (fPx * fPx) / (d * d) * cellsAcross
    }
    const expected = share * L * s0 * Hs * (1 - Math.exp(-Hmax / Hs))
    expect(acc / expected).toBeGreaterThan(0.9)
    expect(acc / expected).toBeLessThan(1.1)
  })
})
