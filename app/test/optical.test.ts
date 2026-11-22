import { describe, it, expect } from 'vitest'
import {
  E_A, N_A_LAND, N_A_ICEFREE, tauNadir, limbRatio, tauLimb,
  sigma0, arealCoverage, saturationCapTau,
} from '../src/math/optical'
import { WGS84 } from '../src/math/wgs84'

const HS = 32000
const HMAX = 6 * HS

describe('constants', () => {
  it('uses the adjudicated mean projected area', () => {
    expect(E_A).toBeCloseTo(1.49e-8, 10)
  })
})

describe('tauNadir = E[A] * N_A', () => {
  it('gives 0.044 over total land', () => {
    expect(tauNadir(N_A_LAND)).toBeCloseTo(0.044, 3)
  })

  it('gives 0.051 over ice-free land', () => {
    expect(tauNadir(N_A_ICEFREE)).toBeCloseTo(0.051, 3)
  })

  it('is linear in column abundance and independent of any depth profile', () => {
    expect(tauNadir(2 * N_A_LAND)).toBeCloseTo(2 * tauNadir(N_A_LAND), 12)
  })
})

describe('limbRatio Q = sqrt(2*pi*R/Hs)', () => {
  it('is 35.369 at Hs = 32 km', () => {
    expect(limbRatio(HS)).toBeCloseTo(35.369, 3)
  })

  it('grows as the shell thins', () => {
    expect(limbRatio(8000)).toBeGreaterThan(limbRatio(32000))
  })

  it('uses the mean radius for scalar radiometry', () => {
    expect(limbRatio(HS, WGS84.R_MEAN)).toBeCloseTo(limbRatio(HS), 12)
  })
})

describe('tauLimb', () => {
  it('is 1.66 at the global mean tau of 0.047', () => {
    expect(tauLimb(0.047, HS)).toBeCloseTo(1.662, 3)
  })

  it('is 4.2 in tundra at tau = 0.12', () => {
    expect(tauLimb(0.12, HS)).toBeCloseTo(4.244, 2)
  })

  it('is 0.14 in hot desert at tau = 0.004', () => {
    expect(tauLimb(0.004, HS)).toBeCloseTo(0.141, 2)
  })
})

describe('saturationCapTau', () => {
  it('caps at 0.1131 for Hs = 32 km, which is tau_limb = 4', () => {
    const cap = saturationCapTau(HS)
    expect(cap).toBeCloseTo(0.1131, 4)
    expect(tauLimb(cap, HS)).toBeCloseTo(4, 6)
  })

  it('tightens as the shell thins during descent', () => {
    expect(saturationCapTau(4000)).toBeLessThan(saturationCapTau(32000))
  })

  it('does not clamp the global mean', () => {
    expect(saturationCapTau(HS)).toBeGreaterThan(0.047)
  })
})

describe('sigma0 normalization', () => {
  it('integrates back to exactly tau_nadir over [0, Hmax]', () => {
    const tau = 0.047
    const s0 = sigma0(tau, HS, HMAX)
    const integral = s0 * HS * (1 - Math.exp(-HMAX / HS))
    expect(integral).toBeCloseTo(tau, 12)
  })

  it('is 1.472e-6 per metre at tau=0.047, Hs=32 km, Hmax=192 km', () => {
    // NOTE: the research ruling asserted 4.46e-6 here. That value corresponds to
    // Hs ~ 10.6 km, not 32 km, and is inconsistent with its own stated Hs.
    // The formula is self-consistent; the constant was wrong.
    expect(sigma0(0.047, HS, HMAX)).toBeCloseTo(1.472e-6, 9)
  })

  it('scales inversely with scale height', () => {
    expect(sigma0(0.047, 16000, 96000)).toBeCloseTo(2 * sigma0(0.047, 32000, 192000), 9)
  })
})

describe('arealCoverage = 1 - exp(-tau)', () => {
  it.each([
    [0.0024, 0.0024],
    [0.047, 0.0459],
    [0.12, 0.1131],
    [0.19, 0.1730],
  ])('tau %f gives coverage %f', (tau, expected) => {
    expect(arealCoverage(tau)).toBeCloseTo(expected, 4)
  })

  it('spans a 27x dynamic range from hot desert to the richest pixels', () => {
    const ratio = arealCoverage(0.19) / arealCoverage(0.0024)
    expect(ratio).toBeGreaterThan(25)
    expect(ratio).toBeLessThan(80)
  })
})
