import { describe, it, expect } from 'vitest'
import { cobbMultiplier, dataMultiplier, densityMultiplier } from '../src/field/cobb'

describe('cobbMultiplier', () => {
  it('is 1 (no change) at zero built-up fraction', () => {
    expect(cobbMultiplier(0)).toBe(1)
  })

  it('is 4 (1914 prediction: 4x) at fully built-up', () => {
    expect(cobbMultiplier(1)).toBe(4)
  })

  it('is linear in built-up fraction', () => {
    expect(cobbMultiplier(0.5)).toBeCloseTo(2.5, 10)
  })

  it('clamps out-of-range input', () => {
    expect(cobbMultiplier(-1)).toBe(1)
    expect(cobbMultiplier(2)).toBe(4)
  })
})

describe('dataMultiplier', () => {
  it('is exactly 1 regardless of built-up fraction — the corrected null result, not a curve', () => {
    for (const u of [0, 0.1, 0.5, 0.9, 1]) {
      expect(dataMultiplier(u)).toBe(1)
    }
  })
})

describe('densityMultiplier', () => {
  it('equals dataMultiplier at cobbMix = 0', () => {
    for (const u of [0, 0.3, 1]) {
      expect(densityMultiplier(u, 0)).toBeCloseTo(dataMultiplier(u), 10)
    }
  })

  it('equals cobbMultiplier at cobbMix = 1', () => {
    for (const u of [0, 0.3, 1]) {
      expect(densityMultiplier(u, 1)).toBeCloseTo(cobbMultiplier(u), 10)
    }
  })

  it('is exactly 1 at cobbMix = 0 even over a fully built-up pixel — dragging toward "data" makes cities stop mattering, not invert', () => {
    expect(densityMultiplier(1, 0)).toBe(1)
  })

  it('interpolates linearly in cobbMix for a fixed built-up fraction', () => {
    const u = 1 // cobbMultiplier(1) = 4, dataMultiplier(1) = 1
    expect(densityMultiplier(u, 0.5)).toBeCloseTo(2.5, 10)
  })

  it('clamps cobbMix to [0,1]', () => {
    expect(densityMultiplier(1, -1)).toBeCloseTo(dataMultiplier(1), 10)
    expect(densityMultiplier(1, 2)).toBeCloseTo(cobbMultiplier(1), 10)
  })
})
