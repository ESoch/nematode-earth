import { describe, expect, it } from 'vitest'
import { supportedAreaMean } from '../src/field/densityCoverage'

describe('all-cell density and independent valid-model coverage', () => {
  it('does not dilute a half-covered cell for a second time', () => {
    // Half the equal-area source cells have 100/m², half have no data.
    const coarseMean = (100 + 0) / 2
    expect(supportedAreaMean(coarseMean, 0.5)).toBe(50)
  })
  it('retains fully covered density and measured zero', () => {
    expect(supportedAreaMean(100, 1)).toBe(100)
    expect(supportedAreaMean(0, 1)).toBe(0)
  })
  it('suppresses any sentinel over unsupported cells', () => {
    for (const sentinel of [1, 100, 1e6]) expect(supportedAreaMean(sentinel, 0)).toBe(0)
  })
})
