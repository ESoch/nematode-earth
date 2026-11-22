import { describe, it, expect } from 'vitest'
import {
  syntheticTau, syntheticStats, TAU_DESERT, TAU_MEAN, TAU_TUNDRA, TAU_RICH,
} from '../src/field/synthetic'

const dirFromLat = (latDeg: number): [number, number, number] => {
  const p = (latDeg * Math.PI) / 180
  return [Math.cos(p), 0, Math.sin(p)]
}

describe('synthetic field range', () => {
  it('exposes the biome anchors from the spec', () => {
    expect(TAU_DESERT).toBeCloseTo(0.0024, 5)
    expect(TAU_MEAN).toBeCloseTo(0.047, 5)
    expect(TAU_TUNDRA).toBeCloseTo(0.12, 5)
    expect(TAU_RICH).toBeCloseTo(0.19, 5)
  })

  it('stays within the physical range everywhere', () => {
    const s = syntheticStats(20000)
    expect(s.min).toBeGreaterThan(0)
    expect(s.max).toBeLessThanOrEqual(TAU_RICH * 1.0001)
  })

  it('reproduces the 27x coverage dynamic range', () => {
    const s = syntheticStats(20000)
    expect(s.coverageRatio).toBeGreaterThan(20)
  })
})

describe('latitudinal structure', () => {
  it('is brighter at high latitude than at the equator', () => {
    const [ex, ey, ez] = dirFromLat(0)
    const [hx, hy, hz] = dirFromLat(68)
    expect(syntheticTau(hx, hy, hz)).toBeGreaterThan(syntheticTau(ex, ey, ez))
  })

  it('shows polar brightening in both hemispheres, not just the north', () => {
    const zonalMean = (latDeg: number) => {
      let sum = 0
      const N = 360
      for (let i = 0; i < N; i++) {
        const lon = (i / N) * 2 * Math.PI
        const p = (latDeg * Math.PI) / 180
        sum += syntheticTau(Math.cos(p) * Math.cos(lon), Math.cos(p) * Math.sin(lon), Math.sin(p))
      }
      return sum / N
    }
    const tropics = zonalMean(5)
    expect(zonalMean(68)).toBeGreaterThan(tropics)
    expect(zonalMean(-68)).toBeGreaterThan(tropics)
  })

  it('inverts the naive expectation: poles beat tropics', () => {
    const [tx, ty, tz] = dirFromLat(10)
    const [px, py, pz] = dirFromLat(75)
    expect(syntheticTau(px, py, pz)).toBeGreaterThan(syntheticTau(tx, ty, tz))
  })
})

describe('determinism and normalization', () => {
  it('is deterministic for a given direction', () => {
    expect(syntheticTau(0.3, 0.5, 0.81)).toBe(syntheticTau(0.3, 0.5, 0.81))
  })

  it('normalizes non-unit directions', () => {
    const a = syntheticTau(1, 0, 0)
    const b = syntheticTau(7, 0, 0)
    expect(a).toBeCloseTo(b, 12)
  })

  it('has a global mean in the right neighbourhood', () => {
    const s = syntheticStats(50000)
    expect(s.mean).toBeGreaterThan(0.02)
    expect(s.mean).toBeLessThan(0.09)
  })
})
