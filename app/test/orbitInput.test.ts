import { describe, it, expect } from 'vitest'
import { dragToOrbit, wheelDeltaToTicks, projectedUp, autoRotateDeltaLon } from '../src/core/orbitInput'

describe('dragToOrbit', () => {
  it('produces zero orbit for zero delta', () => {
    // -0 (from -dyPx*scale with dyPx=0) is mathematically 0; toBeCloseTo
    // treats it as such, unlike toEqual's Object.is-based comparison.
    const { dLon, dLat } = dragToOrbit(0, 0, 1000)
    expect(dLon).toBeCloseTo(0, 12)
    expect(dLat).toBeCloseTo(0, 12)
  })

  it('increases longitude when dragging right', () => {
    const { dLon } = dragToOrbit(100, 0, 1000)
    expect(dLon).toBeGreaterThan(0)
  })

  it('decreases longitude when dragging left', () => {
    const { dLon } = dragToOrbit(-100, 0, 1000)
    expect(dLon).toBeLessThan(0)
  })

  it('increases latitude when dragging up (negative screen-space dy)', () => {
    const { dLat } = dragToOrbit(0, -100, 1000)
    expect(dLat).toBeGreaterThan(0)
  })

  it('decreases latitude when dragging down (positive screen-space dy)', () => {
    const { dLat } = dragToOrbit(0, 100, 1000)
    expect(dLat).toBeLessThan(0)
  })

  it('scales inversely with viewport size, so the same physical drag orbits by the same angle', () => {
    const small = dragToOrbit(50, 0, 500)
    const large = dragToOrbit(100, 0, 1000)
    expect(small.dLon).toBeCloseTo(large.dLon, 12)
  })

  it('is safe against a non-positive viewport size', () => {
    expect(dragToOrbit(50, 50, 0)).toEqual({ dLon: 0, dLat: 0 })
    expect(dragToOrbit(50, 50, -10)).toEqual({ dLon: 0, dLat: 0 })
  })
})

describe('wheelDeltaToTicks', () => {
  it('maps scroll-up (negative deltaY) to positive (descending) ticks', () => {
    expect(wheelDeltaToTicks(-100)).toBeGreaterThan(0)
  })

  it('maps scroll-down (positive deltaY) to negative (ascending) ticks', () => {
    expect(wheelDeltaToTicks(100)).toBeLessThan(0)
  })

  it('is zero for zero delta', () => {
    expect(wheelDeltaToTicks(0)).toBeCloseTo(0, 12)
  })

  it('scales with pxPerTick', () => {
    expect(wheelDeltaToTicks(-200, 100)).toBeCloseTo(2, 9)
    expect(wheelDeltaToTicks(-200, 200)).toBeCloseTo(1, 9)
  })
})

describe('projectedUp', () => {
  it('returns straight-up (polar axis) for a direction ON the equator', () => {
    expect(projectedUp([1, 0, 0])).toEqual([0, 0, 1])
    expect(projectedUp([0, 1, 0])).toEqual([0, 0, 1])
    expect(projectedUp([1, 1, 0])).toEqual([0, 0, 1])
  })

  it('is unit length for a range of directions', () => {
    const dirs: Array<[number, number, number]> = [
      [1, 0, 0], [0, 1, 0], [1, 1, 1], [0.1, 0.2, 0.9], [-1, -2, 0.5],
    ]
    for (const d of dirs) {
      const up = projectedUp(d)
      expect(Math.hypot(...up)).toBeCloseTo(1, 9)
    }
  })

  it('is orthogonal to the view direction (tangent-plane property)', () => {
    const dirs: Array<[number, number, number]> = [
      [1, 0, 0], [1, 1, 1], [0.1, 0.2, 0.9], [-1, -2, 0.5], [0.01, 0.01, 0.999],
    ]
    for (const d of dirs) {
      const n = Math.hypot(...d)
      const dn: [number, number, number] = [d[0] / n, d[1] / n, d[2] / n]
      const up = projectedUp(d)
      const dot = dn[0] * up[0] + dn[1] * up[1] + dn[2] * up[2]
      expect(Math.abs(dot)).toBeLessThan(1e-9)
    }
  })

  it('stays continuous and horizontal-ish as the direction approaches a pole', () => {
    // Just shy of the +Z pole (ScaleCamera itself never reaches closer than
    // pi/2 - 1e-6 in latitude, i.e. cos(lat) ~ 1e-6 here).
    const up = projectedUp([1e-6, 0, 1])
    expect(Number.isFinite(up[0])).toBe(true)
    expect(Number.isFinite(up[1])).toBe(true)
    expect(Number.isFinite(up[2])).toBe(true)
    expect(Math.hypot(...up)).toBeCloseTo(1, 6)
  })

  it('falls back to the polar axis for a literal pole vector (unreachable via ScaleCamera, still well-defined)', () => {
    expect(projectedUp([0, 0, 1])).toEqual([0, 0, 1])
    expect(projectedUp([0, 0, -1])).toEqual([0, 0, 1])
  })

  it('respects a custom polar axis', () => {
    expect(projectedUp([1, 0, 0], [0, 1, 0])).toEqual([0, 1, 0])
  })
})

describe('autoRotateDeltaLon', () => {
  it('is proportional to elapsed time', () => {
    expect(autoRotateDeltaLon(2, 0.1)).toBeCloseTo(0.2, 12)
    expect(autoRotateDeltaLon(4, 0.1)).toBeCloseTo(0.4, 12)
  })

  it('is zero for zero elapsed time', () => {
    expect(autoRotateDeltaLon(0, 0.1)).toBeCloseTo(0, 12)
  })

  it('accumulates the same total rotation regardless of frame rate (frame-rate independence)', () => {
    // The bug this pins: a fixed per-frame increment applied once per
    // `requestAnimationFrame` callback spins the globe faster at a higher
    // refresh rate, because more callbacks fire per second of wall time.
    // A dt-scaled delta must sum to the same total at any frame rate.
    const durationSec = 8
    const sumSteps = (hz: number) => {
      const dt = 1 / hz
      const frames = Math.round(durationSec * hz)
      let total = 0
      for (let i = 0; i < frames; i++) total += autoRotateDeltaLon(dt)
      return total
    }
    const at60Hz = sumSteps(60)
    const at120Hz = sumSteps(120)
    expect(at120Hz).toBeCloseTo(at60Hz, 9)
  })
})
