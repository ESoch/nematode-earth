import { describe, it, expect } from 'vitest'
import { FrameTimer } from '../src/core/Timing'

describe('FrameTimer', () => {
  it('reports NaN before any samples arrive', () => {
    expect(Number.isNaN(new FrameTimer().median)).toBe(true)
  })

  it('computes the median of collected samples', () => {
    const t = new FrameTimer()
    for (const v of [1, 2, 3, 4, 5]) t.push(v)
    expect(t.median).toBeCloseTo(3, 9)
  })

  it('computes p95', () => {
    const t = new FrameTimer()
    for (let i = 1; i <= 100; i++) t.push(i)
    expect(t.p95).toBeGreaterThanOrEqual(95)
    expect(t.p95).toBeLessThanOrEqual(96)
  })

  it('survives Chrome 100us quantization by taking the median across frames', () => {
    // Chrome quantizes timestamps to 100 us (0.1 ms). FrameTimer takes a
    // MEDIAN, not an average - the samples below (truth 2.03 plus a 0-6e-2 ms
    // ramp, each rounded to the nearest 0.1 ms) fall into only two adjacent
    // buckets, 2.0 ms (~29% of samples) and 2.1 ms (~71%), so the true median
    // is 2.1 ms. The ±0.05 ms band below is half the 0.1 ms quantization
    // step - tight enough that it fails for a broken implementation that
    // returns e.g. the first sample (buf[0], which quantizes to 2.0).
    const t = new FrameTimer()
    const truth = 2.03
    for (let i = 0; i < 400; i++) {
      t.push(Math.round((truth + (i % 7) * 0.01) / 0.1) * 0.1)
    }
    expect(t.median).toBeGreaterThan(2.05)
    expect(t.median).toBeLessThan(2.15)
  })

  it('bounds memory by discarding the oldest samples', () => {
    const t = new FrameTimer(64)
    for (let i = 0; i < 1000; i++) t.push(i)
    expect(t.samples).toBe(64)
  })

  it('ignores non-finite samples', () => {
    const t = new FrameTimer()
    t.push(NaN)
    t.push(Infinity)
    t.push(2)
    expect(t.samples).toBe(1)
    expect(t.median).toBeCloseTo(2, 9)
  })

  it('computes p05', () => {
    const t = new FrameTimer()
    for (let i = 1; i <= 100; i++) t.push(i)
    expect(t.p05).toBeGreaterThanOrEqual(5)
    expect(t.p05).toBeLessThanOrEqual(6)
  })

  it('reports NaN stddev before any samples arrive', () => {
    expect(Number.isNaN(new FrameTimer().stddev)).toBe(true)
  })

  it('computes stddev of collected samples', () => {
    const t = new FrameTimer()
    // Population stddev of [2,4,4,4,5,5,7,9] is exactly 2.
    for (const v of [2, 4, 4, 4, 5, 5, 7, 9]) t.push(v)
    expect(t.stddev).toBeCloseTo(2, 9)
  })

  it('reports zero stddev for a single sample or identical samples', () => {
    const t = new FrameTimer()
    for (let i = 0; i < 5; i++) t.push(3)
    expect(t.stddev).toBeCloseTo(0, 9)
  })
})
