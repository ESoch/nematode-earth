import { describe, it, expect } from 'vitest'
import { CELL_BIAS, MIN_SAFE_CELL_SIZE, pcgHashU32, cellHash, hashUnit } from '../src/math/lattice'
import { WGS84 } from '../src/math/wgs84'

describe('PCG hash', () => {
  it('is deterministic and returns a uint32', () => {
    for (const x of [0, 1, 7, 123456, 4294967295]) {
      const h = pcgHashU32(x)
      expect(h).toBe(pcgHashU32(x))
      expect(Number.isInteger(h)).toBe(true)
      expect(h).toBeGreaterThanOrEqual(0)
      expect(h).toBeLessThanOrEqual(4294967295)
    }
  })
  it('avalanches: a one-cell step flips about half the bits', () => {
    let flips = 0
    const N = 2000
    for (let i = 0; i < N; i++) {
      const a = cellHash(i, 17, -3, 0)
      const b = cellHash(i + 1, 17, -3, 0)
      let x = (a ^ b) >>> 0
      while (x) { flips += x & 1; x >>>= 1 }
    }
    const mean = flips / N
    expect(mean).toBeGreaterThan(12)
    expect(mean).toBeLessThan(20)
  })
  it('separates the three axes and the salt', () => {
    expect(cellHash(1, 2, 3, 0)).not.toBe(cellHash(3, 2, 1, 0))
    expect(cellHash(1, 2, 3, 0)).not.toBe(cellHash(1, 2, 3, 1))
  })
  it('hashUnit is uniform on [0,1): 16 buckets stay within 8 percent of even', () => {
    const bins = new Array(16).fill(0) as number[]
    const N = 160000
    for (let i = 0; i < N; i++) {
      const bucket = Math.floor(hashUnit(cellHash(i, i * 7 + 1, i * 13 - 5, 2)) * 16)
      bins[bucket] = (bins[bucket] ?? 0) + 1
    }
    for (const b of bins) {
      expect(b).toBeGreaterThan((N / 16) * 0.92)
      expect(b).toBeLessThan((N / 16) * 1.08)
    }
  })
  it('hashUnit never reaches 1', () => {
    let minimum = Infinity
    let maximum = -Infinity
    for (let i = 0; i < 100000; i++) {
      const u = hashUnit(cellHash(i, 0, 0, 0))
      minimum = Math.min(minimum, u)
      maximum = Math.max(maximum, u)
    }
    expect(minimum).toBeGreaterThanOrEqual(0)
    expect(maximum).toBeLessThan(1)
  })
})

describe('the i32 lattice outlives regime 2', () => {
  it('supports cells down to 6.1 mm across the whole shell', () => {
    expect(MIN_SAFE_CELL_SIZE).toBeCloseTo(6.119e-3, 6)
  })
  it('is finer than the ~4 cm cell where omega reaches 1 (the 5 m handoff)', () => {
    expect(MIN_SAFE_CELL_SIZE).toBeLessThan(0.037)
  })
  it('biases every reachable cell index into the non-negative, unsigned-32-safe range', () => {
    // MIN_SAFE_CELL_SIZE is DEFINED as (a+H_max)/CELL_BIAS, so this ceils to
    // exactly CELL_BIAS (2^30) by construction - a tautology, not a margin
    // check. What actually matters: the raw (signed) index fits a WGSL
    // `int` (< 2^31), and biasing it stays within an unsigned 32-bit hash
    // input (>= 0 and < 2^32).
    const extreme = Math.ceil((WGS84.a + 192000) / MIN_SAFE_CELL_SIZE)
    expect(extreme).toBeLessThan(2 ** 31)
    expect(extreme + CELL_BIAS).toBeGreaterThanOrEqual(0)
    expect(extreme + CELL_BIAS).toBeLessThan(2 ** 32)
  })
})
