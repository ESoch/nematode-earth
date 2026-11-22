import { describe, it, expect } from 'vitest'
import {
  MEAN_OVER_MEDIAN, LN_SIGMA2, REL_VAR, MEDIAN_A,
  aggregateLogSigma2, aggregateMean, aggregateVariance, relVarPerInstance,
  drawAggregateArea, boxMullerNormal,
} from '../src/math/crossSection'
import { E_A } from '../src/math/optical'

/** Deterministic uniforms, no RNG, reproducible in CI. */
function* lcg(seed = 12345): Generator<number> {
  let s = seed >>> 0
  for (; ;) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    yield (s + 0.5) / 4294967296
  }
}

describe('the adjudicated shape constants', () => {
  it('uses mean/median = 2.02 from the spec', () => {
    expect(MEAN_OVER_MEDIAN).toBeCloseTo(2.02, 10)
  })
  it('derives ln-sigma^2 = 2*ln(mean/median)', () => {
    expect(LN_SIGMA2).toBeCloseTo(1.406195, 6)
  })
  it('gives relative variance (mean/median)^2 - 1 = 3.0804', () => {
    expect(REL_VAR).toBeCloseTo(3.0804, 4)
  })
  it('places the median at E[A]/2.02', () => {
    expect(MEDIAN_A).toBeCloseTo(7.37624e-9, 13)
    expect(MEDIAN_A * MEAN_OVER_MEDIAN).toBeCloseTo(E_A, 15)
  })
})

describe('aggregate of omega draws', () => {
  it('is EXACTLY the single-animal log-normal at omega = 1', () => {
    expect(aggregateLogSigma2(1)).toBeCloseTo(LN_SIGMA2, 12)
  })
  it('collapses toward a delta as omega grows', () => {
    expect(aggregateLogSigma2(1e6)).toBeLessThan(1e-5)
    expect(aggregateLogSigma2(1e6)).toBeGreaterThan(0)
  })
  it('keeps mean linear and variance linear in omega', () => {
    expect(aggregateMean(1000)).toBeCloseTo(1000 * E_A, 18)
    expect(aggregateVariance(1000)).toBeCloseTo(1000 * E_A * E_A * REL_VAR, 24)
  })
  it('reports the noise inflation the spec skew implies', () => {
    expect(relVarPerInstance(1)).toBeCloseTo(3.0804, 4)
    expect(relVarPerInstance(1e6)).toBeCloseTo(3.0804e-6, 10)
  })
})

describe('drawAggregateArea reproduces its own moments', () => {
  const N = 200000
  it.each([[1], [10], [1000]])('omega=%i: sample mean within 1.5 percent', (omega) => {
    const g = lcg(7)
    let sum = 0
    for (let i = 0; i < N; i++) sum += drawAggregateArea(omega, g.next().value, g.next().value)
    expect(sum / N).toBeGreaterThan(aggregateMean(omega) * 0.985)
    expect(sum / N).toBeLessThan(aggregateMean(omega) * 1.015)
  })

  it('is right-skewed at omega = 1: the median is close to half the mean', () => {
    const g = lcg(11)
    const xs: number[] = []
    for (let i = 0; i < N; i++) xs.push(drawAggregateArea(1, g.next().value, g.next().value))
    xs.sort((a, b) => a - b)
    const median = xs[Math.floor(N / 2)]!
    const mean = xs.reduce((a, b) => a + b, 0) / N
    expect(mean / median).toBeGreaterThan(1.85)
    expect(mean / median).toBeLessThan(2.2)
  })

  it('never returns a non-positive area', () => {
    const g = lcg(13)
    for (let i = 0; i < 50000; i++) {
      expect(drawAggregateArea(1, g.next().value, g.next().value)).toBeGreaterThan(0)
    }
  })

  it('is deterministic in its two uniforms', () => {
    expect(drawAggregateArea(5, 0.31, 0.72)).toBe(drawAggregateArea(5, 0.31, 0.72))
  })
})

describe('boxMullerNormal', () => {
  it('has zero mean and unit variance', () => {
    const g = lcg(3)
    let s = 0
    let s2 = 0
    const N = 200000
    for (let i = 0; i < N; i++) {
      const z = boxMullerNormal(g.next().value, g.next().value)
      s += z
      s2 += z * z
    }
    expect(Math.abs(s / N)).toBeLessThan(0.02)
    expect(s2 / N).toBeGreaterThan(0.97)
    expect(s2 / N).toBeLessThan(1.03)
  })
})
