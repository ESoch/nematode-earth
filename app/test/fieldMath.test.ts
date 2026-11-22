import { describe, it, expect } from 'vitest'
import { syntheticTau, TAU_RICH } from '../src/field/synthetic'
import { syntheticTauPure } from '../src/field/fieldMath'
import { arealCoverage } from '../src/math/optical'

/**
 * `fieldMath.ts` is the pure-JS stepping stone the TSL port in
 * `syntheticTSL.ts` mirrors term-for-term (see that file's doc comment for
 * why it's a deliberate duplicate of `synthetic.ts`'s private hash3/fbm
 * rather than shared code — `synthetic.ts` is not modified in this task).
 *
 * These tests hold `syntheticTauPure` to a stronger bar than the port itself
 * can be held to: since both are plain JS, run on the same samples, they
 * must agree with `syntheticTau` EXACTLY (not just statistically), which
 * gives high confidence the duplication is faithful. `syntheticTSL.ts` then
 * only needs to reproduce this file's formulas node-for-node, which is
 * checked by code inspection since TSL can't execute in Node (see the
 * session report).
 */

function syntheticStatsPure(samples = 20000) {
  let min = Infinity
  let max = -Infinity
  let sum = 0
  const ga = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < samples; i++) {
    const z = 1 - (2 * i) / (samples - 1)
    const r = Math.sqrt(Math.max(0, 1 - z * z))
    const th = ga * i
    const t = syntheticTauPure(Math.cos(th) * r, Math.sin(th) * r, z)
    min = Math.min(min, t)
    max = Math.max(max, t)
    sum += t
  }
  return {
    min, max, mean: sum / samples,
    coverageRatio: arealCoverage(max) / arealCoverage(min),
  }
}

describe('fieldMath pointwise equivalence to the CPU reference', () => {
  it('matches syntheticTau exactly on the poles, equator, and off-axis directions', () => {
    const dirs: Array<[number, number, number]> = [
      [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1],
      [0.3, 0.5, 0.81], [-0.6, 0.2, 0.77], [0.1, -0.9, 0.42],
      [7, 0, 0], [1, 1, 1], [-1, -1, -1],
    ]
    for (const [x, y, z] of dirs) {
      expect(syntheticTauPure(x, y, z)).toBe(syntheticTau(x, y, z))
    }
  })

  it('matches across a dense deterministic sphere sampling, not just spot checks', () => {
    const ga = Math.PI * (3 - Math.sqrt(5))
    const samples = 5000
    for (let i = 0; i < samples; i++) {
      const z = 1 - (2 * i) / (samples - 1)
      const r = Math.sqrt(Math.max(0, 1 - z * z))
      const th = ga * i
      const x = Math.cos(th) * r
      const y = Math.sin(th) * r
      expect(syntheticTauPure(x, y, z)).toBe(syntheticTau(x, y, z))
    }
  })
})

describe('fieldMath reproduces the published statistics (spec section 1.1)', () => {
  it('stays within the physical range everywhere', () => {
    const s = syntheticStatsPure(20000)
    expect(s.min).toBeGreaterThan(0)
    expect(s.max).toBeLessThanOrEqual(TAU_RICH * 1.0001)
  })

  it('reproduces the 27x coverage dynamic range (coverageRatio > 20)', () => {
    const s = syntheticStatsPure(20000)
    expect(s.coverageRatio).toBeGreaterThan(20)
  })

  it('has a global mean in the right neighbourhood (0.02..0.09)', () => {
    const s = syntheticStatsPure(50000)
    expect(s.mean).toBeGreaterThan(0.02)
    expect(s.mean).toBeLessThan(0.09)
  })

  it('is brighter at the poles than the tropics', () => {
    const dirFromLat = (latDeg: number): [number, number, number] => {
      const p = (latDeg * Math.PI) / 180
      return [Math.cos(p), 0, Math.sin(p)]
    }
    const [tx, ty, tz] = dirFromLat(10)
    const [px, py, pz] = dirFromLat(75)
    expect(syntheticTauPure(px, py, pz)).toBeGreaterThan(syntheticTauPure(tx, ty, tz))
  })
})
