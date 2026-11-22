import { describe, expect, it } from 'vitest'
import { E_A } from '../src/math/optical'
import { DataUtils } from 'three/webgpu'
import { filteredSoilReference, halfFloatUlp } from '../src/experiments/soilCandidateReference'

describe('independent CPU reference for filtered GPU soil samples', () => {
  it('bounds half-float rounding by its actual spacing, including subnormals', () => {
    expect(halfFloatUlp(-9.23)).toBe(1 / 128)
    expect(halfFloatUlp(.1)).toBe(1 / 16384)
    expect(halfFloatUlp(0)).toBe(2 ** -24)
    expect(halfFloatUlp(1e-8)).toBe(2 ** -24)
  })
  const half = DataUtils.toHalfFloat
  const density = (log: number) => Math.exp(DataUtils.fromHalfFloat(half(log))) * E_A
  const face = new Uint16Array([
    12, .1, .5, .25, 15, .1, .125, .75,
    12, .1, .5, .25, 15, .1, .125, .75,
  ].map(half))
  const faces = Array.from({ length: 6 }, () => face)

  it('interpolates area means and weights CV by supported area', () => {
    const result = filteredSoilReference(faces, 2, [1, 0, 0])
    expect(result.tau).toBeCloseTo((density(12) + density(15)) / 2, 9)
    expect(result.cv).toBe((.5 * .25 + .125 * .75) / (.25 + .75))
    expect(result.support).toBe(.5)
  })

  it('retains the texel center and uses smoothstep weights off center', () => {
    const center = filteredSoilReference(faces, 2, [1, .5, .5])
    expect(center.tau).toBeCloseTo(density(12), 9)
    expect(center.cv).toBe(.5)
    const result = filteredSoilReference(faces, 2, [1, 0, .25])
    expect(result.tau).toBeCloseTo(density(12) * .84375 + density(15) * .15625, 9)
  })

  it('blends a supported +X cell with an empty +Z cell across their shared edge', () => {
    const missing = new Uint16Array(16)
    const result = filteredSoilReference([face, missing, missing, missing, missing, missing], 2, [1, .5, 1])
    expect(result.tau).toBeCloseTo(density(12) / 2, 9)
    expect(result.cv).toBe(.5)
    expect(result.support).toBe(.125)
    const empty = filteredSoilReference(faces.map(() => missing), 2, [1, 0, 0])
    expect([empty.tau, empty.cv, empty.support]).toEqual([0, 0, 0])
  })
})
