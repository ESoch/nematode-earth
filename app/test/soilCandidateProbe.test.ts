import { describe, expect, it } from 'vitest'
import { DataUtils } from 'three/webgpu'
import { filteredSoilReference, halfFloatUlp } from '../src/experiments/soilCandidateReference'

describe('independent CPU reference for filtered GPU soil samples', () => {
  it('bounds half-float rounding by its actual spacing, including subnormals', () => {
    expect(halfFloatUlp(-9.23)).toBe(1 / 128)
    expect(halfFloatUlp(.1)).toBe(1 / 16384)
    expect(halfFloatUlp(0)).toBe(2 ** -24)
    expect(halfFloatUlp(1e-8)).toBe(2 ** -24)
  })
  it('interpolates encoded channels before decode and gates empty support', () => {
    // Query the center of four equally weighted texels. R(logTau)=-4/-2:
    // interpolated tau is exp(-3), not the average of exp(-4),exp(-2).
    const face = new Uint16Array([
      -4, .1, .2, 1, -2, .1, .4, 1,
      -4, .1, .2, 1, -2, .1, .4, 1,
    ].map(DataUtils.toHalfFloat))
    const faces = Array.from({ length: 6 }, () => face)
    const result = filteredSoilReference(faces, 2, [1, 0, 0])
    expect(result.tau).toBeCloseTo(Math.exp(-3), 7)
    expect(result.cv).toBeCloseTo(.3, 3)
    expect(result.support).toBe(1)
    const missing = Array.from({ length: 6 }, () => new Uint16Array(16))
    expect(filteredSoilReference(missing, 2, [1, 0, 0]).tau).toBe(0)
  })

  it('uses the smoothstep reconstruction off-center, not nearest or plain linear', () => {
    const face = new Uint16Array([
      -4, 0, .2, 1, -2, 0, .2, 1,
      -4, 0, .2, 1, -2, 0, .2, 1,
    ].map(DataUtils.toHalfFloat))
    // +X face uses u=-z/x. u=-.25 -> texel coordinate .25, weight=.15625.
    const result = filteredSoilReference(Array.from({ length: 6 }, () => face), 2, [1, 0, .25])
    expect(result.tau).toBeCloseTo(Math.exp(-3.6875), 7)
  })
})
