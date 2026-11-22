import { describe, it, expect } from 'vitest'
import { DataUtils, RedFormat } from 'three/webgpu'
import { syntheticTau } from '../src/field/synthetic'
import {
  faceDirection, encodeLogTau, decodeLogTau, bakeSyntheticCubeTexture,
  bakeSyntheticDensityArrayTexture, CUBE_FACE_COUNT,
} from '../src/field/bakeCube'

// u=0,v=0 produces signed-zero components (e.g. face 0 is [1,-v,-u] = [1,-0,-0]),
// which are mathematically 0 but fail toEqual's Object.is-based comparison
// against a literal `0`. Collapsing -0 to 0 here checks the values the test
// actually cares about without asserting anything about zero's sign.
const dropNegativeZero = (v: readonly number[]): number[] => v.map((n) => (n === 0 ? 0 : n))

describe('faceDirection', () => {
  it('puts the major axis at the face centre', () => {
    expect(dropNegativeZero(faceDirection(0, 0, 0))).toEqual([1, 0, 0])
    expect(dropNegativeZero(faceDirection(1, 0, 0))).toEqual([-1, 0, 0])
    expect(dropNegativeZero(faceDirection(2, 0, 0))).toEqual([0, 1, 0])
    expect(dropNegativeZero(faceDirection(3, 0, 0))).toEqual([0, -1, 0])
    expect(dropNegativeZero(faceDirection(4, 0, 0))).toEqual([0, 0, 1])
    expect(dropNegativeZero(faceDirection(5, 0, 0))).toEqual([0, 0, -1])
  })

  it('throws on an out-of-range face index', () => {
    expect(() => faceDirection(6, 0, 0)).toThrow()
    expect(() => faceDirection(-1, 0, 0)).toThrow()
  })

  it('round-trips at the shared edge between the +X and +Y faces', () => {
    // (1,1,0) is the edge shared by face 0 (+X) and face 2 (+Y). Walking to
    // that edge from each face's own parameterization must land on (nearly)
    // the same direction - this is the self-consistency the module doc
    // promises, checked here rather than only asserted in a comment.
    const fromPlusX = faceDirection(0, 0, -0.999) // u=0, v near -1
    const fromPlusY = faceDirection(2, 0.999, 0) // u near 1, v=0
    const norm = (d: [number, number, number]): [number, number, number] => {
      const n = Math.hypot(...d)
      return [d[0] / n, d[1] / n, d[2] / n]
    }
    const [ax, ay, az] = norm(fromPlusX)
    const [bx, by, bz] = norm(fromPlusY)
    expect(ax).toBeCloseTo(bx, 2)
    expect(ay).toBeCloseTo(by, 2)
    expect(az).toBeCloseTo(bz, 2)
  })

  it('round-trips at the shared edge between the +Z and -Y faces', () => {
    // (0,-1,1): face 4 (+Z, point=(u,-v,1)) at v near +1 (its y=-1-adjacent
    // edge), and face 3 (-Y, point=(u,-1,-v)) at v near -1 (its z=1-adjacent
    // edge) - both converge to (0,-1,1) as their respective v approaches
    // that boundary.
    const fromPlusZ = faceDirection(4, 0, 0.999)
    const fromMinusY = faceDirection(3, 0, -0.999)
    const norm = (d: [number, number, number]): [number, number, number] => {
      const n = Math.hypot(...d)
      return [d[0] / n, d[1] / n, d[2] / n]
    }
    const [ax, ay, az] = norm(fromPlusZ)
    const [bx, by, bz] = norm(fromMinusY)
    expect(ax).toBeCloseTo(bx, 2)
    expect(ay).toBeCloseTo(by, 2)
    expect(az).toBeCloseTo(bz, 2)
  })
})

describe('encodeLogTau / decodeLogTau', () => {
  it('are exact inverses at float64 precision', () => {
    for (const tau of [0.0024, 0.005, 0.02, 0.047, 0.1, 0.19]) {
      expect(decodeLogTau(encodeLogTau(tau))).toBeCloseTo(tau, 12)
    }
  })
})

describe('bakeSyntheticCubeTexture structure', () => {
  it('produces 6 R16Float DataTexture faces at the requested size', () => {
    const cube = bakeSyntheticCubeTexture({ size: 16 })
    expect(cube.images.length).toBe(CUBE_FACE_COUNT)
    for (const face of cube.images) {
      expect(face.image.width).toBe(16)
      expect(face.image.height).toBe(16)
      expect(face.image.data?.length).toBe(16 * 16)
      expect(face.format).toBe(cube.format)
      expect(face.type).toBe(cube.type)
    }
    expect(cube.generateMipmaps).toBe(false)
  })

  it('defaults to a 512-texel face size when no size is given', () => {
    const cube = bakeSyntheticCubeTexture()
    expect(cube.images[0]?.image.width).toBe(512)
    expect(cube.images[0]?.image.height).toBe(512)
  })
})

describe('bakeSyntheticDensityArrayTexture structure', () => {
  it('produces a 6-layer R16Float DataArrayTexture at the requested size', () => {
    const size = 16
    const array = bakeSyntheticDensityArrayTexture({ size })
    expect(array.image.width).toBe(size)
    expect(array.image.height).toBe(size)
    expect(array.image.depth).toBe(CUBE_FACE_COUNT)
    if (!array.image.data) throw new Error('array texture has no data')
    expect(array.image.data.length).toBe(size * size * CUBE_FACE_COUNT)
    expect(array.format).toBe(RedFormat)
    expect(array.generateMipmaps).toBe(false)
  })

  it('is byte-for-byte the same data bakeSyntheticCubeTexture writes, per face — same bytes, two GPU views', () => {
    const size = 8
    const cube = bakeSyntheticCubeTexture({ size })
    const array = bakeSyntheticDensityArrayTexture({ size })
    if (!array.image.data) throw new Error('array texture has no data')
    const arrayFullData = array.image.data
    const texelsPerFace = size * size
    for (let face = 0; face < CUBE_FACE_COUNT; face++) {
      const faceTexture = cube.images[face]
      if (!faceTexture?.image.data) throw new Error(`missing cube face ${face}`)
      const cubeData = faceTexture.image.data
      const arrayData = arrayFullData.subarray(face * texelsPerFace, (face + 1) * texelsPerFace)
      expect(Array.from(arrayData)).toEqual(Array.from(cubeData))
    }
  })
})

describe('bake round-trip against the CPU reference (spec-mandated encoding tolerance)', () => {
  it('decodes every texel of a small bake back to syntheticTau within R16Float precision', () => {
    const size = 16
    const cube = bakeSyntheticCubeTexture({ size })
    let maxRelErr = 0
    for (let face = 0; face < CUBE_FACE_COUNT; face++) {
      const faceTexture = cube.images[face]
      if (!faceTexture) throw new Error(`missing face ${face}`)
      const data = faceTexture.image.data
      if (!data) throw new Error(`face ${face} has no data`)
      for (let py = 0; py < size; py++) {
        const v = ((py + 0.5) / size) * 2 - 1
        for (let px = 0; px < size; px++) {
          const u = ((px + 0.5) / size) * 2 - 1
          const [x, y, z] = faceDirection(face, u, v)
          const reference = syntheticTau(x, y, z)

          const bits = data[py * size + px]
          if (bits === undefined) throw new Error('missing baked texel')
          const decoded = decodeLogTau(DataUtils.fromHalfFloat(bits))

          const relErr = Math.abs(decoded - reference) / reference
          maxRelErr = Math.max(maxRelErr, relErr)
        }
      }
    }
    // Half-float has a 10-bit mantissa (~3 decimal digits); encoding in log
    // space bounds the RELATIVE error in tau by roughly the ABSOLUTE
    // quantization error in log-space, which is well under 1% here. This is
    // the round trip the encoding path is meant to prove before real N_A
    // data (which needs it far more, spanning 6e5..2e7 per m2) replaces the
    // synthetic bake.
    expect(maxRelErr).toBeLessThan(0.01)
  })
})
