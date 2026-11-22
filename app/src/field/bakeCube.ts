import * as THREE from 'three/webgpu'
import { syntheticTau } from './synthetic'
import { packFacesIntoDataArrayTexture } from './textureArray'

/**
 * Bakes `synthetic.ts`'s CPU field into a `THREE.CubeTexture`, so the
 * cube-sampling path - the shape the real baked atlas will eventually take
 * (spec section 4.4) - can be measured and proven correct before the real
 * data pipeline exists. This is a plain (non-EAC) cube map: EAC's warp is
 * data-pipeline work and is not attempted here.
 *
 * `synthetic.ts` is the tested CPU reference and is NOT modified by this
 * file - only its public `syntheticTau` is called.
 */

export const CUBE_FACE_COUNT = 6

/**
 * Direction for a texel at face `face` (0..5, matching `THREE.CubeTexture`'s
 * `images` order [+X,-X,+Y,-Y,+Z,-Z] - see `CubeTextureLoader.js`'s doc
 * comment: "positive-x is to the right when looking up the positive-z
 * axis") and normalized device coordinates `u`,`v` in [-1,1] on that face.
 *
 * This is the algebraic inverse of the standard cubemap face-selection
 * table shared by D3D, Vulkan and WebGPU (WebGPU inherited it from D3D) -
 * given a major-axis direction `(x,y,z)`, the FORWARD table is:
 *   +X: u=-z, v=-y     -X: u=z,  v=-y     +Y: u=x, v=z
 *   -Y: u=x,  v=-z      +Z: u=x,  v=-y     -Z: u=-x, v=-y
 * with the major-axis component fixed at its face's +-1. Inverting each row
 * for its face gives the six branches below. They round-trip by
 * construction - e.g. face 0's near-edge texel at (u=0,v to -1) and face 3's
 * near-edge texel at (u to 1,v=0) both approach the same direction (1,1,0),
 * the edge shared by the +X and -Y faces (checked in bakeCube.test.ts).
 *
 * `(u,v)=(-1,-1)` is the texel-space origin, and row 0 of the raw pixel data
 * this module writes into each face is that `v=-1` row - `flipY` is
 * explicitly `false` throughout (both `DataTexture`'s own default and
 * `CubeTexture`'s explicit override), so no row-order translation happens
 * between bake time and GPU upload.
 */
export function faceDirection(face: number, u: number, v: number): [number, number, number] {
  switch (face) {
    case 0: return [1, -v, -u]
    case 1: return [-1, -v, u]
    case 2: return [u, 1, v]
    case 3: return [u, -1, -v]
    case 4: return [u, -v, 1]
    case 5: return [-u, -v, -1]
    default: throw new Error(`faceDirection: face index ${face} out of range [0,5]`)
  }
}

/**
 * Log-encodes tau for storage, mirroring the real pipeline's plan to
 * log-encode areal density (spec section 4.5): land values there span
 * roughly 6e5 to 2e7 per m2, and a linear encoding crushes the
 * desert-to-boreal gradient at limited precision. `tau`'s own range
 * (TAU_DESERT..TAU_RICH, ~0.0024..0.19) doesn't strictly need it at R16Float
 * precision, but the point of this bake is to prove the encode/decode round
 * trip before the real data arrives, so it is exercised here regardless.
 * Decoded in the shader with `exp(...)` - see `regimes/Shell.ts`.
 */
export function encodeLogTau(tau: number): number {
  return Math.log(tau)
}

export function decodeLogTau(encoded: number): number {
  return Math.exp(encoded)
}

export interface BakeCubeOptions {
  /** Texels per face edge. Default 512 -> ~19.5 km/texel at R=6371 km, finer than the real data's effective resolution. */
  size?: number
}

/**
 * Bakes `syntheticTau` into a 6-face cube texture, R16Float
 * (`RedFormat`+`HalfFloatType` - filterable everywhere on WebGPU, unlike
 * `r32float`, which needs the optional `float32-filterable` feature at only
 * 52% iOS availability), natural-log encoded per `encodeLogTau` above.
 *
 * No mip chain is generated: the march samples level 0 explicitly via
 * `cubeTexture(...).level(0)` (spec-mandated inside the march loop's
 * non-uniform control flow - WGSL forbids implicit-LOD `textureSample`
 * there, and has no `textureLoad` overload for `texture_cube` at all), so
 * there is nothing for a mip filter to select between.
 */
/**
 * Computes the six faces' raw texel data (half-float bit patterns,
 * log-tau-encoded) shared by `bakeSyntheticCubeTexture` (below) and
 * `bakeSyntheticDensityArrayTexture` — same bytes, two GPU views, so the
 * cube-sampling path (regime 1, `regimes/Shell.ts`) and the array-sampling
 * path (regime 2, `field/cubeArrayNodes.ts` — `CubeTextureNode` cannot run
 * inside a compute kernel, see that file's doc) agree on every direction by
 * construction. Not exported: an internal factoring only, so it does not
 * touch `bakeSyntheticCubeTexture`'s public signature or behaviour, which
 * `app/src/experiments/**` depends on directly.
 */
function bakeSyntheticFaceData(size: number): Uint16Array[] {
  const faces: Uint16Array[] = []
  for (let face = 0; face < CUBE_FACE_COUNT; face++) {
    const data = new Uint16Array(size * size)
    for (let py = 0; py < size; py++) {
      const v = ((py + 0.5) / size) * 2 - 1
      const row = py * size
      for (let px = 0; px < size; px++) {
        const u = ((px + 0.5) / size) * 2 - 1
        const [x, y, z] = faceDirection(face, u, v)
        const tau = syntheticTau(x, y, z)
        data[row + px] = THREE.DataUtils.toHalfFloat(encodeLogTau(tau))
      }
    }
    faces.push(data)
  }
  return faces
}

export function bakeSyntheticCubeTexture(opts: BakeCubeOptions = {}): THREE.CubeTexture<THREE.DataTexture> {
  const size = opts.size ?? 512
  const faceData = bakeSyntheticFaceData(size)

  const faces = faceData.map((data) => {
    const faceTexture = new THREE.DataTexture(data, size, size, THREE.RedFormat, THREE.HalfFloatType)
    faceTexture.needsUpdate = true
    return faceTexture
  })

  const cube = new THREE.CubeTexture(
    faces,
    THREE.CubeReflectionMapping,
    THREE.ClampToEdgeWrapping,
    THREE.ClampToEdgeWrapping,
    THREE.LinearFilter,
    THREE.LinearFilter,
    THREE.RedFormat,
    THREE.HalfFloatType,
    1,
    THREE.NoColorSpace,
  )
  cube.generateMipmaps = false
  cube.needsUpdate = true
  return cube
}

/**
 * The regime-2-safe twin of `bakeSyntheticCubeTexture`: same
 * `bakeSyntheticFaceData` bytes, packed into a `THREE.DataArrayTexture`
 * (one layer per face, `field/textureArray.ts`) instead of a
 * `THREE.CubeTexture`. Sample it with `field/cubeArrayNodes.ts`'s
 * `sampleDensityArrayNode` — plain `cubeTexture()` cannot run inside a
 * compute kernel (see that file's doc for the confirmed mechanism).
 */
export function bakeSyntheticDensityArrayTexture(opts: BakeCubeOptions = {}): THREE.DataArrayTexture {
  const size = opts.size ?? 512
  const faceData = bakeSyntheticFaceData(size)

  const array = packFacesIntoDataArrayTexture(faceData, size)
  array.format = THREE.RedFormat
  array.type = THREE.HalfFloatType
  array.minFilter = THREE.LinearFilter
  array.magFilter = THREE.LinearFilter
  array.wrapS = THREE.ClampToEdgeWrapping
  array.wrapT = THREE.ClampToEdgeWrapping
  array.colorSpace = THREE.NoColorSpace
  array.generateMipmaps = false
  array.needsUpdate = true
  return array
}
