import * as THREE from 'three/webgpu'
import {
  float, vec3, uniform, dot, abs, mix, smoothstep, clamp, mx_fractal_noise_float,
} from 'three/tsl'
import { LAYERS, DEFAULT_SEED, type LayerName, type LayerParams } from './Dissolve'
import type { Vec3 } from '../math/raySpheroid'

/**
 * TSL mirror of Dissolve.ts.
 *
 * The FRONT ALGEBRA is reproduced exactly - it is three operations, and the
 * endpoint-totality property depends on the constants -2w and 1+4w, so they
 * are written here in the same form as `layerAlpha` for a reader diffing the
 * two files.
 *
 * The NOISE is deliberately NOT the same function. The CPU side (Dissolve.ts)
 * uses a value-noise built on a sin-hash, which is not reproducible across
 * GPU vendors; this uses MaterialX fractal noise instead, which is.
 * `noiseEnabled` exists so a future exact-agreement GPU probe can switch phi
 * to its analytic geometric form, where both implementations agree exactly
 * and a numeric comparison actually means something - nothing in this
 * session compares the two noises pointwise.
 *
 * No texture is sampled anywhere in this module, so the explicit-LOD rule
 * (spec section 10, see regimes/Shell.ts) does not apply to it - worth
 * keeping true, since this code is meant to run inside a matter pass whose
 * fragment shader discards (non-uniform control flow).
 */

type FloatNode = THREE.Node<'float'>
type Vec3Node = THREE.Node<'vec3'>

export interface DissolveUniforms {
  front: THREE.UniformNode<'float', number>
  seed: THREE.UniformNode<'vec3', THREE.Vector3>
  /** 0 or 1, not boolean: blended via `mix` against the noise-free phi rather than branched. */
  noiseEnabled: THREE.UniformNode<'float', number>
}

export function createDissolveUniforms(seed: Vec3 = DEFAULT_SEED): DissolveUniforms {
  return {
    front: uniform(0),
    seed: uniform(new THREE.Vector3(seed[0], seed[1], seed[2])),
    noiseEnabled: uniform(1),
  }
}

export function erosionPhiNode(dirNode: Vec3Node, p: LayerParams, u: DissolveUniforms): FloatNode {
  const n = vec3(dirNode).normalize()
  const geo = float(1).sub(dot(n, u.seed)).mul(0.5)
  if (p.noiseWeight === 0) return clamp(geo, 0, 1)
  const raw = mx_fractal_noise_float(n.mul(p.noiseFreq), p.octaves, 2.07, 0.5, 1.0).mul(0.5).add(0.5)
  const shaped = p.ridged ? float(1).sub(abs(raw.mul(2).sub(1))) : raw
  const withNoise = geo.mul(1 - p.noiseWeight).add(shaped.mul(p.noiseWeight))
  return clamp(mix(geo, withNoise, u.noiseEnabled), 0, 1)
}

export function layerAlphaNode(phiNode: FloatNode, frontNode: FloatNode, width: number): FloatNode {
  const f = clamp(frontNode, 0, 1).mul(1 + 4 * width).add(-2 * width)
  return smoothstep(f.sub(width), f.add(width), phiNode)
}

export function matterAlphaNode(dirNode: Vec3Node, layer: LayerName, u: DissolveUniforms): FloatNode {
  const p = LAYERS[layer]
  return layerAlphaNode(erosionPhiNode(dirNode, p, u), u.front, p.width)
}
