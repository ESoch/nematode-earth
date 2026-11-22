import type { Node } from 'three/webgpu'
import {
  float, sin, fract, abs, asin, pow, exp, max, clamp, PI, normalize,
} from 'three/tsl'
import { TAU_DESERT, TAU_RICH } from './synthetic'

type FloatNode = Node<'float'>
type Vec3Node = Node<'vec3'>

/**
 * TSL port of `synthetic.ts`'s `syntheticTau`, mirrored term-for-term against
 * the pure-JS reference in `fieldMath.ts` (see that file's doc comment for
 * why this is a deliberate duplicate rather than shared code).
 *
 * Analytic, not a texture lookup: the shell march's transmittance `Break()`
 * makes control flow non-uniform, and WGSL forbids implicit-LOD
 * `textureSample` there (spec section 10). An analytic field needs no
 * texture at all, so the hazard doesn't apply.
 */

/** Mirrors `fieldMath.ts`'s `hash3`. `s - floor(s)` is exactly `fract(s)`. */
function hash3Node(x: FloatNode, y: FloatNode, z: FloatNode): FloatNode {
  const s = sin(x.mul(127.1).add(y.mul(311.7)).add(z.mul(74.7))).mul(43758.5453)
  return fract(s)
}

/**
 * Mirrors `fieldMath.ts`'s `fbm`. `octaves` is a compile-time JS constant
 * (3 or 4, exactly as called in `syntheticTau`), so the loop is unrolled at
 * graph-construction time rather than built as a runtime TSL `Loop` — there
 * is nothing dynamic about the octave count to justify GPU-side looping.
 */
function fbmNode(x: FloatNode, y: FloatNode, z: FloatNode, octaves: number): FloatNode {
  let sum: FloatNode = float(0)
  let amp = 0.5
  let freq = 1.0
  for (let i = 0; i < octaves; i++) {
    sum = sum.add(hash3Node(x.mul(freq), y.mul(freq), z.mul(freq)).mul(amp))
    freq *= 2.07
    amp *= 0.5
  }
  return sum
}

/**
 * `syntheticTau(dirX, dirY, dirZ)` ported to a TSL graph. `dir` need not be
 * pre-normalized; this normalizes internally, exactly as the CPU reference
 * does via `Math.hypot`.
 */
export function syntheticTauNode(dir: Vec3Node): FloatNode {
  const d = normalize(dir)
  const x = d.x
  const y = d.y
  const z = d.z

  const absLat = abs(asin(clamp(z, -1, 1))).div(PI.div(2))

  const subArctic = exp(pow(absLat.sub(0.78).div(0.28), 2).negate())
  const tropicalFloor = float(0.05).add(float(0.18).mul(exp(pow(absLat.div(0.20), 2).negate())))
  const envelope = max(tropicalFloor, subArctic)

  const continents = fbmNode(x.mul(2.3), y.mul(2.3), z.mul(2.3), 3)
  const detail = fbmNode(x.mul(11.0), y.mul(11.0), z.mul(11.0), 4)
  const modulation = continents.mul(0.55).add(detail.mul(0.45))

  const t = envelope.mul(float(0.12).add(float(0.88).mul(modulation)))
  const tc = clamp(t, 0, 1)

  return float(TAU_DESERT).add(float(TAU_RICH - TAU_DESERT).mul(tc))
}
