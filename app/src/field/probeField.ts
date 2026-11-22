import { Fn, float, abs, asin, clamp, exp, max, pow } from 'three/tsl'
import type { Node } from 'three/webgpu'
import { TAU_DESERT, TAU_RICH } from './synthetic'

/**
 * A deliberately minimal analytic stand-in for the baked atlas, used ONLY by
 * regime 2's own harness/comparison code: a smooth latitude band with the
 * POLES BRIGHTEST, anchored to the same biome constants as
 * `field/synthetic.ts`. No noise, no fbm - regime 2's own checks need a
 * field whose exact analytic integral is knowable so energy comparisons
 * have a reference; the real field (with its spatial texture) is
 * `field/syntheticTSL.ts` / the baked cube, and `TslFieldPort` below exists
 * so regime 2 takes whichever one by dependency injection rather than
 * importing either directly.
 */
export const probeTauNadir = Fn(([dir]: [Node<'vec3'>]) => {
  const absLat = abs(asin(clamp(dir.z, -1.0, 1.0))).div(Math.PI / 2)
  const subArctic = exp(pow(absLat.sub(0.78).div(0.28), 2.0).negate())
  const tropics = float(0.05).add(float(0.18).mul(exp(pow(absLat.div(0.2), 2.0).negate())))
  const env = max(tropics, subArctic)
  return float(TAU_DESERT).add(float(TAU_RICH - TAU_DESERT).mul(clamp(env, 0.0, 1.0)))
})

/** Dependency-injection seam: regime 2 code takes a field by this shape rather than importing a specific implementation. */
export interface TslFieldPort {
  tauNadirAt: (dir: Node<'vec3'>) => Node<'float'>
}
