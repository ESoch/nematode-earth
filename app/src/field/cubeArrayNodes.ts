import type * as THREE from 'three/webgpu'
import {
  Fn, vec2, vec3, float, int, abs, max, min, texture,
} from 'three/tsl'

/**
 * Manual cube-face lookup for sampling a `THREE.DataArrayTexture` (one
 * layer per cube face) in place of `THREE.CubeTextureNode` — required
 * because `cubeTexture()` cannot run inside a compute kernel on this
 * three.js version.
 *
 * The mechanism, confirmed against three.js's own shipped source (not
 * guessed): `CubeTextureNode.setupUV()` (`node_modules/three/src/nodes/
 * accessors/CubeTextureNode.js`) unconditionally does
 * `uvNode = materialEnvRotation.mul(uvNode)` for every non-depth cube
 * texture, no opt-out. `materialEnvRotation`'s `onObjectUpdate`
 * (`MaterialProperties.js`) destructures `{ material, scene }` and reads
 * `scene.environment`. `WebGPURenderer.compute()`/`.computeAsync()` never
 * supplies a scene, so that destructure throws on `null.environment` every
 * dispatch. The base `TextureNode` class (what a plain `DataArrayTexture`
 * uses) has no such dependency — its own `setupUV` only touches a local
 * flip-Y uniform seeded with a literal `false`. See
 * `docs/superpowers/results/` for the full write-up team lead asked for.
 *
 * `directionToFaceUvNode` below is a TERM-FOR-TERM TSL port of
 * `loadCube.ts`'s `directionToFaceUV` (itself the tested inverse of
 * `bakeCube.ts`'s `faceDirection`) — same major-axis test, same face
 * numbering (0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z, matching `cube_meta.json`'s
 * `face_order` and every atlas this project bakes), same per-branch
 * algebra, so a shader built from this samples exactly the texel the CPU
 * function would pick for the same direction.
 *
 * All SIX u/v candidates (x-major, y-major, z-major, both signs of each)
 * are computed UNCONDITIONALLY — `.select()` is not short-circuiting, both
 * operands are always evaluated — and only one is picked at the end. That
 * means every division below runs even for branches that get discarded, so
 * an axis-aligned direction (e.g. straight up at a pole, `(0,0,1)`) always
 * divides by a literal zero in one of the discarded branches (x/y with
 * y=0, say). At runtime this is legal WGSL (float division by zero is
 * IEEE754 infinity, discarded harmlessly by `.select()`), but WGSL's
 * constant folder evaluates fully-literal expressions at PARSE time and
 * rejects `1.0 / 0.0` outright — hit for real when a fixed diagnostic probe
 * direction was passed as a literal `vec3(1,0,0)` rather than a `uniform()`
 * (`Particles.ts`'s `probeDir`), which failed compute-pipeline creation
 * entirely. `safeDenom` below floors every division's DENOMINATOR magnitude
 * away from exact zero (sign-preserving, so it cannot flip a real,
 * non-degenerate selected-branch result) — fixes the runtime fragility for
 * real axis-aligned inputs, independent of the uniform-vs-literal fix,
 * which is what actually unblocks compilation for a literal probe.
 */

type FloatNode = THREE.Node<'float'>
type Vec3Node = THREE.Node<'vec3'>

/**
 * Floors `d`'s MAGNITUDE away from zero while preserving its sign, so
 * `num.div(safeDenom(d))` never divides by exactly zero. Only affects
 * discarded branches or the true (near-)zero-vector edge case — the
 * SELECTED branch's own denominator is, by construction, the
 * largest-magnitude component of a genuinely nonzero direction, so it is
 * never anywhere near `1e-20` in practice and this guard is a no-op for it.
 */
const safeDenom = (d: FloatNode): FloatNode => (
  d.greaterThanEqual(0.0).select(max(d, float(1e-20)), min(d, float(-1e-20)))
)

/**
 * Packed as `vec3(face, u, v)` — a TSL `Fn` body returns one node, and
 * `face` (an exact small integer 0..5) round-trips losslessly through a
 * float. Unpacked by `sampleDensityArrayNode` below.
 */
export const directionToFaceUvNode = Fn(([dir]: [Vec3Node]): Vec3Node => {
  const x = dir.x
  const y = dir.y
  const z = dir.z
  const ax = abs(x)
  const ay = abs(y)
  const az = abs(z)

  const xMajor = ax.greaterThanEqual(ay).and(ax.greaterThanEqual(az))
  const yMajor = ay.greaterThanEqual(ax).and(ay.greaterThanEqual(az))

  const xPos = x.greaterThan(0.0)
  const yPos = y.greaterThan(0.0)
  const zPos = z.greaterThan(0.0)

  const face = xMajor.select(
    xPos.select(float(0), float(1)),
    yMajor.select(
      yPos.select(float(2), float(3)),
      zPos.select(float(4), float(5)),
    ),
  )

  // u,v: term-for-term match to directionToFaceUV's three branches —
  //   x-major: u=-z/x (both signs), v = xPos ? -y/x : y/x
  //   y-major: u = yPos ? x/y : -x/y, v=z/y (both signs)
  //   z-major: u=x/z (both signs), v = zPos ? -y/z : y/z
  // Numerators are the raw components (a zero numerator is fine - 0/d is
  // just 0); denominators go through safeDenom (see its doc above).
  const xSafe = safeDenom(x)
  const ySafe = safeDenom(y)
  const zSafe = safeDenom(z)

  const uX = z.negate().div(xSafe)
  const vX = xPos.select(y.negate().div(xSafe), y.div(xSafe))
  const uY = yPos.select(x.div(ySafe), x.negate().div(ySafe))
  const vY = z.div(ySafe)
  const uZ = x.div(zSafe)
  const vZ = zPos.select(y.negate().div(zSafe), y.div(zSafe))

  const u = xMajor.select(uX, yMajor.select(uY, uZ))
  const v = xMajor.select(vX, yMajor.select(vY, vZ))

  return vec3(face, u, v)
})

/**
 * Samples a regime-2 density array texture (currently only
 * `bakeCube.ts`'s `bakeSyntheticDensityArrayTexture` — built the same way
 * `field/loadCube.ts`'s real atlas loader could, via
 * `field/textureArray.ts`'s `packFacesIntoDataArrayTexture`, if a future
 * regime wants the real data instead of the synthetic bake) along `dir`.
 * Returns the RAW (still log-encoded) R channel — callers decode with
 * `exp(...)` themselves, matching the shape of the existing
 * `cubeTexture(tex, dir).level(float(0)).r` call sites this replaces.
 *
 * NO CROSS-FACE BLENDING — deliberate, not an oversight. Unlike the
 * hardware-seamless cube path, a face boundary here is a hard seam
 * (nearest included layer only, `ClampToEdgeWrapping` within it). Chosen
 * because regime 2's particles are discrete point samples — one call here
 * decides ONE instance's own cross-section/emission weight, not a shaded
 * continuous surface — so there is no rendered edge to blend across. Any
 * seam-sized tau discontinuity sits far inside the ~14% sampling noise the
 * sigma_pix schedule already assumes (session4 regime2 report), and
 * instances straddling a face boundary are independently hashed/jittered
 * regardless, so a seam cannot itself produce a visible line of particles.
 * If a future consumer DOES render a continuous surface from this array
 * (unlike regime 2), it will need real neighbour-face blending — this
 * function does not provide it.
 *
 * Explicit `.level(float(0))`: required inside compute / non-uniform
 * control flow, same discipline as every other texture sample in this
 * codebase (spec section 10).
 */
export function sampleDensityArrayNode(arrayTexture: THREE.DataArrayTexture, dirNode: Vec3Node): FloatNode {
  const packed = directionToFaceUvNode(dirNode)
  const face = int(packed.x)
  const uv = vec2(packed.y, packed.z)
  return texture(arrayTexture, uv).depth(face).level(float(0)).r
}
