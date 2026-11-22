import type * as THREE from 'three/webgpu'
import {
  vec3, float, mix, clamp, cross, abs, smoothstep, max, min, exp, asin,
  floor, fract, screenCoordinate,
  mx_fractal_noise_float,
} from 'three/tsl'
import {
  ASTRONOMICAL_TWILIGHT_NDOTSUN, LAND_STOPS_ARID, LAND_STOPS_LUSH, OCEAN_STOPS, RAYLEIGH_RGB_WEIGHTS,
  ICE_LAT_START_DEG, ICE_LAT_END_DEG, ICE_MAX_LOWERING_M,
  HIGH_LAT_COOL_COLOR, HIGH_LAT_COOL_START_DEG, HIGH_LAT_COOL_END_DEG, HIGH_LAT_COOL_MAX,
  UNCERTAINTY_STOPS, UNCERTAINTY_LAND_MIN, UNCERTAINTY_LAND_FULL,
  VEIL_DITHER_AMPLITUDE,
  type Stop,
} from './terrain'
import { WGS84 } from '../math/wgs84'
import { geographicCubeTexture as cubeTexture } from './geographicCube'

/**
 * TSL mirror of `terrain.ts`.
 *
 * The colour ramps reuse `terrain.ts`'s `LAND_STOPS`/`OCEAN_STOPS` arrays
 * directly (unrolled into a `mix`/`select` cascade at graph-BUILD time,
 * since the stops are compile-time constants) rather than re-typing the
 * numbers a second time — the CPU and GPU ramps cannot drift apart. The
 * displacement and bump-normal algebra mirror `terrainDisplacementM` /
 * `heightfieldNormal` term-for-term (see each function's doc), the same
 * CPU-source-of-truth / GPU-mirror split as `Dissolve.ts` / `dissolveNodes.ts`.
 *
 * Every cube sample here uses explicit `.level(float(0))` — some call sites run
 * inside `sweep/Matter.ts`'s vertex stage (no derivatives exist there at
 * all) and others sit ahead of a fragment-stage `Discard`, so the whole
 * module holds to the same explicit-LOD discipline as `regimes/Shell.ts`
 * (spec section 10) regardless of which stage a given function ends up
 * used from.
 */

type FloatNode = THREE.Node<'float'>
type Vec3Node = THREE.Node<'vec3'>

/** Elevation cube's R channel is raw signed metres — no log/exp, unlike the density cube. */
export function sampleElevationNode(elevationTex: THREE.CubeTexture, dirNode: Vec3Node): FloatNode {
  return cubeTexture(elevationTex, dirNode).level(float(0)).r
}

/**
 * Land fraction at the ELEVATION atlas's resolution (G channel) — 9.77
 * km/texel at the 1024 tier, eight times sharper than the density cube's,
 * and registered to the very same texels as the relief it is shaded
 * alongside. `pipeline/build/bake_elevation_cube.py` derives it per source
 * cell as `surface_elev >= 0 OR (surface_elev - bed_elev) > 1.0` — the
 * second clause is ice thickness, and it is what keeps West Antarctica from
 * opening up as a hole in the continent, since the bed under it sits
 * hundreds of metres below sea level.
 *
 * Still a FRACTION, not a boolean, so coastal texels blend exactly as the
 * old land fraction did and every consumer's `clamp01`-weighted blend keeps
 * working unchanged.
 */
export function sampleLandMaskNode(elevationTex: THREE.CubeTexture, dirNode: Vec3Node): FloatNode {
  return cubeTexture(elevationTex, dirNode).level(float(0)).g
}

/**
 * The density cube's R channel is `log(tau)` on the GPU (`field/loadCube.ts`'s
 * `densityFaceToTauFace` conversion at load time — NOT raw log-density,
 * that conversion already happened before this texture reached the GPU),
 * the exact same quantity `regimes/Shell.ts`'s `logTau = cubeTexture(...).r`
 * feeds into `exp()` for the veil's optical depth. Deliberately returned
 * un-exponentiated here: `vegetationIndexNode`'s smoothstep operates
 * directly in log space, where the real land distribution is well
 * conditioned (see `terrain.ts::VEGETATION_ARID_LOG_TAU`'s doc), rather
 * than on raw `tau` which spans many more orders of magnitude.
 */
export function sampleLogTauNode(densityTex: THREE.CubeTexture, dirNode: Vec3Node): FloatNode {
  return cubeTexture(densityTex, dirNode).level(float(0)).r
}

/**
 * Piecewise-linear colour ramp, unrolled into a cascading `select` at
 * graph-build time — `stops` is a plain JS array, iterated in JS, not a
 * runtime loop. Working from the HIGHEST segment down to the lowest and
 * overwriting `result` each time `x` falls at-or-below that segment's upper
 * bound means the LAST (lowest-index) matching overwrite wins, which is
 * always the correct segment — see `terrain.ts::rampColor`'s doc for the
 * CPU version this mirrors exactly.
 */
function rampColorNode(stops: readonly Stop[], x: FloatNode): Vec3Node {
  const last = stops[stops.length - 1]
  if (!last) throw new Error('rampColorNode: stops must be non-empty')
  let result: Vec3Node = vec3(last[1], last[2], last[3])
  for (let i = stops.length - 2; i >= 0; i--) {
    const a = stops[i]
    const b = stops[i + 1]
    if (!a || !b) continue
    const t = clamp(x.sub(a[0]).div(b[0] - a[0]), 0, 1)
    const segColor = mix(vec3(a[1], a[2], a[3]), vec3(b[1], b[2], b[3]), t)
    result = x.lessThanEqual(float(b[0])).select(segColor, result)
  }
  return result
}

/** TSL mirror of `terrain.ts::iceElevationLoweringM`. */
export function iceElevationLoweringMNode(latDegNode: FloatNode): FloatNode {
  return smoothstep(float(ICE_LAT_START_DEG), float(ICE_LAT_END_DEG), abs(latDegNode)).mul(ICE_MAX_LOWERING_M)
}

/** Latitude in degrees from a unit direction, this app's Z-polar convention (`math/wgs84.ts`'s doc) — geocentric, not geodetic (the up-to-11.5-arcmin difference is irrelevant at the ~20-degree width of the ice-blend band this feeds). */
export function latitudeDegNode(dirNode: Vec3Node): FloatNode {
  return asin(clamp(dirNode.z, -1, 1)).mul(180 / Math.PI)
}

/** TSL mirror of `terrain.ts::vegetationIndex`. */
export function vegetationIndexNode(logTauNode: FloatNode, aridLogTau: number, lushLogTau: number): FloatNode {
  return smoothstep(float(aridLogTau), float(lushLogTau), logTauNode)
}

/** TSL mirror of `terrain.ts::highLatitudeCoolness`. */
export function highLatitudeCoolnessNode(latDegNode: FloatNode): FloatNode {
  return smoothstep(float(HIGH_LAT_COOL_START_DEG), float(HIGH_LAT_COOL_END_DEG), abs(latDegNode)).mul(HIGH_LAT_COOL_MAX)
}

/**
 * TSL mirror of `terrain.ts::landColor` — elevation (with `latDegNode`'s
 * ice-lowering folded in by the caller, `hypsometricColorNode` below)
 * still drives light-to-dark and the snowline; `vegetationIndexNode`
 * drives the green-to-tan hue between `LAND_STOPS_ARID` and
 * `LAND_STOPS_LUSH`, which converge above 2800m by construction (see
 * `terrain.ts`'s doc on the two ramps).
 */
export function landColorNode(elevationNode: FloatNode, vegetationIndexNode_: FloatNode): Vec3Node {
  const arid = rampColorNode(LAND_STOPS_ARID, elevationNode)
  const lush = rampColorNode(LAND_STOPS_LUSH, elevationNode)
  return mix(arid, lush, clamp(vegetationIndexNode_, 0, 1))
}

/**
 * TSL mirror of `terrain.ts::hypsometricColor`, including its latitude-driven
 * ice, high-latitude coolness, and density-driven vegetation terms, AND its
 * `oceanProgress` gating: the ocean branch's elevation input is scaled by
 * `oceanProgressNode` (clamped to [0,1]) before hitting `OCEAN_STOPS`, so
 * ocean colour runs from the shoreline stop toward true depth as bathymetry
 * rises — see `hypsometricColor`'s doc for why this (not displacement, which
 * is sub-pixel at the piece's opening altitude) is what actually carries the
 * collapse beat. LAND colour is untouched by `oceanProgressNode`.
 */
export function hypsometricColorNode(
  elevationNode: FloatNode,
  landFractionNode: FloatNode,
  latDegNode: FloatNode,
  vegetationIndexNode_: FloatNode,
  oceanProgressNode: FloatNode,
): Vec3Node {
  const biome = landColorNode(elevationNode.add(iceElevationLoweringMNode(latDegNode)), vegetationIndexNode_)
  const cool = highLatitudeCoolnessNode(latDegNode)
  const land = mix(biome, vec3(HIGH_LAT_COOL_COLOR[0], HIGH_LAT_COOL_COLOR[1], HIGH_LAT_COOL_COLOR[2]), cool)
  const ocean = rampColorNode(OCEAN_STOPS, elevationNode.mul(clamp(oceanProgressNode, 0, 1)))
  return mix(ocean, land, clamp(landFractionNode, 0, 1))
}

/** TSL mirror of `terrain.ts::terrainDisplacementM`. */
export function terrainDisplacementNode(
  elevationNode: FloatNode,
  landFractionNode: FloatNode,
  exaggerationNode: FloatNode,
  bathymetryNode: FloatNode,
): FloatNode {
  const land = elevationNode.mul(exaggerationNode)
  const ocean = elevationNode.mul(clamp(bathymetryNode, 0, 1))
  return mix(ocean, land, clamp(landFractionNode, 0, 1))
}

/**
 * Relief-shading normal from 4 neighbour elevation taps, direction-
 * perturbed rather than UV-offset — a cube map has no single UV space that
 * survives a face-boundary crossing, but perturbing the SAMPLING DIRECTION
 * by a small angle and letting hardware cube sampling pick the (possibly
 * different) face handles that transparently, the same reason every other
 * cube sample in this codebase is direction-based rather than face+uv.
 *
 * `texelAngularWidthRad` is a plain JS number (from the loaded atlas's
 * `km_per_texel`, converted to radians via `R_MEAN`), not a runtime node —
 * baked into the graph at material-build time, matching how `bakeCube.ts`
 * and friends treat face geometry as build-time constants throughout.
 *
 * Mirrors `terrain.ts`'s `tangentBasis` + `centralDifferenceSlope` +
 * `heightfieldNormal` + `applyTangentSpaceBump` chain exactly, just fused
 * into one function since TSL has no equivalent of returning a small
 * struct of intermediate nodes without extra ceremony — AND, as of
 * `landFractionNode`, `terrain.ts::effectiveExaggeration`: the slope is
 * scaled by an exaggeration factor gated by land fraction (1 on pure
 * ocean, `exaggerationNode` on pure land), not by `exaggerationNode`
 * unconditionally as before. That was a real bug, not a hypothetical one —
 * spec 7.5 says marine relief is never exaggerated, and displacement
 * already honoured that (`terrainDisplacementNode`'s ocean term has no `k`
 * at all), but this function was multiplying every fragment's slope by the
 * land exaggeration factor regardless of land/ocean, so the SEAFLOOR'S
 * SHADING NORMAL — never its actual displaced position — was lit as if
 * bathymetry were exaggerated by k (measured ~3.6, `K_MAX`, essentially
 * throughout the piece).
 */
export function terrainNormalNode(
  elevationTex: THREE.CubeTexture,
  dirNode: Vec3Node,
  exaggerationNode: FloatNode,
  landFractionNode: FloatNode,
  texelAngularWidthRad: number,
): Vec3Node {
  const n = dirNode.normalize()
  // tangentBasis: pick a helper axis not parallel to n, matching terrain.ts exactly.
  const helper = abs(n.z).lessThan(0.99).select(vec3(0, 0, 1), vec3(1, 0, 0))
  const tangent = cross(helper, n).normalize()
  const bitangent = cross(n, tangent) // unit: n, tangent already unit and orthogonal

  const eps = float(texelAngularWidthRad)
  const dirPX = n.add(tangent.mul(eps)).normalize()
  const dirNX = n.sub(tangent.mul(eps)).normalize()
  const dirPY = n.add(bitangent.mul(eps)).normalize()
  const dirNY = n.sub(bitangent.mul(eps)).normalize()

  const hPX = sampleElevationNode(elevationTex, dirPX)
  const hNX = sampleElevationNode(elevationTex, dirNX)
  const hPY = sampleElevationNode(elevationTex, dirPY)
  const hNY = sampleElevationNode(elevationTex, dirNY)

  // centralDifferenceSlope: spacing is the FULL separation between the two
  // taps (angle 2*eps), converted to metres via the mean radius.
  const spacingM = float(WGS84.R_MEAN * 2 * texelAngularWidthRad)
  // effectiveExaggeration: 1 (unexaggerated) on pure ocean, exaggerationNode
  // on pure land — see this function's doc for why this must not be
  // exaggerationNode unconditionally.
  const slopeExaggeration = mix(float(1), exaggerationNode, clamp(landFractionNode, 0, 1))
  const dhdx = hPX.sub(hNX).div(spacingM).mul(slopeExaggeration)
  const dhdy = hPY.sub(hNY).div(spacingM).mul(slopeExaggeration)

  // heightfieldNormal([-dhdx, -dhdy, 1]).normalize(), then applyTangentSpaceBump.
  const bump = vec3(dhdx.negate(), dhdy.negate(), float(1)).normalize()
  const bumped = tangent.mul(bump.x).add(bitangent.mul(bump.y)).add(n.mul(bump.z))
  return bumped.normalize()
}

/** TSL mirror of `terrain.ts::detailFadeByDistance`. */
export function detailFadeByDistanceNode(viewDistanceNode: FloatNode, fullStrengthM: number, fadeOutM: number): FloatNode {
  return float(1).sub(smoothstep(float(fullStrengthM), float(fadeOutM), viewDistanceNode))
}

/**
 * Fine-scale, NORMAL-ONLY detail bump — see `terrain.ts`'s "Surface
 * detail" section doc for why this is a lighting trick (legitimate) and
 * not fabricated terrain (not legitimate): it perturbs the LIGHTING
 * normal handed in (`baseNormalNode`, e.g. `terrainNormalNode`'s coarse
 * relief output) and returns a new shading normal; it is never wired to
 * `positionGeometry`/`positionNode` anywhere and must not be.
 *
 * Procedural MaterialX fractal noise (same GPU-portable choice as
 * `dissolveNodes.ts`'s erosion fronts, for the same reason: not
 * reproducible across vendors if built from a CPU-style hash instead)
 * stands in for a heightfield the atlas doesn't have resolution for, finite-
 * differenced along the BASE normal's own tangent/bitangent (i.e. this
 * bumps the already-bumped relief normal further, in its own local frame —
 * standard layered detail-normal-mapping practice) exactly like
 * `terrainNormalNode`'s real-elevation-tap version, just fed by noise
 * instead of a texture. `detailStrengthNode` (expected to already fold in
 * `detailFadeByDistanceNode` AND a maximum-amplitude constant — see
 * `sweep/Matter.ts`) linearly blends between the untouched base normal (0)
 * and the fully bumped one (1).
 */
/**
 * Central-difference step for `detailBumpedNormalNode`'s finite-difference
 * gradient, in the SAME p-space the noise itself is sampled in (i.e.
 * already implicitly scaled by `DETAIL_FREQUENCY` - this function knows
 * nothing about metres or the atlas, only about the noise's own octave
 * structure).
 *
 * `mx_fractal_noise_float` (see `node_modules/three`'s implementation)
 * samples octave `i` (0-indexed) at p-space frequency `lacunarity^i`
 * relative to the base octave, so the base octave's period is 1 p-unit
 * (the underlying Perlin lattice spacing) and the FINEST octave's period is
 * `1 / lacunarity^(octaves-1)`. A two-tap central difference is only a
 * faithful estimate of a signal's slope while its step stays a fraction of
 * that signal's own period; once the step approaches or exceeds a full
 * period, the two taps land on essentially uncorrelated noise values and
 * that octave's contribution to `dh` averages toward zero instead of
 * resolving its actual gradient. This is exactly what a flat `eps = 0.5`
 * did to octaves 2 and 3 here: 2.6x the finest octave's own wavelength, so
 * they contributed almost nothing and the detail normal was effectively
 * driven by the base octave alone. Scaling `eps` to a FRACTION of the
 * finest octave's period keeps the step inside that octave's own Nyquist
 * limit regardless of how many octaves or what lacunarity gets configured.
 */
export function detailGradientEps(octaves: number, lacunarity: number, fractionOfFinestPeriod = 0.5): number {
  const finestOctavePeriod = 1 / lacunarity ** (octaves - 1)
  return fractionOfFinestPeriod * finestOctavePeriod
}

export function detailBumpedNormalNode(
  baseNormalNode: Vec3Node,
  dirNode: Vec3Node,
  detailStrengthNode: FloatNode,
  frequency: number,
): Vec3Node {
  const n = baseNormalNode.normalize()
  const helper = abs(n.z).lessThan(0.99).select(vec3(0, 0, 1), vec3(1, 0, 0))
  const tangent = cross(helper, n).normalize()
  const bitangent = cross(n, tangent)

  const p = dirNode.normalize().mul(frequency)
  const octaves = 3
  const lacunarity = 2.3
  const eps = detailGradientEps(octaves, lacunarity)
  const hC = mx_fractal_noise_float(p, octaves, lacunarity, 0.5, 1.0)
  const hX = mx_fractal_noise_float(p.add(tangent.mul(eps)), octaves, lacunarity, 0.5, 1.0)
  const hY = mx_fractal_noise_float(p.add(bitangent.mul(eps)), octaves, lacunarity, 0.5, 1.0)

  const dhdx = hX.sub(hC).div(eps)
  const dhdy = hY.sub(hC).div(eps)

  // heightfieldNormal([-dhdx, -dhdy, 1]).normalize(), expressed in the base
  // normal's own tangent frame - mirrors terrainNormalNode's tail exactly.
  const bump = vec3(dhdx.negate(), dhdy.negate(), float(1)).normalize()
  const bumped = tangent.mul(bump.x).add(bitangent.mul(bump.y)).add(n.mul(bump.z))
  return mix(n, bumped.normalize(), clamp(detailStrengthNode, 0, 1))
}

/**
 * TSL mirrors of `terrain.ts`'s atmospheric-scattering and ocean-specular
 * functions, term-for-term — see each CPU function's doc for the physical
 * reasoning. Tunable knobs (`maxPathLength`, `density`, `width`,
 * `nightFloor`, `shininess`, `f0`, `strength`) are plain JS numbers baked
 * into the graph at material-build time, the same convention as
 * `terrainNormalNode`'s `texelAngularWidthRad` and `nightFactorNode`'s
 * `terminatorWidth` — only genuinely per-fragment values (angles, path
 * length) are nodes.
 */

/** TSL mirror of `terrain.ts::atmosphericPathLength`. */
export function atmosphericPathLengthNode(cosViewAngleNode: FloatNode, maxPathLength: number): FloatNode {
  const c = max(cosViewAngleNode, 1 / maxPathLength)
  return min(float(1).div(c), maxPathLength)
}

/** TSL mirror of `terrain.ts::rayleighScatterIntensity`. */
export function rayleighScatterIntensityNode(pathLengthNode: FloatNode, wavelengthWeight: number, density: number): FloatNode {
  return float(1).sub(exp(pathLengthNode.mul(-wavelengthWeight * density)))
}

/** TSL mirror of `terrain.ts::atmosphereScatterColor`. */
export function atmosphereScatterColorNode(cosViewAngleNode: FloatNode, density: number, maxPathLength: number): Vec3Node {
  const pathLength = atmosphericPathLengthNode(cosViewAngleNode, maxPathLength)
  return vec3(
    rayleighScatterIntensityNode(pathLength, RAYLEIGH_RGB_WEIGHTS[0], density),
    rayleighScatterIntensityNode(pathLength, RAYLEIGH_RGB_WEIGHTS[1], density),
    rayleighScatterIntensityNode(pathLength, RAYLEIGH_RGB_WEIGHTS[2], density),
  )
}

/** TSL mirror of `terrain.ts::terminatorWarmthWeight` — a Gaussian in `ndotSun`, see that function's doc for why (a narrow, physically-shaped band, not a wide linear tent). */
export function terminatorWarmthWeightNode(ndotSunNode: FloatNode, sigma: number): FloatNode {
  const r = ndotSunNode.div(sigma)
  return exp(r.mul(r).negate())
}

/** TSL mirror of `terrain.ts::terminatorSunTransmittance` — MULTIPLY this into a scatter colour; see that function's doc for why mixing toward a constant amber instead is the bug it exists to prevent. */
export function terminatorSunTransmittanceNode(
  warmthNode: FloatNode,
  extraPathAtTerminator: number,
  density: number,
): Vec3Node {
  const path = clamp(warmthNode, 0, 1).mul(extraPathAtTerminator)
  return vec3(
    exp(path.mul(RAYLEIGH_RGB_WEIGHTS[0] * density).negate()),
    exp(path.mul(RAYLEIGH_RGB_WEIGHTS[1] * density).negate()),
    exp(path.mul(RAYLEIGH_RGB_WEIGHTS[2] * density).negate()),
  )
}

/** TSL mirror of `terrain.ts::twilightAmbientScale` — see that function for why ambient has to fall off at all, and why its ramp is far wider than the terminator effects beside it. */
export function twilightAmbientScaleNode(ndotSunNode: FloatNode, nightFloor: number): FloatNode {
  const t = smoothstep(float(ASTRONOMICAL_TWILIGHT_NDOTSUN), float(0), ndotSunNode)
  return mix(float(nightFloor), float(1), t)
}

/** TSL mirror of `terrain.ts::atmosphereSunAvailability`. */
export function atmosphereSunAvailabilityNode(ndotSunNode: FloatNode, width: number, nightFloor: number): FloatNode {
  const t = clamp(ndotSunNode.add(width).div(2 * width), 0, 1)
  return mix(float(nightFloor), float(1), t)
}

/** TSL mirror of `terrain.ts::schlickFresnel`. */
export function schlickFresnelNode(cosViewAngleNode: FloatNode, f0: number): FloatNode {
  const c = clamp(cosViewAngleNode, 0, 1)
  return float(f0).add(float(1 - f0).mul(float(1).sub(c).pow(5)))
}

/** TSL mirror of `terrain.ts::specularLobe`. */
export function specularLobeNode(ndotHNode: FloatNode, shininess: number): FloatNode {
  return max(ndotHNode, 0).pow(shininess)
}

/** TSL mirror of `terrain.ts::oceanSpecularIntensity`. */
export function oceanSpecularIntensityNode(
  ndotHNode: FloatNode,
  cosViewAngleNode: FloatNode,
  shininess: number,
  f0: number,
  strength: number,
): FloatNode {
  return specularLobeNode(ndotHNode, shininess).mul(schlickFresnelNode(cosViewAngleNode, f0)).mul(strength)
}

/** GHSL's built-up fraction: raw 0..1, no log — see `field/terrain.ts`'s city-lights section doc. */
export function sampleBuiltUpFractionNode(urbanTex: THREE.CubeTexture, dirNode: Vec3Node): FloatNode {
  return cubeTexture(urbanTex, dirNode).level(float(0)).r
}

/** TSL mirror of `terrain.ts::nightFactor` — `ndotSun` MUST be the base (unbumped) surface normal's dot with the sun direction, not the relief-shading normal (see that function's doc for why). */
export function nightFactorNode(ndotSun: FloatNode, terminatorWidth: number): FloatNode {
  const w = float(terminatorWidth)
  return smoothstep(w.negate(), w, ndotSun.negate())
}

/** TSL mirror of `terrain.ts::cityLightIntensity`. */
export function cityLightIntensityNode(builtUpFractionNode: FloatNode, gamma: number): FloatNode {
  return clamp(builtUpFractionNode, 0, 1).pow(gamma)
}

/**
 * The density cube's B channel: per-pixel bootstrap coefficient of variation
 * (van den Hoogen et al. 2019, 100 iterations). Raw, dimensionless, no
 * encoding — unlike R, which `loadCube.ts` re-encodes to log(tau);
 * `densityFaceToTauFace` passes G/B/A through untouched, so what reaches the
 * GPU here is exactly what the pipeline baked.
 */
export function sampleBootstrapCvNode(densityTex: THREE.CubeTexture, dirNode: Vec3Node): FloatNode {
  return cubeTexture(densityTex, dirNode).level(float(0)).b
}

/**
 * TSL mirror of `terrain.ts::uncertaintyColor` — the ember ramp, built from
 * that file's exported `UNCERTAINTY_STOPS` through the same `rampColorNode`
 * cascade the terrain ramps use, so there is exactly one copy of the stop
 * table and the CPU and GPU ramps cannot drift apart.
 */
export function uncertaintyColorNode(cvNode: FloatNode): Vec3Node {
  return rampColorNode(UNCERTAINTY_STOPS, cvNode)
}

/**
 * TSL mirror of `terrain.ts::uncertaintyCoverage`.
 *
 * Note the parameter: LAND FRACTION (channel A), never the CV itself. Over
 * open ocean the CV channel reads 0, which is a perfectly plausible-looking
 * "extremely well measured" value — gating on it would paint the ocean as
 * the most certain surface on Earth. See the CPU function's doc.
 */
export function uncertaintyCoverageNode(landFractionNode: FloatNode): FloatNode {
  return smoothstep(float(UNCERTAINTY_LAND_MIN), float(UNCERTAINTY_LAND_FULL), landFractionNode)
}

/** TSL mirror of `terrain.ts::uncertaintyOverlayColor` — see that function's doc for why the overlay REPLACES the veil's tint inside the model's domain instead of modulating it. */
export function uncertaintyOverlayColorNode(
  baseNode: Vec3Node,
  cvNode: FloatNode,
  landFractionNode: FloatNode,
  strengthNode: FloatNode,
): Vec3Node {
  const w = clamp(strengthNode, 0, 1).mul(uncertaintyCoverageNode(landFractionNode))
  return mix(baseNode, uncertaintyColorNode(cvNode), w)
}

// --- Reconstruction filter and output dither ---------------------------------
//
// Two independent fixes for the veil reading as blocky, mirrored here from
// their CPU sources: `loadCube.ts::smoothstepCubeDirection` (the cube's
// reconstruction filter creasing on every texel edge) and
// `terrain.ts::veilDitherScale` (the output contour-banding). Both are used
// by `regimes/Shell.ts`; see each CPU function's doc for the measurement
// that motivated it.

/**
 * TSL mirror of `loadCube.ts::smoothstepCubeDirection` — see that function
 * for the whole derivation, in particular why a component-wise warp needs no
 * per-face branching and why it is provably the identity on face seams.
 *
 * `faceSize` is a plain JS number (the atlas's texels per side, from
 * `loadCube.ts::cubeFaceSize`), not a node: it is fixed for the lifetime of
 * a material, so folding it in at graph-build time keeps the per-step cost
 * to the warp itself. Same convention as `terrainNormalNode`'s
 * `texelAngularWidthRad`.
 *
 * The `max(m, 1e-20)` guard is for the degenerate zero direction only. It
 * cannot arise from the shell's march (every sampled point sits on or above
 * the datum ellipsoid, so `|ps| >= 1`), but a division that can produce NaN
 * inside the hottest loop in the piece is not worth leaving to an invariant
 * held elsewhere in another file.
 */
export function smoothstepCubeDirectionNode(dirNode: Vec3Node, faceSize: number): Vec3Node {
  const a = abs(dirNode)
  const m = max(max(a.x, a.y), a.z)
  // Face coordinates: the major component becomes exactly +/-1, the other
  // two exactly +/- this face's (u, v).
  const c = dirNode.div(max(m, float(1e-20)))
  // Texel-centre space: texel j's centre sits at p = j.
  const p = c.add(1).mul(0.5 * faceSize).sub(0.5)
  const i = floor(p)
  const f = p.sub(i)
  // smoothstep(f) = f*f*(3 - 2f), written so the scalar broadcasts onto the
  // vec3 from the right (`f.mul(-2).add(3)`) rather than `float(3).sub(f)`.
  const warped = i.add(f.mul(f).mul(f.mul(-2).add(3)))
  return warped.add(0.5).div(faceSize).mul(2).sub(1)
}

/** TSL mirror of `terrain.ts::interleavedGradientNoise`, on `screenCoordinate` (window-relative fragment coordinates, in pixels). */
export function interleavedGradientNoiseNode(): FloatNode {
  const t = fract(screenCoordinate.x.mul(0.06711056).add(screenCoordinate.y.mul(0.00583715)))
  return fract(t.mul(52.9829189))
}

/** TSL mirror of `terrain.ts::veilDitherScale` — multiply a shell fragment's whole premultiplied vec4 by this; see the CPU function for why one shared scalar rather than per-channel noise. */
export function veilDitherScaleNode(amplitude: number = VEIL_DITHER_AMPLITUDE): FloatNode {
  return interleavedGradientNoiseNode().sub(0.5).mul(amplitude).add(1)
}
