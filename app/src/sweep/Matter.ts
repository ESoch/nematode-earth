import * as THREE from 'three/webgpu'
import {
  Fn, vec3, vec4, float, max, dot, mix, clamp, length, normalWorld, positionWorld, positionGeometry,
  normalGeometry, cameraPosition, screenCoordinate, interleavedGradientNoise, Discard,
} from 'three/tsl'
import { matterAlphaNode, type DissolveUniforms } from './dissolveNodes'
import { VEGETATION_ARID_LOG_TAU, VEGETATION_LUSH_LOG_TAU } from '../field/terrain'
import {
  sampleElevationNode, sampleLandMaskNode, sampleLogTauNode, vegetationIndexNode,
  hypsometricColorNode, terrainDisplacementNode, terrainNormalNode, detailBumpedNormalNode,
  detailFadeByDistanceNode, latitudeDegNode, atmosphereScatterColorNode, terminatorWarmthWeightNode,
  atmosphereSunAvailabilityNode, terminatorSunTransmittanceNode, twilightAmbientScaleNode,
  oceanSpecularIntensityNode, sampleBuiltUpFractionNode,
  nightFactorNode, cityLightIntensityNode,
} from '../field/terrainNodes'

/**
 * The opaque globe writes depth in front of the nematode shell. The
 * atmosphere front dims the rim and city lights. Water and rock fronts
 * discard whole fragments. Once both fronts finish, the viewer skips the
 * mesh until playback or scrubbing brings it back.
 *
 * ETOPO elevation and land fraction shape the surface. During the drain,
 * bathymetry moves the ocean surface from sea level to the measured bed.
 * The same progress changes its colour, reveals seafloor slopes and fades
 * the water reflection. The water dissolve starts after the drain finishes.
 * Land keeps its elevation exaggeration throughout.
 *
 * GHSL built-up surface provides a visual proxy for city lights. The
 * soil-abundance model supplies the green-to-tan palette on land.
 */

/**
 * Sub-solar point ~10°N 40°E.
 *
 * THE LATITUDE, 10°N, is a ruling and must stay near it. A point at
 * latitude φ has `min(ndotSun) = -cos(φ + φ_sun)` over a full rotation, so
 * it reaches the night side at all only when `φ + φ_sun < 90°`. An earlier
 * value put the sun at 54.2°N, which caps night at 35.8°N — putting London,
 * Paris, Berlin, Moscow, Beijing, Tokyo, New York and Chicago in PERMANENT
 * daylight, from any camera angle, at any point in the drift, so the city
 * lights could not have been seen on most of the world's cities even once
 * they worked. At 10°N the bound moves to 80°N, i.e. every inhabited place
 * on Earth. ~10°N is also an ordinary sub-solar latitude (roughly early
 * spring), so neither pole sits pinned in permanent day. `matter.test.ts`
 * asserts the bound directly on this constant.
 *
 * THE LONGITUDE was 155°E, and the comment that carried it said the
 * longitude "only sets where the viewer happens to come in". That was the
 * defect. The opening pose was then at longitude 0, 155° away, so the
 * piece opened on, and descended into, the NIGHT side.
 * Measured at the opening pose with the drift stopped, matter intact:
 * frame mean 0.746/255 at 11,500 km, 0.0025 at 1,000 km, and exactly 0.000
 * (0.00% non-zero pixels) at 381 km, against 36.9 / 170.0 / 170.1 at the
 * sub-solar point. A viewer who loaded the page and scrolled descended into
 * a black screen.
 *
 * 40°E is chosen against the current opening pose (10°N, 10°W), not picked
 * for a landmass. The sub-camera point begins about 50° west of the sun,
 * with ndotSun ≈ 0.654. Idle rotation now takes 12 minutes per turn and
 * stops when playback begins, so the guided reveal retains that lighting.
 *
 * The night hemisphere is then 130°E through -50°E, which holds Japan,
 * Korea, eastern Australia and every North American city — city lights,
 * the setup half of spec 6.3's rhyme, remain available by orbiting toward
 * the night side. The opening at 10°W still shows Africa and Europe rather
 * than an ocean face. `matter.test.ts` asserts the lighting and framing
 * properties against `OPENING_POSE` and `SUN_DIR`.
 */
export const SUN_DIR = new THREE.Vector3(0.7544, 0.6330, 0.1736).normalize()
const AMBIENT = 0.10
/**
 * Fraction of AMBIENT that survives deep on the night side — airglow,
 * starlight and moonlight, which are genuinely there and genuinely tiny.
 * Small enough that city lights are unambiguously the brightest thing on
 * the unlit hemisphere, non-zero so the planet keeps a readable silhouette
 * against space rather than vanishing into it.
 */
const AMBIENT_NIGHT_FLOOR = 0.08
const DIFFUSE = 0.85

/**
 * The viewer exposes the whole scene at 8× to show the translucent shell.
 * Scale the terrain palette so its brightest land and ocean colours retain
 * detail at the same exposure.
 */
const EXPOSURE_COMPENSATION = 1 / 8 // must track viewer.ts's default `exposure`
/**
 * Cheap analytic Rayleigh atmosphere: density-with-altitude is stood in by
 * the classic airmass/secant growth of path length toward grazing view
 * angles (`atmosphericPathLengthNode`), and 1/wavelength^4 scattering
 * (`RAYLEIGH_RGB_WEIGHTS`, `field/terrain.ts`) is what makes the limb blue
 * and lets it warm toward the terminator — not a physically complete
 * integral, but the right SHAPE. Replaces the old flat-tinted fresnel rim
 * entirely; still gated by the same "atmosphere" dissolve front (see the
 * module doc's THREE LAYERS section above), so it is still the first thing
 * to visibly fade as erosion starts.
 *
 * Density/path-length picked so NO channel gets near full saturation even
 * at the grazing limb. `rayleighScatterIntensity` is a Beer-Lambert
 * saturating curve — push density too high and every channel converges on
 * 1 regardless of its wavelength weight, i.e. WHITE, which is exactly the
 * "limb blown to white" bug team lead flagged. At 0.14, red tops out
 * around 0.45 and blue around 0.97 at the limb — clearly blue, never white.
 */
const ATMOSPHERE_DENSITY = 0.14
const ATMOSPHERE_MAX_PATH_LENGTH = 10
/** Restrained per spec 6.2 (near-monochrome veil) AND so the atmosphere never outshines the terrain or the veil — "a rim you notice, not the brightest thing on the planet" (team lead). */
const ATMOSPHERE_STRENGTH = 0.16
/** Real Earth has NO Rayleigh scattering on the unlit hemisphere — nothing there for it to scatter — so this is 0, not a glow-preserving fudge. City lights sit against true black, not a haze (team lead's explicit check). */
const ATMOSPHERE_NIGHT_FLOOR = 0

/**
 * One shared Gaussian weight (`terminatorWarmthWeightNode`) feeds the single
 * `sunTransmittance` colour below, which in turn tints EVERYTHING sunlight
 * touches — the diffuse term, the ocean glint, and the atmosphere's own
 * scattering. That is why the three agree at the terminator without being
 * tuned against each other.
 *
 * `sigma`, in `ndotSun` units — tight on purpose. A first pass used a
 * linear/quadratic tent with a much wider effective half-width, which put
 * the warm band ~40-50 degrees wide ("reads as a planet-scale dust storm"
 * — team lead). The physical effect (the SUN's own path length through
 * the atmosphere, secant-law steep only near the horizon — see
 * `terminatorWarmthWeight`'s doc) is narrow and sharply peaked, so this is
 * now a Gaussian: 0.06 keeps the visible band within a few degrees of the
 * terminator and negligible by ~10-15 degrees (`terrain.test.ts` asserts
 * this directly on this exact constant).
 */
const TERMINATOR_GLOW_WIDTH = 0.06
/**
 * Extra optical path the SUNLIGHT itself traverses at the terminator, in
 * `atmosphericPathLength` units — the input to
 * `terminatorSunTransmittanceNode`, which is what actually makes the
 * atmosphere warm there.
 *
 * There is no hand-picked amber constant here any more, and there were two
 * separate attempts at one. The atmosphere mixed toward it — a mix returns
 * the constant outright at full warmth, discarding the scatter along with
 * its view-angle dependence — and the terrain ADDED it, which against
 * near-black unlit ground is all signal and no shading. Both produced the
 * identical artefact: a hard-edged warm stripe of uniform brightness ruled
 * straight across the disc, brightest exactly where the ground beneath it
 * was darkest.
 *
 * Extinction on the existing 1/lambda^4 weights gets the same orange out of
 * physics rather than out of a colour picker, and being bounded in (0,1] it
 * cannot brighten anything at any parameter value. 10 puts blue
 * transmittance near 0.03 and red near 0.55 at the terminator: genuinely
 * red-dominant, and 12x dimmer at the disc centre than the stripe it
 * replaces. `matter.test.ts` asserts both failed shapes are gone
 * structurally, since each was reintroduced once already.
 */
const TERMINATOR_SUN_PATH = 10

/**
 * A broad Blinn-Phong sun-glint lobe with Schlick Fresnel brightening.
 * The reflection fades as the sweep exposes the dry seabed.
 */
const OCEAN_SPECULAR_SHININESS = 50
const OCEAN_SPECULAR_F0 = 0.02
const OCEAN_SPECULAR_STRENGTH = 0.45
const OCEAN_SPECULAR_COLOR = new THREE.Color(0.85, 0.92, 1.0)

/** Warm amber — the one deliberately warm note in an otherwise cold-and-muted piece, so it reads as human. */
const CITY_LIGHT_COLOR = new THREE.Color(1.0, 0.72, 0.4)
const CITY_LIGHT_GAMMA = 0.42
/** Faint by design ("something you notice, not something that shouts") — tune once viewed under the piece's actual exposure. */
const CITY_LIGHT_STRENGTH = 0.4
const TERMINATOR_WIDTH = 0.08

/**
 * Fine-scale, NORMAL-ONLY surface detail (`detailBumpedNormalNode`) — see
 * `field/terrain.ts`'s "Surface detail" section for why this is a lighting
 * trick, not fabricated terrain: `material.positionNode` below never reads
 * any of this. `DETAIL_FREQUENCY` is a plain scalar applied to the unit
 * direction (same convention as `Dissolve.ts`'s `noiseFreq`, whose erosion
 * fronts run 0.5-9.0 on the same unit-sphere scale) — well above the
 * erosion fronts' broad, continental-scale noise, since this stands in for
 * genuinely small terrain features the elevation atlas has no resolution
 * for. `DETAIL_MAX_STRENGTH` keeps it a texture, not a takeover, even at
 * full fade-in.
 */
const DETAIL_FREQUENCY = 55
const DETAIL_MAX_STRENGTH = 0.45
/** Full detail once within this many metres of the surface... */
const DETAIL_FULL_STRENGTH_M = 30_000
/** ...fully faded out by this many metres, so the high spatial frequency never has a chance to alias/shimmer at cruise altitude (`detailFadeByDistanceNode`). */
const DETAIL_FADE_OUT_M = 300_000

export interface MatterOpts {
  atmosphere: DissolveUniforms
  water: DissolveUniforms
  rock: DissolveUniforms
  /** The fraction of seafloor relief exposed by the drain, from 0 to 1. */
  bathymetry: THREE.UniformNode<'float', number>
  /** `math/descent.ts`'s `exaggerationAt(altM)` — land only; marine relief is never exaggerated (spec 7.5). */
  exaggeration: THREE.UniformNode<'float', number>
  elevationTexture: THREE.CubeTexture
  /** Soil optical depth supplies the log-scaled land tint through sampleLogTauNode. */
  densityTexture: THREE.CubeTexture
  /** GHSL built-up fraction in the red channel supplies the night-light proxy. */
  urbanTexture: THREE.CubeTexture
  /** One elevation-atlas texel's angular width in radians (from the loaded atlas's `km_per_texel`) — sets the relief-normal's neighbour-tap spacing. A plain JS number baked into the graph at build time, not a uniform. */
  texelAngularWidthRad: number
}

export function buildMatterMaterial(opts: MatterOpts): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial()
  material.transparent = false

  const shade = Fn(() => {
    // Sphere-local normal, unaffected by the mesh's own position - correct
    // under floating-origin translation without any extra plumbing, unlike
    // Shell.ts's manual ray math (see ShellOpts.cameraWorldPosition's doc).
    const dir = normalWorld.normalize()
    const elevation = sampleElevationNode(opts.elevationTexture, dir)
    const landFraction = sampleLandMaskNode(opts.elevationTexture, dir)

    const waterAlpha = matterAlphaNode(dir, 'water', opts.water)
    const rockAlpha = matterAlphaNode(dir, 'rock', opts.rock)
    const alpha = mix(waterAlpha, rockAlpha, clamp(landFraction, 0, 1))

    const ditherThreshold = interleavedGradientNoise(screenCoordinate)
    Discard(alpha.lessThan(ditherThreshold))

    const latDeg = latitudeDegNode(dir)
    // Vegetation as colour: the SAME cube Shell.ts turns into the veil's
    // optical depth, here driving land's green-to-tan axis (see
    // field/terrain.ts's "Vegetation as colour" section for why that's
    // thematically exact, not just visually convenient).
    const logTau = sampleLogTauNode(opts.densityTexture, dir)
    const vegIndex = vegetationIndexNode(logTau, VEGETATION_ARID_LOG_TAU, VEGETATION_LUSH_LOG_TAU)
    // Reveal the measured seabed while ocean fragments are still opaque.
    const oceanProgress = clamp(opts.bathymetry, 0, 1)
    const baseColor = hypsometricColorNode(elevation, landFraction, latDeg, vegIndex, oceanProgress)
    const coarseNormal = terrainNormalNode(
      opts.elevationTexture, dir, opts.exaggeration, landFraction, oceanProgress, opts.texelAngularWidthRad,
    )
    const waterSurface = float(1).sub(clamp(landFraction, 0, 1)).mul(float(1).sub(oceanProgress))

    // Fine-scale detail, normal-only (see DETAIL_* constants' doc) - camera
    // sits pinned at the render origin under floating-origin translation
    // (see viewDir's own comment below), so positionWorld's own length IS
    // the camera-to-fragment distance, no extra uniform needed.
    const viewDistance = length(positionWorld)
    const detailStrength = detailFadeByDistanceNode(viewDistance, DETAIL_FULL_STRENGTH_M, DETAIL_FADE_OUT_M)
      .mul(DETAIL_MAX_STRENGTH).mul(float(1).sub(waterSurface))
    const normal = detailBumpedNormalNode(coarseNormal, dir, detailStrength, DETAIL_FREQUENCY)

    const sunDir = vec3(SUN_DIR.x, SUN_DIR.y, SUN_DIR.z)
    const ndotl = max(dot(normal, sunDir), 0.0)

    // View direction: camera sits pinned at the render origin under
    // floating-origin translation (see Shell.ts's doc), so cameraPosition
    // reads (0,0,0) and this reduces to normalize(-positionWorld) with no
    // extra uniform needed - same reasoning as `dir` above.
    const viewDir = cameraPosition.sub(positionWorld).normalize()
    const atmosphereAlpha = matterAlphaNode(dir, 'atmosphere', opts.atmosphere)

    // Base (unbumped) dir for both the day/night test and the atmosphere -
    // see nightFactorNode's doc for why the relief normal would be wrong
    // for a terminator test (it would flicker across every ridge/gully).
    const ndotSun = dot(dir, sunDir)
    const cosView = clamp(dot(dir, viewDir), 0, 1)
    const warmth = terminatorWarmthWeightNode(ndotSun, TERMINATOR_GLOW_WIDTH)

    /**
     * The colour of the SUNLIGHT arriving at this fragment — white across
     * the day side, reddening to deep orange in the last few degrees before
     * the terminator as its own path through the atmosphere lengthens.
     *
     * One quantity, used by all three things sunlight touches: the diffuse
     * term, the sun glint, and the atmosphere's own scattering. That is why
     * they agree at the terminator without three separately-tuned bands,
     * and it is bounded in (0,1] so none of them can be BRIGHTENED there —
     * the failure mode both earlier versions had.
     */
    const sunTransmittance = terminatorSunTransmittanceNode(warmth, TERMINATOR_SUN_PATH, ATMOSPHERE_DENSITY)

    // Sunset light on the ground, not amber emitted BY the ground. Ambient
    // stays white and untinted (it stands in for sky and bounce light, which
    // does not travel the sun's grazing path) but it DOES fall away across
    // twilight - see twilightAmbientScale for why, and what applying it flat
    // on the night side cost the city lights.
    const twilight = twilightAmbientScaleNode(ndotSun, AMBIENT_NIGHT_FLOOR)
    const lit = baseColor.mul(ndotl.mul(DIFFUSE).mul(sunTransmittance).add(twilight.mul(AMBIENT)))

    // Rayleigh atmosphere glow: grazing-angle limb brightening with a real
    // blue/red wavelength split, reddened toward the terminator by the same
    // sunlight tint and easing down to nothing on the night side (no
    // sunlight there to scatter) - replaces the old thin fresnel rim (see
    // ATMOSPHERE_* constants' doc above).
    const scatterColor = atmosphereScatterColorNode(cosView, ATMOSPHERE_DENSITY, ATMOSPHERE_MAX_PATH_LENGTH)
    const sunAvailability = atmosphereSunAvailabilityNode(ndotSun, TERMINATOR_GLOW_WIDTH, ATMOSPHERE_NIGHT_FLOOR)
    const atmosphereGlow = scatterColor.mul(sunTransmittance)
      .mul(sunAvailability).mul(ATMOSPHERE_STRENGTH).mul(atmosphereAlpha)

    // Ocean specular sun-glint: Blinn-Phong half-vector lobe, Fresnel-
    // brightened toward grazing angles and confined to the water surface
    // as it drains (see OCEAN_SPECULAR_* constants' doc above).
    // Carries the same sunlight tint, so a glint caught near the terminator
    // is the orange one you actually see from orbit.
    const halfDir = viewDir.add(sunDir).normalize()
    const ndotH = max(dot(dir, halfDir), 0.0)
    const specIntensity = oceanSpecularIntensityNode(
      ndotH, cosView, OCEAN_SPECULAR_SHININESS, OCEAN_SPECULAR_F0, OCEAN_SPECULAR_STRENGTH,
    )
    const specular = vec3(OCEAN_SPECULAR_COLOR.r, OCEAN_SPECULAR_COLOR.g, OCEAN_SPECULAR_COLOR.b)
      .mul(specIntensity).mul(sunTransmittance).mul(waterSurface)

    // City lights (base, unbumped dir - same reasoning as above).
    const night = nightFactorNode(ndotSun, TERMINATOR_WIDTH)
    const builtUp = sampleBuiltUpFractionNode(opts.urbanTexture, dir)
    const cityGlow = cityLightIntensityNode(builtUp, CITY_LIGHT_GAMMA)
      .mul(night).mul(atmosphereAlpha).mul(clamp(landFraction, 0, 1))

    // EXPOSURE_COMPENSATION folded into `color` itself (not applied at the
    // call site) so this stays the exact `vec4(color, 1.0)` matter.test.ts
    // checks for - alpha is still a hardcoded constant, unrelated to any
    // dissolve/alpha value, which is the actual invariant that regex guards.
    const color = lit
      .add(specular)
      .add(atmosphereGlow)
      .add(vec3(CITY_LIGHT_COLOR.r, CITY_LIGHT_COLOR.g, CITY_LIGHT_COLOR.b).mul(cityGlow).mul(CITY_LIGHT_STRENGTH))
      .mul(EXPOSURE_COMPENSATION)

    return vec4(color, 1.0)
  })

  material.fragmentNode = shade()

  // Vertex displacement: real geometric relief (not just normal-mapped
  // shading), so the opaque depth buffer this mesh writes already carries
  // exaggerated mountains and (once revealed) true bathymetry - the
  // ordinary per-fragment depth test against that buffer is what keeps the
  // veil's shell from rendering over a mountain, with no extra depth-clip
  // logic needed in Shell.ts's march (the shell's own far backside is
  // always vastly farther than anything on/near the surface, so the same
  // simple "is opaque matter nearer than the shell's own geometry"
  // occlusion this file already relied on for the flat placeholder holds
  // regardless of terrain).
  material.positionNode = Fn(() => {
    const dir = normalGeometry.normalize()
    const elevation = sampleElevationNode(opts.elevationTexture, dir)
    const landFraction = sampleLandMaskNode(opts.elevationTexture, dir)
    // Match the drain progress used for lighting and colour.
    const oceanProgress = clamp(opts.bathymetry, 0, 1)
    const displacement = terrainDisplacementNode(elevation, landFraction, opts.exaggeration, oceanProgress)
    return positionGeometry.add(dir.mul(displacement))
  })()

  return material
}
