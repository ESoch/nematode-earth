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
 * "Matter": the real, ETOPO-derived planet standing in front of the veil
 * until the sweep erodes it away. Real elevation (land relief AND seafloor
 * bathymetry, both real ETOPO 2022 data, `app/public/data/elevation_*`) and
 * land fraction (the same atlas's G channel — see `sampleLandMaskNode`,
 * which is why the coastline is as sharp as the relief rather than eight
 * times softer) drive shape and colour; nothing here is invented. Dissolving a recognisable Earth is the whole point of the
 * sweep (a placeholder ball dissolving is nothing), so this is the moment
 * the piece stops looking like an experiment and starts looking like the
 * planet Cobb was talking about.
 *
 * This mesh is what makes the sweep's governing rule (spec 6.1) literally
 * true in the render: "the veil renders at full opacity from frame zero,
 * OCCLUDED by matter rather than faded in." The veil's own material
 * (regimes/Shell.ts) is untouched by anything in this file or by the sweep
 * at all - it always computes the same density it always would. What the
 * sweep controls is whether THIS mesh is still standing in front of it.
 * Opaque + normal depth write/test is what does the actual occluding; as
 * fragments here get discarded, the shell material behind them is simply
 * no longer blocked, not faded up. `matter.test.ts` enforces this
 * structurally (a literal `Discard(` and a literal, non-variable
 * `vec4(color, 1.0)` alpha), unchanged and unweakened by this rewrite.
 *
 * THREE LAYERS, the spec's stated erosion order (atmosphere, then water,
 * then rock — `Dissolve.ts`'s three fronts, one per layer):
 *  - Atmosphere: not a separate haze geometry, but the intensity of the
 *    Rayleigh-shaped scattering glow below (see the ATMOSPHERE_* constants)
 *    - it is the first thing to visibly fade, exactly as "atmosphere
 *    clears first" would look, at zero extra geometry cost.
 *  - Water: drives OCEAN fragments' discard only (see `landFraction`-
 *    weighted blend below) — NOT their displacement or colour any more; see
 *    the OCEAN COLLAPSE section below for why those two were pulled apart.
 *  - Rock: drives LAND fragments' discard, unchanged from the placeholder.
 *
 * LAND is always shown at full exaggerated relief from frame zero — it is
 * part of "a recognisable planet," not something the sweep animates into
 * being. Only OCEAN's shape and colour animate (via `bathymetry`).
 *
 * OCEAN COLLAPSE: `SweepState.bathymetry` alone drives both the seafloor's
 * displacement AND its colour — deliberately NOT combined with `waterAlpha`
 * (the water dissolve front) any more. It used to be: `oceanProgress =
 * clamp(max(1 - waterAlpha, opts.bathymetry), 0, 1)`, "as the water
 * dissolves, the surface drops to bathymetry" read literally. A multi-agent
 * audit found this was provably a no-op: `1 - waterAlpha` saturates to
 * EXACTLY 1, network-wide, the instant `frontWater` reaches 1 (`Dissolve.ts`'s
 * `layerAlpha` endpoint-totality — not asymptotic), which happened at
 * t=7.4s under the old timing, well before `opts.bathymetry` ever left 0
 * (t=9.0s) — so the `max` was ALWAYS won by `1 - waterAlpha`, and
 * `bathymetry` never once affected the result. Worse: since the SAME
 * `waterAlpha` also drove the fragment `Discard`, a fragment's displacement
 * progress and its odds of being visible at all were the exact same number
 * on opposite sides of 1 — the more a patch of seafloor had dropped, the
 * more likely it was to already be 100% discarded. Expected visible signal
 * (fraction rendered x how far it had dropped) is bounded by
 * `alpha * (1 - alpha) <= 0.25` by construction; the reveal could not have
 * been seen no matter how it was lit.
 *
 * Fixing the coupling alone would still not have produced a VISIBLE beat:
 * the actual geometric drop (up to ~6 km of bathymetry) was independently
 * measured at well under one screen pixel from the piece's opening
 * altitude. So displacement is fixed because it was WRONG (bathymetry now
 * drives it alone, timed via `Timeline.ts`'s retimed `WIN.bathymetry` to
 * complete before water's own discard front ever moves — see that file's
 * doc), not because the fix alone makes the beat readable. What actually
 * carries the beat is COLOUR: `hypsometricColorNode` now takes the same
 * `oceanProgress` and scales the ocean branch's elevation input by it
 * before the `OCEAN_STOPS` ramp, so the sea visibly deepens — running from
 * the shoreline stop toward the true-depth stop — while still fully opaque.
 * Marine relief is still never exaggerated (spec 7.5) in either the
 * displacement OR the shading normal now — see `terrainNormalNode`'s doc
 * for a second, unrelated case of that rule that had slipped through.
 *
 * CITY LIGHTS (GHSL built-up fraction, `field/terrain.ts`'s city-lights
 * section): the setup half of the piece's central visual rhyme (spec 6.3) —
 * the same built-up fraction that glows warm here becomes a SUPPRESSION on
 * the veil's density once matter is gone (paved ground holds less life),
 * so the brightest coastlines before are the darkest voids after. Only
 * the glow is wired here; the veil-side suppression is a separate,
 * not-yet-calibrated consumer of the same cube (see `viewer.ts`'s wiring
 * comment) and deliberately isn't guessed at in this file.
 *
 * Lights are warm (the only warm thing on screen — the planet is muted
 * blue-green-brown, the veil cold cyan), night-side only (so this reads as
 * "Earth at night," not a data overlay), and tied to the ATMOSPHERE front
 * alongside the atmosphere glow — civilization's glow is the first thing to go dark as
 * erosion starts, well before the rest of the terrain visibly changes,
 * which is what gives a scrubbing viewer room to actually notice the
 * lights going out as its own beat rather than losing it in the rest of
 * the dissolve.
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
 * The veil is genuinely faint (4.06% mean coverage over land, spec 6.2 -
 * PER WALL; see `viewer.ts`'s exposure comment for why the far wall,
 * `regimes/Shell.ts`'s `ShellFarWallOpts`, roughly doubles that near disc
 * centre without changing the underlying density) and needs
 * `viewer.ts`'s default 8x `toneMappingExposure` to read at all - that
 * lift is applied globally, post-render, to everything in the scene, not
 * per-material. Terrain colour authored at a normal-ish albedo (the
 * LAND_STOPS/OCEAN_STOPS ramps in `field/terrain.ts`) blows out to
 * near-white under the SAME 8x lift ("reads as an ice planet" - team
 * lead's finding from the first real render; confirmed it wasn't the
 * colour ramp itself, whose grey/white stops only start at 4500 m -
 * every stop was clipping, not just the snow line).
 *
 * Fix, per team lead's explicit call ("scale the matter's emitted colour
 * ... simplest, keeps a single exposure control meaningful" over a second,
 * separate tonemap): divide matter's own output by the SAME factor the
 * veil is lifted by, so one shared exposure control lands both correctly
 * instead of requiring two different renders bolted together. This is
 * authored for the DEFAULT exposure specifically - if the user pushes
 * exposure higher/lower via `[`/`]`, matter and veil both move together
 * proportionally from their respective correctly-authored baselines, which
 * is exactly what "a single exposure control" is supposed to mean.
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
 * Ocean specular: a broad, restrained Blinn-Phong sun-glint lobe
 * (`specularLobeNode`) brightened toward grazing angles by Schlick's
 * Fresnel approximation (`schlickFresnelNode`) — both are lighting terms,
 * not a texture. Masked to ocean fragments only (`1 - landFraction`) and
 * tied to the WATER dissolve front, so the glint fades with the same layer
 * that governs ocean discard/bathymetry. Kept soft on purpose — "a
 * blown-out sun glint would fight the veil" (brief).
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
  /** `SweepState.bathymetry` — the SOLE driver of ocean displacement AND colour (see module doc's OCEAN COLLAPSE section); deliberately not combined with the water dissolve front any more. */
  bathymetry: THREE.UniformNode<'float', number>
  /** `math/descent.ts`'s `exaggerationAt(altM)` — land only; marine relief is never exaggerated (spec 7.5). */
  exaggeration: THREE.UniformNode<'float', number>
  elevationTexture: THREE.CubeTexture
  /**
   * `.a` (land fraction) AND `.r` (log-density, this app's vegetation
   * proxy — see `field/terrain.ts`'s "Vegetation as colour" section) are
   * both read here now; `.r` is the exact same tau-encoded channel
   * `regimes/Shell.ts` turns into the veil's optical depth, sampled
   * un-exponentiated (`sampleLogTauNode`).
   */
  densityTexture: THREE.CubeTexture
  /** GHSL built-up fraction (`.r`) — drives the city-lights glow. Same cube `impl-pipeline` bakes for the veil's (separate, not-yet-wired) urban suppression — see module doc. */
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
    // oceanProgress: see module doc's OCEAN COLLAPSE section - this is the
    // SOLE driver of both the seafloor's displacement (positionNode, below)
    // and its colour, deliberately not combined with waterAlpha.
    const oceanProgress = clamp(opts.bathymetry, 0, 1)
    const baseColor = hypsometricColorNode(elevation, landFraction, latDeg, vegIndex, oceanProgress)
    // landFraction gates the exaggeration applied to the SHADING NORMAL's
    // slope, exactly as terrainDisplacementNode already gates displacement -
    // marine relief must not be exaggerated anywhere (spec 7.5), and this
    // was the one place that slipped through (see terrainNormalNode's doc).
    const coarseNormal = terrainNormalNode(opts.elevationTexture, dir, opts.exaggeration, landFraction, opts.texelAngularWidthRad)

    // Fine-scale detail, normal-only (see DETAIL_* constants' doc) - camera
    // sits pinned at the render origin under floating-origin translation
    // (see viewDir's own comment below), so positionWorld's own length IS
    // the camera-to-fragment distance, no extra uniform needed.
    const viewDistance = length(positionWorld)
    const detailStrength = detailFadeByDistanceNode(viewDistance, DETAIL_FULL_STRENGTH_M, DETAIL_FADE_OUT_M)
      .mul(DETAIL_MAX_STRENGTH)
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
    // brightened toward grazing angles, masked to ocean fragments and tied
    // to the water front (see OCEAN_SPECULAR_* constants' doc above).
    // Carries the same sunlight tint, so a glint caught near the terminator
    // is the orange one you actually see from orbit.
    const halfDir = viewDir.add(sunDir).normalize()
    const ndotH = max(dot(dir, halfDir), 0.0)
    const specIntensity = oceanSpecularIntensityNode(
      ndotH, cosView, OCEAN_SPECULAR_SHININESS, OCEAN_SPECULAR_F0, OCEAN_SPECULAR_STRENGTH,
    )
    const oceanMask = float(1).sub(clamp(landFraction, 0, 1))
    const specular = vec3(OCEAN_SPECULAR_COLOR.r, OCEAN_SPECULAR_COLOR.g, OCEAN_SPECULAR_COLOR.b)
      .mul(specIntensity).mul(sunTransmittance).mul(oceanMask).mul(waterAlpha)

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
    // Driven by bathymetry ALONE - see module doc's OCEAN COLLAPSE section
    // for why this must NOT be coupled to waterAlpha (it used to be
    // clamp(max(1-waterAlpha, opts.bathymetry), 0, 1), which made bathymetry
    // mathematically dead and confounded displacement with the discard).
    const oceanProgress = clamp(opts.bathymetry, 0, 1)
    const displacement = terrainDisplacementNode(elevation, landFraction, opts.exaggeration, oceanProgress)
    return positionGeometry.add(dir.mul(displacement))
  })()

  return material
}
