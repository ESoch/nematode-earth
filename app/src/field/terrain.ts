import * as THREE from 'three/webgpu'
import type { Vec3 } from '../math/raySpheroid'

/**
 * Pure CPU math for the terrain-shaded "matter" Earth (`sweep/Matter.ts`):
 * decoding the ETOPO elevation atlas, the hypsometric/bathymetric colour
 * ramps, land/ocean blending, and the heightfield-normal algebra relief
 * shading needs. `sweep/terrainNodes.ts` mirrors the algebra in TSL for the
 * GPU, the same CPU-source-of-truth / GPU-mirror split as `Dissolve.ts` /
 * `dissolveNodes.ts`.
 *
 * Colour language: near-monochrome bioluminescence is the veil's (spec
 * 6.2), so the matter Earth is deliberately DESATURATED — "a printed relief
 * map left in the sun," not a satellite composite — so the colour draining
 * away during the sweep reads as part of the reveal rather than two
 * unrelated pieces spliced together.
 */

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t

/** `cube_meta.json`'s R channel: signed elevation in metres, raw (no log encoding) — a bare half-float decode. */
export function decodeElevation(halfBits: number): number {
  return THREE.DataUtils.fromHalfFloat(halfBits)
}

export type Color3 = readonly [number, number, number]

/**
 * Ordered `[elevationM, r, g, b]` colour stops, linearly interpolated
 * between the pair straddling a query elevation and clamped at the ends.
 * All colours desaturated on purpose (see module doc).
 */
export type Stop = readonly [number, number, number, number]

/**
 * TWO land elevation ramps, not one — `LAND_STOPS_ARID` (barren/sparse) and
 * `LAND_STOPS_LUSH` (productive), blended by `vegetationIndex` in
 * `landColor` below. Elevation alone used to drive land colour end-to-end
 * (the single old `LAND_STOPS`, now retired); that made deserts read
 * exactly as green as cropland at the same elevation, which is wrong — the
 * Sahara is tan because it is barren, not because of its elevation.
 *
 * Both ramps share the SAME breakpoints and CONVERGE to identical colour
 * from 2800m up (highlands/peaks/summits: rock, then snow/ice) — regional
 * productivity doesn't grow grass above the treeline, so vegetation has
 * zero effect there by construction, and elevation alone still drives
 * light-to-dark and the snowline exactly as before (`iceElevationLoweringM`,
 * unchanged). Only the LOW/MID bands (lowlands through uplands, where real
 * biome variation actually lives) differ between the two ramps. Exported so
 * `terrainNodes.ts` builds its TSL ramps from the exact same stops.
 */
export const LAND_STOPS_ARID: readonly Stop[] = [
  [0, 0.48, 0.38, 0.22], // desert floor: warm sandy tan
  [400, 0.52, 0.39, 0.20], // dry plains: ochre
  [1200, 0.46, 0.35, 0.24], // dry uplands/badlands: warm brown
  [2800, 0.40, 0.33, 0.29], // highlands: converges with LUSH
  [4500, 0.50, 0.48, 0.47], // peaks: converges (grey)
  [7000, 0.86, 0.86, 0.88], // summits: converges (near-white)
]

export const LAND_STOPS_LUSH: readonly Stop[] = [
  [0, 0.20, 0.32, 0.19], // lush lowlands: cool deep green
  [400, 0.23, 0.36, 0.21], // productive plains: bright green
  [1200, 0.28, 0.33, 0.23], // temperate uplands/treeline: green fading to brown
  [2800, 0.40, 0.33, 0.29], // highlands: converges with ARID
  [4500, 0.50, 0.48, 0.47], // peaks: converges (grey)
  [7000, 0.86, 0.86, 0.88], // summits: converges (near-white)
]

// Ascending elevation order (deepest first) — rampColor assumes stops[0]
// is the LOWEST elevation and clamps below/above the two ends.
//
// Two extra stops versus the original ramp (-1000 slope, -20 shallow
// shelf) plus more contrast between them: continental shelves are real,
// free structure in the SAME bathymetry already driving this ramp (mean
// ocean depth ~3700m, shelf break ~-200m, most shelf area shallower than
// -50m), and the old six-stop ramp compressed all of that into one nearly
// flat "shelf" band. Still desaturated/dark by design — this is contrast
// between dark tones, not new saturation.
export const OCEAN_STOPS: readonly Stop[] = [
  [-10500, 0.004, 0.010, 0.018], // hadal
  [-6000, 0.010, 0.020, 0.035], // trench
  [-4000, 0.018, 0.035, 0.055], // abyssal plain
  [-1000, 0.035, 0.065, 0.09], // continental slope
  [-200, 0.065, 0.11, 0.15], // shelf break
  [-20, 0.10, 0.16, 0.20], // shallow continental shelf
  [0, 0.13, 0.19, 0.23], // shoreline: cool dark teal-grey
]

function rampColor(stops: readonly Stop[], elevationM: number): Color3 {
  const first = stops[0]
  const last = stops[stops.length - 1]
  if (!first || !last) throw new Error('rampColor: stops must be non-empty')
  if (elevationM <= first[0]) return [first[1], first[2], first[3]]
  if (elevationM >= last[0]) return [last[1], last[2], last[3]]
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i]
    const b = stops[i + 1]
    if (!a || !b) continue
    if (elevationM >= a[0] && elevationM <= b[0]) {
      const t = (elevationM - a[0]) / (b[0] - a[0])
      return [lerp(a[1], b[1], t), lerp(a[2], b[2], t), lerp(a[3], b[3], t)]
    }
  }
  return [last[1], last[2], last[3]] // unreachable given the two bound checks above
}

/**
 * Hypsometric tint for land: elevation (metres) still drives light-to-dark
 * and the snowline exactly as before, but the hue now also depends on
 * `vegetationIndexValue` (0 = barren/arid, 1 = productive/lush) — see
 * `vegetationIndex` below for where that comes from. Defaults to 0.5 (an
 * unbiased midpoint) so every pre-existing 1-arg call site is unaffected.
 */
export function landColor(elevationM: number, vegetationIndexValue = 0.5): Color3 {
  const arid = rampColor(LAND_STOPS_ARID, elevationM)
  const lush = rampColor(LAND_STOPS_LUSH, elevationM)
  const t = clamp01(vegetationIndexValue)
  return [lerp(arid[0], lush[0], t), lerp(arid[1], lush[1], t), lerp(arid[2], lush[2], t)]
}

/** Depth tint for ocean: dark and cool, shaded by depth so basins/ridges read faintly. */
export function oceanColor(elevationM: number): Color3 {
  return rampColor(OCEAN_STOPS, elevationM)
}

/**
 * Land/ocean colour blend, weighted CONTINUOUSLY by land fraction rather
 * than a hard threshold at elevation == 0 — the brief explicitly: the
 * Caspian, the Dead Sea and Death Valley are all below datum and are not
 * ocean, and thresholding elevation would misclassify them. A coastal texel
 * with partial land coverage (e.g. `loadCube.ts`'s Netherlands probe, land
 * fraction 0.54) blends smoothly instead of picking one side.
 *
 * `latDeg` (signed, default 0) feeds `iceElevationLoweringM` on the LAND
 * side (ice sheets — see that function's doc) AND `highLatitudeCoolness`
 * (a smaller, independent cool cast for sub-polar land generally, not just
 * where it's white with ice). `vegetationIndexValue` (default 0.5) feeds
 * `landColor`'s green-to-tan axis. Both default so every pre-existing
 * 2-arg call site is unaffected.
 *
 * `oceanProgress` (default 1, i.e. full depth — every pre-existing call
 * site unaffected) gates ONLY the ocean branch's elevation input, exactly
 * mirroring how `terrainDisplacementM` gates ocean displacement: at 0 the
 * ocean colour reads as the SHORELINE stop (elevation clamped toward 0,
 * an ordinary-looking sea), and it runs down `OCEAN_STOPS` toward the true
 * depth's stop as `oceanProgress` rises to 1. This is what makes the
 * collapse beat's drop actually VISIBLE — the geometric displacement alone
 * was measured at well under one screen pixel at the piece's opening
 * altitude (six kilometres of bathymetry against several kilometres per
 * device pixel from orbit), so colour is the real carrier of the beat, not
 * geometry. LAND colour is untouched by this parameter; only the ocean
 * branch's elevation is gated.
 */
export function hypsometricColor(
  elevationM: number,
  landFraction: number,
  latDeg = 0,
  vegetationIndexValue = 0.5,
  oceanProgress = 1,
): Color3 {
  const biome = landColor(elevationM + iceElevationLoweringM(latDeg), vegetationIndexValue)
  const cool = highLatitudeCoolness(latDeg)
  const land: Color3 = [
    lerp(biome[0], HIGH_LAT_COOL_COLOR[0], cool),
    lerp(biome[1], HIGH_LAT_COOL_COLOR[1], cool),
    lerp(biome[2], HIGH_LAT_COOL_COLOR[2], cool),
  ]
  const ocean = oceanColor(elevationM * clamp01(oceanProgress))
  const t = clamp01(landFraction)
  return [lerp(ocean[0], land[0], t), lerp(ocean[1], land[1], t), lerp(ocean[2], land[2], t)]
}

// --- Vegetation as colour (density is a real productivity proxy) ----------
//
// `cube_plain_128`'s R channel (soil nematode density, decoded here as
// `logTau` — see `field/loadCube.ts`'s `densityFaceToTauFace`, the exact
// same quantity `regimes/Shell.ts` turns into the veil's optical depth)
// tracks soil organic carbon and productivity: it is a genuine, measured
// vegetation proxy, not an invented one. High density means boreal forest
// or temperate cropland; low means desert. Driving the land's green-to-tan
// axis from it (`landColor` above) is thematically exact, not just
// visually convenient: THE SAME MEASUREMENT that colours the living world
// here is the one that becomes the veil once matter dissolves away. When
// the rock and water discard and only nematode density is left glowing,
// what is left IS the thing that was colouring the land all along, seen
// directly instead of through the proxy of green vs. tan. Keep this
// function even if it looks like a trivial smoothstep wrapper worth
// inlining — it is the one place that connects the two halves of the
// piece, and losing that connection during a later cleanup would be a real
// loss, not a simplification.

/** Below this logTau, land reads fully arid (`vegetationIndex` returns 0) — calibrated from the real extraction's land percentiles (`pipeline-report.md`: land density_per_sqm spans roughly p05=8.6e5 to p95=1.0e7 individuals/m²; converted to this atlas's logTau encoding via `+ln(E_A)`), a bit below the measured p05 so only the genuinely sparsest land clips to pure arid. */
export const VEGETATION_ARID_LOG_TAU = -5.5
/** Above this logTau, land reads fully lush — a bit above the measured p95 (see above), so only the genuinely densest land clips to pure lush. */
export const VEGETATION_LUSH_LOG_TAU = -2.0

/**
 * Continuous 0 (arid) .. 1 (lush) vegetation index from `logTau` (the
 * density cube's raw sampled value — see module doc). A smoothstep, not a
 * threshold: real biome transitions (savanna into forest, steppe into
 * cropland) are gradients, and a hard cutoff would read as a data overlay
 * rather than terrain that happens to be greener where life is denser.
 */
export function vegetationIndex(
  logTau: number,
  aridLogTau = VEGETATION_ARID_LOG_TAU,
  lushLogTau = VEGETATION_LUSH_LOG_TAU,
): number {
  return smoothstep01(aridLogTau, lushLogTau, logTau)
}

// --- High-latitude coolness (beyond just the ice) --------------------------
//
// The hard ice band (`iceElevationLoweringM`) makes the POLES read white.
// Real land also reads cooler and greyer through the sub-polar belt BEFORE
// it's white — taiga and tundra, not just ice sheets. This is a small,
// independent desaturating cool cast on top of whatever `landColor`
// already produced, continuous in latitude (no hard threshold), and
// capped well below full strength so it never fights the ice band itself
// or reads as a second colour ramp.

/** Exported (with the three constants below) so `terrainNodes.ts` mirrors the exact same numbers. */
export const HIGH_LAT_COOL_COLOR: Color3 = [0.30, 0.34, 0.36]
export const HIGH_LAT_COOL_START_DEG = 40
export const HIGH_LAT_COOL_END_DEG = 62
export const HIGH_LAT_COOL_MAX = 0.3

/** 0 through the tropics/temperate zone, ramping smoothly to `HIGH_LAT_COOL_MAX` across the sub-polar belt — deliberately narrower/earlier than `iceElevationLoweringM`'s belt (55-78°) so the cool cast is already established by the time ice starts taking over. */
export function highLatitudeCoolness(latDeg: number): number {
  return smoothstep01(HIGH_LAT_COOL_START_DEG, HIGH_LAT_COOL_END_DEG, Math.abs(latDeg)) * HIGH_LAT_COOL_MAX
}

// --- Polar ice ------------------------------------------------------------
//
// LAND_STOPS' own grey/white band (4500-7000m) already exists for mountain
// snow. Ice sheets are the SAME physical phenomenon at a latitude-lowered
// elevation — the permanent snowline sits roughly flat through the tropics
// and mid-latitudes, then drops toward sea level approaching the poles, a
// real climatological shape — so this is a continuous elevation SHIFT
// feeding the existing ramp, not a second colour or a hard latitude cutoff.
// Greenland and Antarctica read white because the effective snowline
// reaches sea level there, not because of a separate rule ("use |latitude|
// and blend, do not hard-threshold" — brief).

/** Sub-polar belt (degrees from the equator) across which the effective snowline drops from untouched (tropics/mid-latitudes) to fully sea-level (near the poles). Exported so `terrainNodes.ts` mirrors the exact same numbers rather than a second hand-copied literal. */
export const ICE_LAT_START_DEG = 55
export const ICE_LAT_END_DEG = 78
/** Metres the snowline drops by `ICE_LAT_END_DEG` — enough to pull LAND_STOPS' 4500-7000m grey/white band all the way down to sea level at the poles. */
export const ICE_MAX_LOWERING_M = 7000

/**
 * How far (metres) the effective snowline has dropped at a given latitude:
 * 0 through the tropics/mid-latitudes, ramping smoothly to `ICE_MAX_LOWERING_M`
 * across the sub-polar belt — continuous in latitude, no hard threshold.
 * Hemisphere-symmetric (`|latDeg|` only).
 */
export function iceElevationLoweringM(latDeg: number): number {
  return smoothstep01(ICE_LAT_START_DEG, ICE_LAT_END_DEG, Math.abs(latDeg)) * ICE_MAX_LOWERING_M
}

/**
 * Vertex displacement along the surface normal, in metres.
 *
 * LAND is always shown at its full exaggerated relief (`k * elevationM`) —
 * it is part of "a recognisable planet" from frame zero, not something the
 * sweep animates into being.
 *
 * OCEAN is displaced by `elevationM * bathymetryProgress`: at
 * `bathymetryProgress = 0` the surface sits flat at datum (an ordinary-
 * looking sea, the cold open's ordinary Earth); at `1` it has dropped to
 * the true (unexaggerated — marine relief gets no `k`, spec 7.5's "treat
 * marine as its own profile class") seafloor depth, the collapse beat's
 * reveal. `Timeline.ts`'s `SweepState.bathymetry` ramps exactly across the
 * collapse beat's wall-clock window (9-13s), so driving displacement
 * directly from it makes the drop a real, data-driven animation timed to
 * that beat, not a separate hand-authored one.
 */
export function terrainDisplacementM(
  elevationM: number,
  landFraction: number,
  exaggeration: number,
  bathymetryProgress: number,
): number {
  const land = elevationM * exaggeration
  const ocean = elevationM * clamp01(bathymetryProgress)
  return lerp(ocean, land, clamp01(landFraction))
}

/**
 * Exaggeration factor to apply to a fragment's SHADING-NORMAL slope, gated
 * by land fraction the same way `terrainDisplacementM` gates displacement:
 * exactly `exaggeration` on pure land, exactly 1 (unexaggerated) on pure
 * ocean, blended continuously in between.
 *
 * Marine relief must not be exaggerated ANYWHERE — not just in
 * displacement (spec 7.5) — and this is the fix for the case that slipped
 * through: `terrainNormalNode` used to multiply EVERY fragment's slope by
 * the land exaggeration factor unconditionally (no land/ocean gating at
 * all), so the seafloor's SHADING NORMAL — though never its actual
 * displaced position, `terrainDisplacementM`'s ocean term was already
 * correct — was lit as if bathymetry were exaggerated by `k`. Measured at
 * k=3.6 (`K_MAX`, `math/descent.ts`) essentially throughout the piece,
 * since the camera never descends below the ~192km altitude where
 * `exaggerationAt` starts backing off from its maximum.
 */
export function effectiveExaggeration(landFraction: number, exaggeration: number): number {
  return lerp(1, exaggeration, clamp01(landFraction))
}

/**
 * Central-difference SLOPE (dimensionless, dHeight/dLength) from a pair of
 * neighbouring heights straddling the sample point and their physical
 * separation. Positive `spacingM` is the distance between the two samples
 * (not to the centre), matching `(hPos - hNeg) / (2 * halfSpacing)`.
 */
export function centralDifferenceSlope(hNeg: number, hPos: number, spacingM: number): number {
  return (hPos - hNeg) / spacingM
}

function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2])
  if (len === 0) throw new Error('normalize3: zero-length vector')
  return [v[0] / len, v[1] / len, v[2] / len]
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/**
 * Tangent-space bump normal from two orthogonal slopes: tilts a `+Z`
 * (locally "up") normal away from steepest ascent, standard heightfield
 * relief-shading algebra. `dhdx`/`dhdy` already carry any exaggeration the
 * caller wants reflected in the shading (steeper `k` -> steeper apparent
 * slope), NOT applied again here.
 */
export function heightfieldNormal(dhdx: number, dhdy: number): Vec3 {
  return normalize3([-dhdx, -dhdy, 1])
}

/**
 * An arbitrary but consistent orthonormal basis perpendicular to unit
 * direction `n`, right-handed (`tangent x bitangent == n`) — the frame
 * `heightfieldNormal`'s tangent-space bump gets expressed in before it's
 * combined with the sphere's own geometric normal (`applyTangentSpaceBump`).
 */
export function tangentBasis(n: Vec3): { tangent: Vec3; bitangent: Vec3 } {
  const helper: Vec3 = Math.abs(n[2]) < 0.99 ? [0, 0, 1] : [1, 0, 0]
  const tangent = normalize3(cross3(helper, n))
  const bitangent = cross3(n, tangent) // unit: n, tangent already unit and orthogonal
  return { tangent, bitangent }
}

/**
 * Expresses a tangent-space bump `[bx, by, bz]` (`bz` along `base`) in
 * world space via the `(tangent, bitangent, base)` frame, returning a unit
 * vector — the standard TBN transform.
 */
export function applyTangentSpaceBump(base: Vec3, tangent: Vec3, bitangent: Vec3, bump: Vec3): Vec3 {
  const [bx, by, bz] = bump
  const v: Vec3 = [
    tangent[0] * bx + bitangent[0] * by + base[0] * bz,
    tangent[1] * bx + bitangent[1] * by + base[1] * bz,
    tangent[2] * bx + bitangent[2] * by + base[2] * bz,
  ]
  return normalize3(v)
}

// --- Atmospheric scattering (cheap analytic Rayleigh) ----------------------
//
// Not a physically complete multi-scattering integral — a shape-correct
// stand-in, per the brief: density falling off with altitude is stood in by
// the classic "airmass"/secant-law growth of optical path length toward
// grazing view angles (the same approximation atmospheric optics uses for a
// thin exponential shell), and Rayleigh's own 1/wavelength^4 scattering law
// is what makes blue scatter several times more than red — the limb's blue
// tint and its warming toward the terminator both fall out of that one law,
// not separate hand-picked colours.

/** Representative visible-spectrum wavelengths (nm) for the three colour channels — not exact sRGB primaries, chosen so red:blue lands near the commonly-cited ~5.5x Rayleigh ratio (see `rayleighWeight`'s test). */
export const RAYLEIGH_WAVELENGTHS_NM = { r: 680, g: 550, b: 440 } as const

/** Relative Rayleigh scattering strength ∝ 1/wavelength^4, normalized to green = 1 so the three channel weights compose directly into a colour with no separate brightness normalization step. */
export function rayleighWeight(wavelengthNm: number): number {
  return (RAYLEIGH_WAVELENGTHS_NM.g / wavelengthNm) ** 4
}

/** `[r, g, b]` scattering weights — `terrainNodes.ts`'s TSL mirror imports this directly rather than re-deriving it, the same shared-constant pattern as `LAND_STOPS`/`OCEAN_STOPS`. */
export const RAYLEIGH_RGB_WEIGHTS: Color3 = [
  rayleighWeight(RAYLEIGH_WAVELENGTHS_NM.r),
  rayleighWeight(RAYLEIGH_WAVELENGTHS_NM.g),
  rayleighWeight(RAYLEIGH_WAVELENGTHS_NM.b),
]

/**
 * Airmass-style relative optical path length through a thin exponential
 * atmosphere shell, from the cosine of the view ray's angle off the local
 * surface normal (1 = straight down/nadir, 0 = grazing/limb). The classic
 * secant approximation (`1/cosAngle`) diverges at the true limb, so it's
 * clamped to `maxPathLength` — a thin shell has a finite maximum path even
 * at a perfectly grazing look, and the clamp is what keeps the limb a
 * bright band rather than a singularity.
 */
export function atmosphericPathLength(cosViewAngle: number, maxPathLength = 10): number {
  const c = Math.max(cosViewAngle, 1 / maxPathLength)
  return Math.min(1 / c, maxPathLength)
}

/** Beer-Lambert saturating scattering response for one wavelength band: 0 with no path, approaching (never exceeding) 1 as the optical depth grows — the limb brightens but never blows out past white. */
export function rayleighScatterIntensity(pathLength: number, wavelengthWeight: number, density: number): number {
  return 1 - Math.exp(-pathLength * wavelengthWeight * density)
}

/** RGB atmosphere-glow colour from view-grazing-angle Rayleigh scattering — the limb effect itself, before any terminator warmth (`terminatorWarmthWeight`) is mixed in. */
export function atmosphereScatterColor(cosViewAngle: number, density: number, maxPathLength = 10): Color3 {
  const pathLength = atmosphericPathLength(cosViewAngle, maxPathLength)
  return [
    rayleighScatterIntensity(pathLength, RAYLEIGH_RGB_WEIGHTS[0], density),
    rayleighScatterIntensity(pathLength, RAYLEIGH_RGB_WEIGHTS[1], density),
    rayleighScatterIntensity(pathLength, RAYLEIGH_RGB_WEIGHTS[2], density),
  ]
}

/**
 * Warmth weight peaking at the terminator itself (`ndotSun == 0`) — the
 * shared primitive behind both the atmosphere's blue-to-warm colour shift
 * at the limb AND the terrain's own soft, warming day/night boundary
 * (`Matter.ts`), so the two effects read as one continuous phenomenon
 * rather than two independently-tuned bands.
 *
 * A GAUSSIAN in `ndotSun`, not a wide linear/quadratic tent — real
 * terminator reddening is driven by the SUN's own path length through the
 * atmosphere, which (like `atmosphericPathLength`'s secant-law growth for
 * the view ray) only gets large in the last few degrees before the sun
 * crosses the horizon, not across a broad linear ramp. `sigma` sets how
 * many degrees wide the visible band is: at `ndotSun == sigma` the weight
 * has already dropped to `1/e` (~37%), and it is negligible within about
 * 2.5 sigma. Keep `sigma` small (hundredths, in `ndotSun` units — a few
 * degrees) so the band stays a thin warm line, not a wash across a third
 * of the globe.
 */
export function terminatorWarmthWeight(ndotSun: number, sigma: number): number {
  return Math.exp(-((ndotSun / sigma) ** 2))
}

/**
 * How much of the SUN's own light survives its path to a surface point near
 * the terminator, per colour channel — a Beer-Lambert extinction on the
 * same `RAYLEIGH_RGB_WEIGHTS` that `atmosphereScatterColor` scatters with.
 * Multiply it INTO a scatter colour; never mix toward it.
 *
 * This is the honest mechanism behind a red sunset, and using it rather
 * than a hand-picked amber matters for a reason that is structural, not
 * cosmetic. The first version of this effect was
 * `mix(scatterColor, TERMINATOR_GLOW_COLOR, warmth)`, and at the terminator
 * (`warmth == 1`) a mix returns the constant and discards `scatterColor`
 * ENTIRELY — including its dependence on view angle, which is the entire
 * physical basis of a limb glow. The result on screen was a hard-edged
 * orange stripe of uniform brightness ruled straight across the disc,
 * identical at the centre and at the limb and about three times brighter
 * than the terrain it crossed. A multiplied transmittance cannot do that
 * at any parameter value: it is bounded in `(0, 1]`, so it only ever
 * REMOVES light, and whatever view-angle shape the scatter had survives it.
 *
 * `extraPathAtTerminator` is how many extra units of optical path the
 * sunlight traverses at the terminator itself, in the same units as
 * `atmosphericPathLength`. It is large (order 10) because sunlight at the
 * terminator is arriving nearly tangentially — the same secant-law growth
 * the view ray gets, applied to the other leg of the light's journey.
 */
export function terminatorSunTransmittance(
  warmth: number,
  extraPathAtTerminator: number,
  density: number,
): Color3 {
  const path = clamp01(warmth) * extraPathAtTerminator
  return [
    Math.exp(-path * RAYLEIGH_RGB_WEIGHTS[0]! * density),
    Math.exp(-path * RAYLEIGH_RGB_WEIGHTS[1]! * density),
    Math.exp(-path * RAYLEIGH_RGB_WEIGHTS[2]! * density),
  ]
}

/**
 * Sine of the sun's depression angle at the end of astronomical twilight —
 * the point past which the sky contributes no useful light. The real
 * definition is 18° below the horizon, and `ndotSun` at the surface is
 * exactly the sine of the sun's altitude, so the bound converts directly.
 */
export const ASTRONOMICAL_TWILIGHT_NDOTSUN = -Math.sin((18 * Math.PI) / 180)

/**
 * Scale factor for AMBIENT across the day/night boundary, 1 on the day side
 * falling to `nightFloor` by the end of astronomical twilight.
 *
 * Ambient in this shading model stands for sky and bounce light, and both
 * of those ARE sunlight — they have simply scattered once more before
 * arriving. Applying the full value on the unlit hemisphere lit the night
 * side to a readable dim daytime, so the terrain competed with the very
 * city lights it was meant to be a backdrop for.
 *
 * The ramp is deliberately much wider than the terminator effects around it
 * (`terminatorWarmthWeight`'s few degrees): direct sunlight cuts off at the
 * horizon, but the sky stays lit for a good while after, which is what
 * twilight IS. `nightFloor` is small and non-zero — airglow, starlight and
 * moonlight are real, and a hard zero would silhouette the planet against
 * space instead of letting its shape stay just legible.
 */
export function twilightAmbientScale(ndotSun: number, nightFloor: number): number {
  return lerp(nightFloor, 1, smoothstep01(ASTRONOMICAL_TWILIGHT_NDOTSUN, 0, ndotSun))
}

/** Smooth day-side/night-side sun availability — how much direct light there is to scatter at a given point on the surface. Full on the day side, ramping down to `nightFloor` on the night side across the same tight `width` as `terminatorWarmthWeight`'s `sigma` — real Earth has NO Rayleigh scattering on the unlit hemisphere (nothing there for it to scatter), so `nightFloor` should normally be 0, not a glow-preserving fudge; it exists only in case a caller wants a deliberate stylistic exception. */
export function atmosphereSunAvailability(ndotSun: number, width: number, nightFloor = 0): number {
  const t = clamp01((ndotSun + width) / (2 * width))
  return lerp(nightFloor, 1, t)
}

// --- Ocean specular ---------------------------------------------------------
//
// Water is the shiniest surface on the planet; a broad Blinn-Phong sun-glint
// lobe plus Schlick's grazing-angle Fresnel brightening are the two cheap
// terms that make it read as liquid rather than matte ground. Both are
// LIGHTING (a function of geometry and the sun/view directions), not a
// texture, so they stay inside "everything data-driven or defensible."

/** Schlick's approximation to Fresnel reflectance: near `f0` (water's own low ~2% normal-incidence reflectance) looking straight down, rising sharply toward 1 as the view grazes the surface (`cosViewAngle -> 0`) — the physical reason the ocean brightens toward the limb. */
export function schlickFresnel(cosViewAngle: number, f0: number): number {
  const c = clamp01(cosViewAngle)
  return f0 + (1 - f0) * (1 - c) ** 5
}

/** Blinn-Phong half-vector specular lobe: peaks at `ndotH == 1` (the surface normal exactly bisects sun and view), zero once the half-vector tilts past grazing. `shininess` narrows the lobe — kept low here so the result is a soft, broad sun-glint patch, not a laser-point highlight. */
export function specularLobe(ndotH: number, shininess: number): number {
  return Math.max(ndotH, 0) ** shininess
}

/** Combined restrained ocean sun-glint intensity: the specular lobe modulated by grazing-angle Fresnel brightening and an overall `strength` the caller tunes so it never fights the veil. */
export function oceanSpecularIntensity(
  ndotH: number,
  cosViewAngle: number,
  shininess: number,
  f0: number,
  strength: number,
): number {
  return specularLobe(ndotH, shininess) * schlickFresnel(cosViewAngle, f0) * strength
}

// --- Surface detail (normal-only, never displacement) ----------------------
//
// At 39.1 km/texel (soon 9.8) the baked relief is inherently smooth —
// individual hills and ridgelines below that scale simply aren't in the
// data. The fix here is a fine-scale procedural noise perturbation of the
// SHADING NORMAL ONLY (`terrainNodes.ts`'s `detailBumpedNormalNode`,
// mirroring this file's own `heightfieldNormal`/tangent-space-bump algebra
// but fed by noise instead of a texture tap). This is standard "detail
// normal mapping": it changes how light bounces off a surface, not the
// surface's actual position — `material.positionNode` (`sweep/Matter.ts`)
// is NEVER touched by it. That distinction matters here specifically:
// displacing the real geometry with fabricated height would be inventing
// terrain, which this project does not do (module doc); perturbing the
// LIGHTING normal with noise is a lighting trick, exactly like the
// Rayleigh scattering and ocean specular above, and stays inside
// "everything data-driven or defensible."
//
// The noise itself has no CPU pure-math equivalent worth mirroring (same
// as `Dissolve.ts`/`dissolveNodes.ts`'s deliberately-GPU-only MaterialX
// noise) — what IS pure and tested is the rule that makes it safe: fade it
// out with viewing distance so the high frequency never aliases/shimmers
// during descent.

/**
 * 1 close to the surface (full detail), fading smoothly to 0 by
 * `fadeOutM` — NOT altitude-proportional exaggeration (`math/descent.ts`'s
 * `exaggerationAt`, which runs the opposite direction for a different,
 * artistic reason), but real camera-to-fragment distance, so fine detail
 * is simply invisible until a viewer is close enough for it to matter and
 * never has a chance to alias at cruise altitude.
 */
export function detailFadeByDistance(viewDistanceM: number, fullStrengthM: number, fadeOutM: number): number {
  return 1 - smoothstep01(fullStrengthM, fadeOutM, viewDistanceM)
}

// --- City lights (GHSL built-up fraction) -----------------------------
//
// "Earth at night, exactly inverted" (spec 6.3's urban thesis) is the
// setup half of the piece's central visual rhyme: the same coastlines and
// river valleys that glow warm here become dark VOIDS in the veil once
// matter dissolves away and only nematode density remains, because paved
// ground holds less life, not more. Only the GLOW (this file) is built
// this session; the veil-side suppression (the "holes where they were"
// payoff) depends on `impl-pipeline`'s in-progress urban-suppression
// calibration and is deliberately left unwired here rather than shipped
// with a guessed exponent — see spec 6.3's own caveat against exactly that.

/** GHSL's built-up fraction: raw 0..1, no log encoding (unlike density/tau) — a bare half-float decode. */
export function decodeBuiltUpFraction(halfBits: number): number {
  return THREE.DataUtils.fromHalfFloat(halfBits)
}

function smoothstep01(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0))
  return t * t * (3 - 2 * t)
}

/**
 * Smooth day/night factor from N·L (dot of the BASE, unbumped surface
 * normal with the sun direction — deliberately not the relief-bumped
 * shading normal, which would flicker the terminator across every ridge
 * and gully): 0 on the day side, 1 on the night side, softened across
 * `terminatorWidth` so the transition is a band, not a hard edge.
 */
export function nightFactor(ndotSun: number, terminatorWidth = 0.08): number {
  return smoothstep01(-terminatorWidth, terminatorWidth, -ndotSun)
}

/**
 * City-light glow intensity from built-up fraction. GHSL's fraction is
 * "mostly near 0" even over real urban area (the sidecar's own words) —
 * a linear response would render everything but a few mega-city pixels
 * invisible, so a sub-linear (`gamma` < 1) curve lifts faint-but-real
 * texels into visibility without blowing out the rare dense ones.
 */
export function cityLightIntensity(builtUpFraction: number, gamma = 0.42): number {
  return clamp01(builtUpFraction) ** gamma
}

// --- Model uncertainty overlay (`u`) ---------------------------------------
//
// The sample overlay shows the locations of the 2020 field records. This
// overlay displays the model's variation among bootstrap predictions.
// Channel B of soil_plain_512_v2 is an area-weighted mean of the native
// pixel bootstrap coefficients of variation. It excludes pixels without
// source support. This describes the constituent pixels. Uncertainty in an
// aggregated count requires a separate estimate.
//
// HISTORICAL DISTRIBUTION, measured from the August 2026 `cube_plain_128`
// bake. These values informed the authored display ramp and coverage fade;
// they do not describe the full-local `soil_plain_512_v2` candidate:
//
//   land texels, A >= 0.99 (n=19885):
//     min 0.0916  p05 0.1216  p50 0.1847  p95 0.2917  p99 0.3709  max 0.5034
//   land texels, A > 0 (n=30920):
//     min 0.0000  p05 0.0331  p50 0.1688  p95 0.2827  max 0.5034
//   ocean texels, A == 0 (n=67384): B is 0 at all but 27 of them (max 0.1034)
//
//   CV by land-fraction band (p50): A<0.001 0.0000 | 0.001-0.1 0.0255 |
//     0.1-0.3 0.0793 | 0.3-0.5 0.1238 | 0.5-0.7 0.1502 | 0.9-1.0 0.1821
//
//   regional mean over land (A >= 0.9): Borneo 0.3413 | Gobi/Alxa 0.3323 |
//     Australia arid 0.2622 | Sahara 0.2592 | Arabian 0.2575 |
//     Amazon 0.2271 | Congo 0.2025 | W. Europe 0.1297 | US cornbelt 0.1284
//
// The legacy table motivated `uncertaintyCoverage`: an unsupported B=0
// must not appear maximally certain. Its low coastal CV values belonged to
// that older bake. The full-local candidate instead stores a support-
// normalized mean of native pixel CVs, NOT the CV of an aggregated count;
// it must not be described as CV diluted by ocean. The retained coverage
// fade is authored display opacity, not a numerical confidence weight.

/** Bottom of the ramp: just below the measured minimum over solid land (0.0916), so the most-certain real land sits at the ramp's floor without a hard clip. */
export const UNCERTAINTY_CV_LOW = 0.09
/** Top of the ramp: between the measured p95 (0.2917) and p99 (0.3709) over solid land. Deliberately NOT the measured max (0.5034) — anchoring there would spend half the visual range on the ~1% tail (a handful of texels in the Gobi and Borneo) and flatten the desert-versus-cropland contrast that is the whole point. The tail clips; that is the better trade. */
export const UNCERTAINTY_CV_HIGH = 0.35

/**
 * Single perceptually-ordered "ember" ramp in CV units, dark maroon through
 * red and orange to pale gold — magma-family, monotone in luminance, and no
 * rainbow (a rainbow would invent boundaries that are not in the data).
 *
 * WARM on purpose, across the whole range: the veil is tinted
 * `vec3(0.72, 0.86, 0.95)` (`regimes/Shell.ts`) and the piece runs dark at
 * low exposure, so a cool ramp — the obvious choice for a magma/viridis port
 * — would read as more veil rather than as an overlay on it. Every stop
 * keeps `r > b` for that reason, which is what `terrain.test.ts` pins.
 *
 * Brighter means LESS certain, not more: uncertainty is the thing that
 * should draw the eye, and the arid interiors that dominate the bright end
 * are exactly the places this dataset knows least about.
 *
 * Exported so `terrainNodes.ts` builds its TSL ramp from the very same
 * stops — the same shared-constant discipline as `LAND_STOPS_*`/`OCEAN_STOPS`,
 * so the CPU and GPU ramps cannot drift apart.
 */
export const UNCERTAINTY_STOPS: readonly Stop[] = [
  [UNCERTAINTY_CV_LOW, 0.16, 0.05, 0.09], // ~p05 and below: very dark maroon
  [0.155, 0.42, 0.10, 0.16], // deep red
  [0.22, 0.70, 0.22, 0.16], // brick
  [0.285, 0.91, 0.45, 0.18], // ~p95: orange
  [UNCERTAINTY_CV_HIGH, 0.99, 0.80, 0.48], // pale gold
]

/** The ember ramp evaluated at a bootstrap CV, clamped at both ends (see `UNCERTAINTY_STOPS`). */
export function uncertaintyColor(cv: number): Color3 {
  return rampColor(UNCERTAINTY_STOPS, cv)
}

/** Valid-model support at or below which the authored uncertainty overlay is hidden; this display threshold does not classify soil, ocean, or missing predictions. */
export const UNCERTAINTY_LAND_MIN = 0.15
/** Valid-model support at which the authored overlay reaches full opacity. The intervening fade does not alter CV or express numerical confidence; candidate CV is already support-normalized. */
export const UNCERTAINTY_LAND_FULL = 0.6

/**
 * Authored display opacity, 0 to 1, from channel A's valid-model support.
 * The historical `landFraction` name does not make A an ocean mask.
 *
 * Coverage comes from A, not B: an unsupported B=0 must not be painted as
 * low uncertainty. Supported predictions are model outputs, not direct
 * observations; zero support means unavailable, not measured absence.
 *
 * The smoothstep softens fractional coverage boundaries. It is neither a
 * numerical confidence estimate nor a correction to CV: the full-local
 * candidate's B is a support-normalized mean of native pixel CVs, not CV
 * of an aggregated count. This opacity fade is separate from density's
 * zero-only A gate, which must not dilute the all-cell density mean again.
 */
export function uncertaintyCoverage(landFraction: number): number {
  return smoothstep01(UNCERTAINTY_LAND_MIN, UNCERTAINTY_LAND_FULL, landFraction)
}

/**
 * The overlay composited over whatever the veil was already showing:
 * `base` untouched outside the model's domain, the ember ramp inside it,
 * faded continuously between by `uncertaintyCoverage`.
 *
 * REPLACING the veil's tint rather than modulating it (multiplying, or
 * mixing toward it) is deliberate. The veil's own colour carries DENSITY,
 * and CV is strongly ANTI-correlated with density in this dataset — the
 * arid interiors are both the sparsest and the least certain places on the
 * map (Sahara: 9.1e5/m² against Europe's 4.0e6/m², CV 0.259 against 0.130).
 * A multiplied overlay would therefore be dimmest exactly where it has the
 * most to say, and the Sahara — the single clearest thing this channel has
 * to show — would be the hardest part of it to see. Replacing the hue
 * decouples the two: the ramp reads at its own value everywhere, and
 * `regimes/Shell.ts` adds a luminance floor over covered land so the beat
 * survives the low-density regions too.
 *
 * At `strength` 0 this returns `base` EXACTLY (not approximately), which is
 * what lets the `u` toggle be a uniform rather than a material rebuild.
 */
export function uncertaintyOverlayColor(
  base: Color3,
  cv: number,
  landFraction: number,
  strength: number,
): Color3 {
  const w = clamp01(strength) * uncertaintyCoverage(landFraction)
  const ramp = uncertaintyColor(cv)
  return [lerp(base[0], ramp[0], w), lerp(base[1], ramp[1], w), lerp(base[2], ramp[2], w)]
}

/**
 * A coefficient of variation rendered for the HUD legend as the percentage
 * spread it literally is — CV is std/mean, so 0.35 means the bootstrap
 * distribution's standard deviation is 35% of its own mean. Without this the
 * ramp communicates "some places are more orange" and nothing else.
 */
export function formatUncertaintyCv(cv: number): string {
  return `${Math.round(cv * 100)}%`
}

// --- Output quantisation and dither ------------------------------------------
//
// A second, entirely separate defect from the cube reconstruction filter in
// `loadCube.ts`: the veil CONTOUR-BANDS. Its whole on-screen range spans
// only about a dozen 8-bit codes, and with nothing breaking up the
// quantisation those codes render as wide flat plateaus separated by hard
// edges — the "blocky" complaint has two independent causes and this is the
// other one. Fixing the filter alone leaves smooth-but-terraced; fixing the
// dither alone leaves faceted-but-smoothly-shaded.
//
// The functions below model the exact chain a shell fragment goes through on
// its way to the canvas — `renderer.toneMapping = ACESFilmicToneMapping` at
// `renderer.toneMappingExposure` (viewer.ts), then the sRGB transfer
// function, then 8-bit rounding — so the dither amplitude can be DERIVED
// against it and the banding measured before and after, rather than dialled
// in by eye. They are CPU-only instrumentation plus one shared constant;
// `terrainNodes.ts` mirrors only the two pieces that run per-fragment.

/** sRGB => AP1, the input half of three.js's ACES fit (`ToneMappingFunctions.js`). */
const ACES_INPUT_MAT: readonly (readonly [number, number, number])[] = [
  [0.59719, 0.35458, 0.04823],
  [0.07600, 0.90834, 0.01566],
  [0.02840, 0.13383, 0.83777],
]

/** AP1 => sRGB, the output half. */
const ACES_OUTPUT_MAT: readonly (readonly [number, number, number])[] = [
  [1.60475, -0.53108, -0.07367],
  [-0.10208, 1.10813, -0.00605],
  [-0.00327, -0.07276, 1.07602],
]

function mat3Apply(m: readonly (readonly [number, number, number])[], v: Color3): Color3 {
  const row = (i: number): number => {
    const r = m[i]!
    return r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
  }
  return [row(0), row(1), row(2)]
}

/** three.js's `RRTAndODTFit`, per channel. */
function rrtAndOdtFit(c: number): number {
  return (c * (c + 0.0245786) - 0.000090537) / (c * (c + 0.432951) * 0.983729 + 0.238081)
}

/**
 * three.js's `acesFilmicToneMapping` (`ToneMappingFunctions.js`), term for
 * term including the `/ 0.6` pre-scale and both matrices. Not the Narkowicz
 * one-liner that usually goes by "ACES" — this app renders through the
 * matrix fit, and the two differ enough at the veil's very low luminances
 * that a dither amplitude derived against the wrong one would be off.
 */
export function acesFilmicToneMap(color: Color3, exposure: number): Color3 {
  const scaled: Color3 = [color[0] * exposure / 0.6, color[1] * exposure / 0.6, color[2] * exposure / 0.6]
  const ap1 = mat3Apply(ACES_INPUT_MAT, scaled)
  const fitted: Color3 = [rrtAndOdtFit(ap1[0]), rrtAndOdtFit(ap1[1]), rrtAndOdtFit(ap1[2])]
  const out = mat3Apply(ACES_OUTPUT_MAT, fitted)
  return [clamp01(out[0]), clamp01(out[1]), clamp01(out[2])]
}

/** The sRGB transfer function (`renderer.outputColorSpace` is sRGB by default). */
export function srgbEncode(linear: number): number {
  const x = clamp01(linear)
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055
}

/**
 * The full shell-fragment-to-canvas chain: tone map, encode, round to 8 bits.
 * Returns integer codes 0..255, which is the space the banding lives in and
 * the only space the metrics below are meaningful in.
 */
export function displayCode8(color: Color3, exposure: number): readonly [number, number, number] {
  const mapped = acesFilmicToneMap(color, exposure)
  return [
    Math.round(srgbEncode(mapped[0]) * 255),
    Math.round(srgbEncode(mapped[1]) * 255),
    Math.round(srgbEncode(mapped[2]) * 255),
  ]
}

/**
 * Jorge Jimenez's interleaved gradient noise, on integer pixel coordinates.
 *
 * An ORDERED pattern rather than a hash of white noise: it decorrelates over
 * any small neighbourhood (which is what breaks a contour) while staying
 * spatially even, so it reads as fine grain instead of the clumpy salt and
 * pepper a white-noise dither leaves on a nearly-flat field. Three ALU and
 * no texture, which matters because it is evaluated once per fragment of the
 * hottest shader in the piece.
 */
export function interleavedGradientNoise(px: number, py: number): number {
  const t = 0.06711056 * px + 0.00583715 * py
  const s = 52.9829189 * (t - Math.floor(t))
  return s - Math.floor(s)
}

/**
 * Peak-to-peak dither amplitude, as a FRACTION of the fragment's own value.
 *
 * Relative, not absolute, for two independent reasons.
 *
 * 1. It has to vanish where the veil does. The shell writes premultiplied
 *    radiance over empty space, so an absolute offset would sprinkle noise
 *    across the whole sky (and, being signed, negative radiance at that).
 *    A relative one is exactly zero wherever the accumulated column is zero.
 * 2. It has to survive the exposure control. Exposure is a renderer property
 *    (viewer.ts, `[` and `]`, 1x to 64x) that the shell cannot see, and the
 *    absolute linear step corresponding to one 8-bit code moves by ~50x
 *    across that range (1.4e-3 at 1x down to 2.9e-5 at 64x, measured against
 *    `displayCode8`). The RELATIVE step is far flatter, because exposure
 *    enters the chain as a multiply ahead of a roughly power-law curve.
 *
 * 0.02 is where that lands: across the veil's brightness range it applies
 * between 0.24 and 1.47 codes peak-to-peak over the full 1x-64x exposure
 * range, and 0.67 to 1.47 at the default 8x — i.e. about the one code an
 * ordered dither wants, never several. `test/veilDither.test.ts` pins both
 * the swing and the banding it removes.
 */
export const VEIL_DITHER_AMPLITUDE = 0.02

/**
 * The scalar every channel of a shell fragment is multiplied by.
 *
 * ONE factor for rgb and alpha together, deliberately. The march's `col` and
 * its `alpha` are the same accumulated quantity (`col += T*(1-stepT)` and
 * `alpha = 1 - prod(stepT)` are algebraically identical), and the material
 * is `premultipliedAlpha`, so scaling the whole vec4 by a single number
 * dithers the veil's screen contribution while leaving the premultiplication
 * exact. Per-channel independent noise would instead dither the HUE, which
 * on a near-monochrome veil is the one thing that would actually look wrong.
 */
export function veilDitherScale(px: number, py: number, amplitude = VEIL_DITHER_AMPLITUDE): number {
  return 1 + amplitude * (interleavedGradientNoise(px, py) - 0.5)
}
