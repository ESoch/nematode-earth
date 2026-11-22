import { WGS84 } from '../math/wgs84'

/**
 * Estimated nematode count within the current view frustum (spec 8.3, "the
 * piece's emotional device" — every other UI element is said to serve it).
 * Watching this fall from ~1e19 at orbit to a handful at millimetre scale
 * is what turns 4.4e20 into a feeling rather than a statistic.
 *
 * This is deliberately an ORDER-OF-MAGNITUDE estimate, not an integral:
 * `count ~= visible_ground_area * representative_areal_density`. Two
 * approximations, both explicit rather than hidden:
 *
 * 1. VISIBLE AREA. This app's camera always looks at the planet's centre
 *    regardless of orbital position (`viewer.ts`'s `camera.lookAt`), so it
 *    is always effectively nadir over whatever point is currently closest —
 *    the flat-ground frustum-footprint approximation below is exact for
 *    that camera model, not a worst case. Two bounds, and the tighter one
 *    wins:
 *      - CAP bound: from orbit you can never see more than the horizon
 *        allows, a spherical cap of area `2*pi*R^2*(1 - R/(R+alt))` -
 *        exactly a hemisphere in the alt -> infinity limit, exactly zero
 *        at alt = 0.
 *      - FRUSTUM bound: near the surface, the camera's own field of view
 *        is what limits what's in frame, not the horizon - a flat-ground
 *        footprint `~ (2*alt*tan(fovV/2))^2 * aspect`.
 *    The cap bound dominates (is smaller) from orbit; the frustum bound
 *    dominates from low altitude. See `visibleGroundAreaM2`.
 *
 * 2. REPRESENTATIVE DENSITY. A single sampled point is representative of
 *    a SMALL visible patch (near the surface) but not of a whole visible
 *    hemisphere (from orbit, where the patch spans every biome on the
 *    lit side at once) - so the density term is smoothly blended from the
 *    LOCAL sample toward the GLOBAL MEAN as altitude climbs through a
 *    transition band, rather than switched abruptly. See
 *    `densityMixFactor` / `estimatedCountInView`.
 *
 * Nothing here claims precision beyond the exponent - the whole point,
 * per the brief, is "right in magnitude and smooth as you move," not
 * "correct to three figures."
 */

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)

function smoothstep01(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0))
  return t * t * (3 - 2 * t)
}

/**
 * Total baked areal integral (van den Hoogen et al. 2019's published
 * headline figure, spec section 1.1 / `pipeline/build/bake_cube.py`'s
 * `PUBLISHED_TOTAL`) and the exact total surface area the pipeline
 * integrated it over (`elevation_*`'s sidecar `source.stats.total_area_m2`,
 * land AND ocean, ETOPO 2022) — together giving the whole-globe mean areal
 * density, the value `estimatedCountInView` blends toward at orbital
 * altitude. NOT `math/optical.ts`'s `N_A_LAND`/`N_A_ICEFREE`, which are
 * land-only means over a much smaller area and would overstate a
 * hemisphere that is ~72% ocean.
 */
export const PUBLISHED_TOTAL_INDIVIDUALS = 4.4e20
export const TOTAL_SURFACE_AREA_M2 = 510064471909788.25
export const GLOBAL_MEAN_DENSITY_PER_SQM = PUBLISHED_TOTAL_INDIVIDUALS / TOTAL_SURFACE_AREA_M2

/** Spherical-cap ground area visible from directly overhead at `altitudeM`, radius `R`. Exactly a hemisphere as altitude -> infinity; exactly 0 at altitude 0. */
export function capAreaVisibleM2(altitudeM: number, R: number = WGS84.R_MEAN): number {
  const d = R + Math.max(0, altitudeM)
  return 2 * Math.PI * R * R * (1 - R / d)
}

/** Flat-ground footprint of a nadir camera's frustum at `altitudeM`, given its vertical FOV (radians) and aspect ratio (width/height). */
export function frustumFootprintAreaM2(altitudeM: number, fovVerticalRad: number, aspect: number): number {
  const halfHeight = Math.max(0, altitudeM) * Math.tan(fovVerticalRad / 2)
  const height = 2 * halfHeight
  const width = height * aspect
  return width * height
}

/** The tighter of the two bounds above — see module doc. */
export function visibleGroundAreaM2(
  altitudeM: number,
  fovVerticalRad: number,
  aspect: number,
  R: number = WGS84.R_MEAN,
): number {
  return Math.min(capAreaVisibleM2(altitudeM, R), frustumFootprintAreaM2(altitudeM, fovVerticalRad, aspect))
}

/** 0 (fully local) near the surface, 1 (fully global mean) from orbit, smoothly across [lowAltM, highAltM]. */
export function densityMixFactor(altitudeM: number, lowAltM = 10_000, highAltM = 2_000_000): number {
  return smoothstep01(lowAltM, highAltM, altitudeM)
}

/**
 * The readout's actual value: visible ground area times a density blended
 * from the local sample toward the global mean as altitude climbs. Neither
 * `localDensityPerSqm` nor `meanDensityPerSqm` is computed here — both are
 * plain numbers the caller already has (a CPU cube probe / `syntheticTau`
 * for the former, `GLOBAL_MEAN_DENSITY_PER_SQM` or a synthetic-field
 * equivalent for the latter), keeping this function free of any data-
 * source dependency.
 */
export function estimatedCountInView(
  altitudeM: number,
  localDensityPerSqm: number,
  meanDensityPerSqm: number,
  fovVerticalRad: number,
  aspect: number,
  R: number = WGS84.R_MEAN,
): number {
  const area = visibleGroundAreaM2(altitudeM, fovVerticalRad, aspect, R)
  const mix = densityMixFactor(altitudeM)
  const density = localDensityPerSqm * (1 - mix) + meanDensityPerSqm * mix
  return area * density
}

/**
 * Formats a count for legibility (spec: "`1.2 x 10^19` communicates;
 * `12,041,883,204,182,048,192` does not"). Below 1000, an ordinary integer
 * reads better than exponential notation ("3", not "3 x 10^0").
 */
export function formatCount(count: number): string {
  if (!Number.isFinite(count) || count <= 0) return '0'
  if (count < 1000) return count < 10 ? count.toFixed(1) : Math.round(count).toLocaleString()
  const exponent = Math.floor(Math.log10(count))
  const mantissa = count / 10 ** exponent
  return `${mantissa.toFixed(1)} × 10${toSuperscript(exponent)}`
}

const SUPERSCRIPT_DIGITS: Record<string, string> = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻',
}

function toSuperscript(n: number): string {
  return String(n).split('').map((c) => SUPERSCRIPT_DIGITS[c] ?? c).join('')
}

/** The published figure a comparison band is measured against. */
export interface ComparisonAnchor {
  /** The figure itself, in individuals. */
  readonly value: number
  /** The counted thing, as it appears in the band's own text — so a band cannot cite a paper about ants and then talk about sand. */
  readonly label: string
  /** Where `value` comes from, specific enough to look up: authors, year, journal, DOI. */
  readonly source: string
}

/** One band of the comparison ladder: everything from `atLeast` up to the next band's floor. */
export interface CountComparison {
  /** Inclusive lower bound. */
  readonly atLeast: number
  /** The band's copy. A function because the top band quantifies itself against the global total. */
  readonly text: (count: number) => string
  /** `null` for bands that describe the count without comparing it to anything (those may not say "more than"). */
  readonly anchor: ComparisonAnchor | null
}

/**
 * Human-scale comparisons for the count-in-view readout (spec: "consider
 * showing both" the exponent and something felt).
 *
 * REPLACES a ladder of folk statistics — "every grain of sand on every beach
 * on Earth", "every star in the Milky Way" — whose thresholds were not
 * merely imprecise but WRONG at their own floors: the band claiming to beat
 * every grain of sand began at 1e18 against a commonly cited ~7.5e18, and
 * the band claiming to beat every human alive began at 1e9 against ~8.2e9.
 * Both statements were false on screen across their whole lower range, and
 * the test guarding them asserted only that the string was non-empty.
 *
 * Every band below is now anchored to a published global abundance estimate
 * — deliberately the same GENRE of figure as the 4.4e20 this piece is built
 * on, which also makes the ladder legible as a ladder: ants, then cells,
 * then trees, then people, all of them counted the same way the nematodes
 * were. Each band's floor sits AT OR ABOVE its own published figure, so
 * "more than every tree on Earth" is true at the moment it first appears,
 * not merely true somewhere in the band. `test/countEstimate.test.ts`
 * enforces exactly that, and fails if a threshold slips under its anchor.
 *
 * The humans band is anchored to the PROJECTED PEAK (10.3 billion, mid-2080s)
 * rather than today's ~8.2 billion, so the claim does not quietly become
 * false as the population grows.
 */
export const COUNT_COMPARISONS: readonly CountComparison[] = [
  {
    // The most informative thing this readout can say, and the only one that
    // needs no outside comparison at all: what fraction of the subject is on
    // screen. At this piece's default orbital altitude that is about a third.
    // Below 1% a percentage stops being a gut-check ("0% of every nematode"
    // reads as nothing), so the concrete comparisons take over beneath it.
    atLeast: PUBLISHED_TOTAL_INDIVIDUALS * 0.01,
    text: (count) => `About ${((count / PUBLISHED_TOTAL_INDIVIDUALS) * 100).toFixed(0)}% of the global surface-soil nematode estimate`,
    anchor: {
      value: PUBLISHED_TOTAL_INDIVIDUALS * 0.01,
      label: 'nematode',
      source: 'van den Hoogen et al. 2019, Nature 572:194-198, doi:10.1038/s41586-019-1418-6 — 4.4 +/- 0.64 x 10^20 nematodes in surface soils',
    },
  },
  {
    atLeast: 2.0e16,
    text: () => 'At least as many as the estimated 20 quadrillion ants on Earth',
    anchor: {
      value: 2.0e16,
      label: 'ant',
      source: 'Schultheiss et al. 2022, PNAS 119, doi:10.1073/pnas.2201550119 — "almost 20 x 10^15 individuals"',
    },
  },
  {
    atLeast: 3.72e13,
    text: () => 'At least as many as the estimated 37.2 trillion cells in a reference human body',
    anchor: {
      value: 3.72e13,
      label: 'cells',
      source: 'Bianconi et al. 2013, Annals of Human Biology 40:463-471, doi:10.3109/03014460.2013.807878 — "a total number of 3.72 x 10^13"',
    },
  },
  {
    atLeast: 3.04e12,
    text: () => 'At least as many as the estimated 3.04 trillion trees on Earth',
    anchor: {
      value: 3.04e12,
      label: 'tree',
      source: 'Crowther et al. 2015, Nature 525:201-205, doi:10.1038/nature14967 — "approximately 3.04 trillion" trees',
    },
  },
  {
    // Anchored to the projected PEAK, not the current figure — see the module
    // note above on why a "more than every human alive" band pinned to today's
    // population is a claim with an expiry date.
    atLeast: 1.03e10,
    text: () => 'At least as many as the 10.3 billion humans in the UN’s 2024 peak population projection',
    anchor: {
      value: 1.03e10,
      label: 'human',
      source: 'UN DESA World Population Prospects 2024 (UN DESA/POP/2024/TR/NO. 9) — population projected to peak at approximately 10.3 billion in the mid-2080s',
    },
  },
  {
    // Not an outside comparison but the piece's own unit, derived from the
    // same published total two lines up: it tells the viewer what a square
    // metre of planet holds, at exactly the altitude where a square metre is
    // what they are looking at.
    atLeast: GLOBAL_MEAN_DENSITY_PER_SQM,
    text: () => 'At least the global surface-soil estimate averaged over one square metre of Earth',
    anchor: null,
  },
  { atLeast: 1e3, text: () => 'Thousands in the estimated footprint', anchor: null },
  { atLeast: 1, text: () => 'A local estimate from the model', anchor: null },
  { atLeast: 0, text: () => 'The estimated count is below one', anchor: null },
]

/** A band's copy at a given count (bands that ignore the count still take it, so callers need no special case). */
export function comparisonTextFor(band: CountComparison, count: number): string {
  return band.text(count)
}

/**
 * The band `count` falls in. Deliberately coarse — the point is a gut-check
 * on scale, not a second precise statistic — but every comparative claim it
 * makes is true at the threshold that first shows it. See COUNT_COMPARISONS.
 */
export function humanComparison(count: number): string {
  const band = COUNT_COMPARISONS.find((b) => count >= b.atLeast) ?? COUNT_COMPARISONS.at(-1)!
  return comparisonTextFor(band, count)
}
