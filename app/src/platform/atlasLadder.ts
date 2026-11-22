/**
 * Which atlas tiers a given client is allowed to download.
 *
 * The full set is ~54 MB over the wire and the elevation 2048 tier is 31.7
 * MiB brotli of that — the single largest download in the piece. It was
 * measured MINIFIED at the opening framing: the disc spans 1,323 device px
 * for 12,742 km of planet, so 9.63 km per device pixel puts a 1024 texel at
 * 1.015 px (screen-matched) and a 2048 texel at 0.507 (half a pixel, i.e.
 * thrown away). Its benefit only appears below about 8,400 km of descent.
 *
 * So the cost/benefit is not "2048 never helps on a phone" — it does, deep in
 * the descent — it is that 31.7 MiB is not a defensible price for a phone to
 * pay up front for detail it cannot resolve until it is most of the way down.
 * Small clients get the ladder stopped at 1024.
 *
 * Three properties this file exists to hold:
 *
 *  1. The decision is a PURE function of its inputs (`planAtlasLadder`), so a
 *     test can pin it without a browser and `viewer.ts` can sample the
 *     environment exactly once. A device rotating to landscape must not
 *     trigger a 32 MB download mid-session, and the only way to guarantee
 *     that is for the answer not to depend on anything that can change.
 *  2. The signal is the viewport's SHORT side, in CSS pixels. The camera has
 *     a fixed vertical FOV, so the disc's span in device pixels is set by
 *     whichever viewport dimension is smaller — the same quantity, and the
 *     one that does not change when a phone is turned sideways. A width-only
 *     rule would hand the largest phones on sale (956 CSS px in landscape,
 *     wider than plenty of laptops) the tier this file exists to withhold.
 *     `devicePixelRatio` is deliberately NOT part of it: a 3x phone has more
 *     device pixels than a 1x laptop while being far less able to afford the
 *     bytes.
 *  3. `navigator.connection` is Chromium-only. It is used when present and
 *     never depended on — every absent or unrecognised value falls through
 *     to the viewport rule.
 */

/** One rung of a progressive atlas ladder. */
export interface AtlasTier {
  /** Directory the cube's faces and meta are fetched from. */
  readonly url: string
  /** Texels per cube face, which is also the rung's rank on the ladder. */
  readonly faceSize: number
}

/**
 * Elevation, coarsest first. 256 paints fast (~0.5 MiB), 1024 is the working
 * tier, 2048 is the top. The middle rung earns its 8.5 MiB precisely because
 * 2048 costs 31.7 MiB: without it the second refinement is a long wait on a
 * slow link.
 */
export const ELEVATION_TIERS: readonly AtlasTier[] = [
  { url: '/data/elevation_plain_256', faceSize: 256 },
  { url: '/data/elevation_plain_1024', faceSize: 1024 },
  { url: '/data/elevation_plain_2048', faceSize: 2048 },
]

/**
 * Urban (GHSL), coarsest first. Built-up fraction is mostly exact zeros, so
 * it compresses ~29x and the whole ladder costs ~9 MiB brotli against
 * elevation's ~41 MiB.
 */
export const URBAN_TIERS: readonly AtlasTier[] = [
  { url: '/data/urban_plain_512', faceSize: 512 },
  { url: '/data/urban_plain_1024', faceSize: 1024 },
  { url: '/data/urban_plain_2048', faceSize: 2048 },
]

/** Highest face size a capped client is served. */
export const CAPPED_MAX_FACE_SIZE = 1024

/**
 * Below this many CSS pixels on the viewport's SHORT side, a client is small.
 *
 * 900 is the same number `viewer.html`'s HUD breakpoint uses, so "small"
 * means one thing across the project. It puts every phone (390-440 short
 * side, either orientation) and every tablet (~820) on the capped ladder,
 * leaves a 1440x900 laptop on the full one, and catches genuinely small
 * browser windows on large screens — which really are small, and really are
 * deciding this once, at load.
 */
export const SMALL_CLIENT_SHORT_SIDE_PX = 900

/**
 * `effectiveType` values that mean "do not spend 31.7 MiB here". 4g is the
 * ceiling the API reports and covers everything faster, so it is absent.
 */
export const SLOW_EFFECTIVE_TYPES: readonly string[] = ['slow-2g', '2g', '3g']

/**
 * Everything about the client that may narrow the ladder, sampled once.
 *
 * The two connection fields are optional because they are Chromium-only —
 * `undefined` is the normal case on Safari and Firefox, not an error.
 */
export interface ClientBudgetHints {
  /** `window.innerWidth`, CSS pixels. */
  readonly viewportWidthPx: number
  /** `window.innerHeight`, CSS pixels. */
  readonly viewportHeightPx: number
  /** `navigator.connection?.effectiveType`. */
  readonly effectiveType?: string | null
  /** `navigator.connection?.saveData` — the visitor asking for less data. */
  readonly saveData?: boolean
}

/** Why the ladder was shortened, or `none`. Ranked strongest signal first. */
export type LadderCap = 'save-data' | 'slow-network' | 'small-viewport' | 'none'

export interface AtlasLadderPlan {
  readonly elevation: readonly AtlasTier[]
  readonly urban: readonly AtlasTier[]
  readonly cap: LadderCap
}

/** Drops rungs from the TOP. First paint must survive every cap. */
const upTo = (tiers: readonly AtlasTier[], maxFaceSize: number): readonly AtlasTier[] =>
  tiers.filter((t) => t.faceSize <= maxFaceSize)

/**
 * The ruling. Pure, total, and the only place the cap is decided.
 *
 * Elevation is capped by any of the three signals. Urban is capped only by
 * the two that mean "this visitor wants fewer bytes", never by screen size
 * alone: urban 2048 is 6.65 MiB — 21% of what elevation 2048 costs — and it
 * is the only atlas carrying the piece's SUBJECT. The finding is about
 * cities; a viewer who cannot see cities as cities has lost the thing being
 * tested, whereas a viewer without elevation 2048 has lost sub-pixel relief
 * they could not resolve at the opening framing anyway.
 *
 * A viewport that cannot be measured (0, NaN) counts as small. An unreadable
 * client must not buy the most expensive tier by default.
 */
export function planAtlasLadder(hints: ClientBudgetHints): AtlasLadderPlan {
  const shortSidePx = Math.min(hints.viewportWidthPx, hints.viewportHeightPx)
  const smallViewport = !(shortSidePx >= SMALL_CLIENT_SHORT_SIDE_PX)
  const slowNetwork =
    hints.effectiveType != null && SLOW_EFFECTIVE_TYPES.includes(hints.effectiveType)

  const cap: LadderCap =
    hints.saveData === true
      ? 'save-data'
      : slowNetwork
        ? 'slow-network'
        : smallViewport
          ? 'small-viewport'
          : 'none'

  const wantsFewerBytes = cap === 'save-data' || cap === 'slow-network'
  return {
    elevation: cap === 'none' ? ELEVATION_TIERS : upTo(ELEVATION_TIERS, CAPPED_MAX_FACE_SIZE),
    urban: wantsFewerBytes ? upTo(URBAN_TIERS, CAPPED_MAX_FACE_SIZE) : URBAN_TIERS,
    cap,
  }
}

/**
 * The one place the ambient environment is read. Called once by `viewer.ts`
 * at load and never again — see property 1 in this file's doc.
 *
 * `NetworkInformation` is not in the DOM lib TypeScript ships, because it is
 * not a cross-browser API; the shape is read defensively rather than declared,
 * so a browser that grows a different one cannot throw here.
 */
export function readClientBudgetHints(): ClientBudgetHints {
  const conn = (
    navigator as {
      connection?: { effectiveType?: unknown; saveData?: unknown }
    }
  ).connection
  return {
    viewportWidthPx: window.innerWidth,
    viewportHeightPx: window.innerHeight,
    effectiveType: typeof conn?.effectiveType === 'string' ? conn.effectiveType : null,
    saveData: conn?.saveData === true,
  }
}
