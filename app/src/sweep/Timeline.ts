/**
 * The sweep is a PURE FUNCTION of one number.
 *
 * `stateAt(t)` has no memory, no velocity, no easing state carried between
 * frames. That is what makes it scrubbable rather than merely playable: a
 * matter slider and a playing cinematic both just write `t`, and neither
 * knows the other exists. `Timeline` is a thin cursor over it.
 *
 * NOTE WHAT IS NOT IN SweepState: no sigma, no tau, no scale height, no veil
 * opacity. The veil's optical parameters are not a function of the sweep
 * (spec section 6.1's governing rule) - the veil renders at full opacity
 * from frame zero and is OCCLUDED by matter, never faded. The only things
 * the sweep moves are matter's coverage and the camera/UI choreography
 * around it. `SWEEP_STATE_KEYS` is asserted against the live object in the
 * test suite specifically so this invariant cannot rot silently.
 */
export const BEATS = [
  { name: 'cold-open', t0: 0, t1: 3 },
  { name: 'erosion', t0: 3, t1: 9 },
  { name: 'collapse', t0: 9, t1: 13 },
  { name: 'what-remains', t0: 13, t1: 18 },
  { name: 'release', t0: 18, t1: 20 },
] as const

export type BeatName = (typeof BEATS)[number]['name']

export const T_END = 20
/** Matter starts leaving at the top of the erosion beat. */
export const SWEEP_T0 = 3
/** ...and is gone at the bottom of the collapse beat. */
export const SWEEP_T1 = 13

/** Cobb, N.A. (1914), Nematodes and Their Relationships. */
export const COBB_LINE =
  'if all the matter in the universe except the nematodes were swept away, ' +
  'our world would still be dimly recognizable'

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)

/** Ken Perlin's C2 quintic. C1 easing (plain smoothstep) leaves a visible kink where a front starts or stops. */
export function smootherstep01(x: number): number {
  const t = clamp01(x)
  return t * t * t * (t * (t * 6 - 15) + 10)
}

/** Normalized eased progress of a window [a,b] in u. */
export function ramp(u: number, a: number, b: number): number {
  return smootherstep01((u - a) / (b - a))
}

/**
 * Which beat `t` falls in, and normalized progress through it.
 * Boundaries resolve to the LATER beat (matches spec 6.1's half-open
 * `[t0,t1)` reading of the table); `t` outside `[0,T_END]` clamps into the
 * first/last beat rather than reporting nothing.
 */
export function beatAt(t: number): { beat: BeatName; beatProgress: number } {
  if (t >= T_END) return { beat: 'release', beatProgress: 1 }
  for (let i = BEATS.length - 1; i >= 0; i--) {
    const b = BEATS[i]!
    if (t >= b.t0 || i === 0) {
      return { beat: b.name, beatProgress: clamp01((t - b.t0) / (b.t1 - b.t0)) }
    }
  }
  return { beat: 'cold-open', beatProgress: 0 }
}

/**
 * Front windows expressed in sweep progress u = (t-SWEEP_T0)/(SWEEP_T1-SWEEP_T0),
 * which is what makes the matter slider exactly linear and exactly
 * invertible. atmosphere/water/rock reproduce the spec's stated wall-clock
 * order and overlap so the erosion reads as one propagating event rather
 * than three separate cuts.
 *
 * `bathymetry` is deliberately the FIRST thing to move, not the last (it
 * used to be `[0.6, 1.0]`, entirely after water and rock had both already
 * finished eroding network-wide - see `Matter.ts`'s OCEAN COLLAPSE section
 * doc for the full audit trail). The load-bearing constraint: the sea
 * floor's drop is only visible while the ocean surface fragments showing
 * it are still (fully) OPAQUE. `water`'s own front is exactly 0 - full
 * opacity everywhere, not asymptotically, see `Dissolve.ts`'s `layerAlpha`
 * doc for why the endpoint is exact - for every u at or below `water[0]`,
 * so bathymetry has to finish its own ramp there too. `[0.0, 0.16]` lands
 * exactly on that boundary: the sea visibly drains/deepens for the erosion
 * beat's first 1.6s (t=3-4.6s), fully opaque throughout, and only then does
 * the water layer itself start to visibly dissolve away.
 */
const WIN = {
  bathymetry: [0.0, 0.16],
  atmosphere: [0.0, 0.24],
  water: [0.16, 0.44],
  rock: [0.34, 0.6],
} as const

/** The rock front's last visible fragment, before the settled veil beat. */
export const LAST_VISIBLE_LAYER_T = SWEEP_T0 + WIN.rock[1] * (SWEEP_T1 - SWEEP_T0)

/** Normalized progress through the sweep span [SWEEP_T0, SWEEP_T1]. */
export function sweepProgress(t: number): number {
  return clamp01((t - SWEEP_T0) / (SWEEP_T1 - SWEEP_T0))
}

/** 1 = matter fully intact, 0 = matter gone. Linear in t so the slider maps uniformly. */
export function matterAt(t: number): number {
  return 1 - sweepProgress(t)
}

/** Exact inverse of matterAt on [SWEEP_T0, SWEEP_T1]; clamps outside. */
export function timeForMatter(m: number): number {
  return SWEEP_T0 + (1 - clamp01(m)) * (SWEEP_T1 - SWEEP_T0)
}

export interface SweepState {
  t: number
  beat: BeatName
  beatProgress: number
  /** 1 = intact planet, 0 = matter gone. Monotone non-increasing in t. */
  matter: number
  frontAtmosphere: number
  frontWater: number
  frontRock: number
  /**
   * 0 = ocean surface at datum, 1 = dropped to the true (unexaggerated)
   * seafloor depth. The SOLE driver of `Matter.ts`'s ocean vertex
   * displacement AND its colour (`OCEAN_STOPS` ramp) - deliberately not
   * combined with the water dissolve front (`frontWater`) any more; see
   * this file's `WIN.bathymetry` doc and `Matter.ts`'s OCEAN COLLAPSE
   * section for why coupling the two made the drop invisible.
   */
  bathymetry: number
  cameraAltitudeM: number
  cameraSpinRadPerSec: number
  titleChars: number
  uiOpacity: number
}

export const SWEEP_STATE_KEYS = [
  't', 'beat', 'beatProgress', 'matter',
  'frontAtmosphere', 'frontWater', 'frontRock', 'bathymetry',
  'cameraAltitudeM', 'cameraSpinRadPerSec', 'titleChars', 'uiOpacity',
] as const satisfies readonly (keyof SweepState)[]

const ALT_START = 1.25e7
const ALT_END = 1.05e7
const SPIN_BASE = 0.012

export function stateAt(t: number): SweepState {
  const tc = Math.min(T_END, Math.max(0, t))
  const u = sweepProgress(tc)
  const { beat, beatProgress } = beatAt(tc)
  return {
    t: tc,
    beat,
    beatProgress,
    matter: 1 - u,
    frontAtmosphere: ramp(u, WIN.atmosphere[0], WIN.atmosphere[1]),
    frontWater: ramp(u, WIN.water[0], WIN.water[1]),
    frontRock: ramp(u, WIN.rock[0], WIN.rock[1]),
    bathymetry: ramp(u, WIN.bathymetry[0], WIN.bathymetry[1]),
    cameraAltitudeM: ALT_START + (ALT_END - ALT_START) * smootherstep01(tc / 18),
    cameraSpinRadPerSec: SPIN_BASE * (1 - 0.75 * ramp(tc, 18, 20)),
    titleChars: Math.floor(COBB_LINE.length * clamp01((tc - 0.3) / 2.5)),
    uiOpacity: ramp(tc, 18, 19.5),
  }
}

/**
 * Upper bound on the `dtSec` a caller should feed into `Timeline.advance`.
 * Backgrounding the tab mid-play - or the material rebuild when the 48 MiB
 * elevation tier lands mid-sweep, a real stall on a cold load - hands the
 * render loop a multi-second (or larger) wall-clock delta on the next
 * frame. `advance` has no memory of its own dt history, so a single call
 * with that raw delta jumps straight to T_END, skipping the whole
 * choreography instead of just resuming a beat late.
 *
 * Deliberately clamped at the CALL SITE (`Math.min(dtSec, MAX_TIMELINE_DT_SEC)`
 * before calling `advance`), not inside `advance` itself: `advance` stays a
 * simple "add dt, clamp t to [0, T_END]" cursor, which is what lets a test
 * pass an oversized dt directly to pin the end-of-timeline clamp.
 *
 * 150ms is comfortably above any real frame (16.7ms at 60Hz, 33ms at 30Hz)
 * so normal play is never affected, while still keeping even a run of
 * several dropped frames from reading as a visible skip.
 */
export const MAX_TIMELINE_DT_SEC = 0.15

export function clampTimelineDt(dtSec: number): number {
  return Math.min(dtSec, MAX_TIMELINE_DT_SEC)
}

/**
 * Thin, stateful cursor over `stateAt`. Owns only a clock position and a
 * play/pause flag - all the actual choreography lives in the pure functions
 * above, so nothing here needs its own easing or memory.
 */
export class Timeline {
  t = 0
  playing = false

  /**
   * Restarts from 0 when called at (or past) the end, rather than setting
   * `playing = true` and immediately going inert - the next `advance()`
   * would just re-clamp `t` to `T_END` and pause again, so without this the
   * Play button would do nothing once the sweep has finished and only
   * Replay would work. Restarting is the conventional "play" behaviour
   * (matches most media players); a separate Replay button already exists
   * for "restart from wherever you are RIGHT NOW", which this does not
   * replace - Replay still restarts mid-sweep too, where play() does not.
   */
  play(): void {
    if (this.t >= T_END) this.t = 0
    this.playing = true
  }
  pause(): void { this.playing = false }
  toggle(): void { this.playing ? this.pause() : this.play() }

  advance(dtSec: number): void {
    if (!this.playing) return
    this.t += dtSec
    if (this.t >= T_END) {
      this.t = T_END
      this.playing = false
    }
  }

  /** Any scrub takes control away from playback - the two must not fight. */
  scrubTo(t: number): void {
    this.playing = false
    this.t = Math.min(T_END, Math.max(0, t))
  }

  setMatter(m: number): void { this.scrubTo(timeForMatter(m)) }

  get state(): SweepState { return stateAt(this.t) }
}
