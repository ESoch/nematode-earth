import { pinchDistanceToTicks, pointerSeparation } from './orbitInput'

/**
 * Multi-pointer bookkeeping for the viewer's canvas: one finger orbits, two
 * descend.
 *
 * `viewer.ts` tracked exactly one pointer and bound descent to `wheel` alone,
 * so on a phone the interaction the whole piece is built around — descending
 * from orbit through the scale journey — did not exist. Visitors could spin
 * the globe and nothing else.
 *
 * Kept out of `viewer.ts` and free of every DOM type for the same reason
 * `orbitInput.ts` is: the interesting behaviour here is a state machine over
 * pointer arrivals and departures, and it is only checkable without a browser
 * if it never touches one. `test/pointerGesture.test.ts` drives it into a
 * real `ScaleCamera` and asserts the ALTITUDE, not that a handler fired.
 *
 * The invariant that makes it safe:
 *
 *   ANY change to the set of active pointers re-seats the gesture baselines,
 *   so no gesture ever consumes a delta measured across a different set.
 *
 * That single rule is what stops the camera jumping when a finger arrives or
 * leaves. Without it the two obvious defects are a snap of orbit equal to the
 * whole distance between two fingers when one lifts, and a descent equal to
 * the whole initial separation when the second lands. Both are checked.
 *
 * Two fingers suppress orbit entirely rather than also panning by their
 * centroid. That is a deliberate simplification, not an oversight: a pinch on
 * a phone is not a steady gesture, and letting its centroid wander into
 * `orbitBy` while the separation drives altitude makes the globe lurch during
 * every descent. Single-finger orbit is unchanged and is the only way to
 * orbit, which is also the behaviour the always-visible hint promises.
 */

/** A pointer's identity and position, in client pixels. */
export interface PointerSample {
  readonly id: number
  readonly x: number
  readonly y: number
}

/**
 * What one pointermove produced. Both channels are always present and at most
 * one is ever non-zero, so a caller cannot forget to handle a case.
 */
export interface GestureDelta {
  /** Single-pointer drag, client px. Zero unless exactly one pointer is down. */
  readonly dxPx: number
  readonly dyPx: number
  /** `ScaleCamera.scrollBy` ticks from a two-pointer pinch. */
  readonly ticks: number
}

const STILL: GestureDelta = { dxPx: 0, dyPx: 0, ticks: 0 }

export class PointerGesture {
  /**
   * Active pointers by id, in arrival order — `Map` iterates by insertion, so
   * "the pinching pair" is a stable, explainable choice (the first two down)
   * rather than whichever two the browser happened to report last.
   */
  private readonly active = new Map<number, { x: number; y: number }>()

  /**
   * Separation of the pinching pair as of the last re-seat, in client px.
   * Zero whenever fewer than two pointers are down.
   */
  private pinchPx = 0

  get pointerCount(): number {
    return this.active.size
  }

  down(p: PointerSample): void {
    this.active.set(p.id, { x: p.x, y: p.y })
    this.reseat()
  }

  up(id: number): void {
    // A pointer this never saw (a capture released twice, an `up` for a
    // pointer that went down before the listener attached) must not disturb
    // the pointers that ARE down: deleting nothing still re-seats, which is
    // harmless, but it must not throw or clear the map.
    this.active.delete(id)
    this.reseat()
  }

  /**
   * Drop every pointer at once.
   *
   * `viewer.ts` does NOT call this on `pointercancel`: a cancel names one
   * pointer, and dropping the other finger with it would end a two-finger
   * descent that is still half in progress. It is here for a caller that
   * genuinely loses the whole surface, and for tests that reuse one tracker
   * across gestures.
   */
  clear(): void {
    this.active.clear()
    this.pinchPx = 0
  }

  move(p: PointerSample): GestureDelta {
    const prev = this.active.get(p.id)
    // A hovering mouse fires pointermove with no button down and no
    // `pointerdown` before it. It is not part of any gesture.
    if (!prev) return STILL

    const dxPx = p.x - prev.x
    const dyPx = p.y - prev.y
    prev.x = p.x
    prev.y = p.y

    if (this.active.size === 1) return { dxPx, dyPx, ticks: 0 }

    // Two or more: the pair's separation drives altitude and nothing drives
    // orbit. Positions are still tracked above for every pointer, which is
    // what lets a lift hand orbit back to the surviving finger from where
    // that finger actually is rather than from where it last orbited.
    const pair = this.pinchPair()
    if (!pair) return STILL
    const nextPx = pointerSeparation(pair[0], pair[1])
    const ticks = pinchDistanceToTicks(this.pinchPx, nextPx)
    this.pinchPx = nextPx
    return { dxPx: 0, dyPx: 0, ticks }
  }

  /** The two earliest-arrived pointers, or null if fewer than two are down. */
  private pinchPair(): [{ x: number; y: number }, { x: number; y: number }] | null {
    if (this.active.size < 2) return null
    const it = this.active.values()
    const a = it.next().value
    const b = it.next().value
    return a && b ? [a, b] : null
  }

  /**
   * Re-establish the baselines against the CURRENT pointer set, emitting
   * nothing. Called on every arrival and departure — see the class doc.
   *
   * The orbit baseline needs no work here: each pointer's stored position is
   * updated on every move, so a surviving pointer's next delta is already
   * measured from where it is.
   */
  private reseat(): void {
    const pair = this.pinchPair()
    this.pinchPx = pair ? pointerSeparation(pair[0], pair[1]) : 0
  }
}
