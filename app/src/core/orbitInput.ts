import type { Vec3 } from '../math/raySpheroid'
import { ALT_RATIO_PER_TICK } from './ScaleCamera'

/**
 * Pure input-mapping and camera-framing math for the interactive viewer.
 * Kept separate from `viewer.ts`'s DOM/render-loop glue specifically so it
 * is unit-testable without a browser or GPU.
 */

/**
 * Pointer-drag pixel delta -> `ScaleCamera.orbitBy(dLon, dLat)` arguments.
 *
 * Scaled by `viewportPx` so the same PHYSICAL drag gesture (e.g. "half the
 * window width") orbits by roughly the same angle regardless of window or
 * canvas resolution, rather than by a fixed pixels-per-radian constant that
 * would feel twitchy on a small canvas and sluggish on a large one.
 *
 * Sign convention (matches the common "orbit the camera around the target"
 * feel used by e.g. three.js's own OrbitControls, not "grab and drag the
 * object" - the two are opposite and both are used in the wild): dragging
 * right increases longitude, dragging up increases latitude (screen Y grows
 * downward, so `dyPx` is negated).
 */
export function dragToOrbit(
  dxPx: number,
  dyPx: number,
  viewportPx: number,
  radiansPerViewport = Math.PI,
): { dLon: number; dLat: number } {
  if (viewportPx <= 0) return { dLon: 0, dLat: 0 }
  const scale = radiansPerViewport / viewportPx
  return { dLon: dxPx * scale, dLat: -dyPx * scale }
}

/**
 * Wheel `deltaY` -> `ScaleCamera.scrollBy(ticks)` argument.
 *
 * Sign convention matches the near-universal map/globe zoom gesture (Google
 * Maps, Google Earth): scrolling up / away from the user - which the
 * `WheelEvent` spec reports as NEGATIVE `deltaY` - zooms in, i.e. descends.
 * `ScaleCamera.scrollBy`'s own doc says "positive ticks descend", so a
 * negative `deltaY` must map to a positive tick count: hence the negation.
 */
export function wheelDeltaToTicks(deltaY: number, pxPerTick = 100): number {
  return -deltaY / pxPerTick
}

/**
 * Two-finger separation change -> `ScaleCamera.scrollBy(ticks)` argument.
 *
 * Descent was bound to `wheel` alone, so on a phone the interaction the whole
 * piece is built around did not exist. This is the touch half of it, and it
 * deliberately returns TICKS rather than an altitude: `scrollBy` stays the
 * single altitude path, so pinch and wheel cannot drift apart and the ground
 * floor in `ScaleCamera.groundFloor` binds both without knowing about either.
 *
 * The mapping is the direct one — the altitude ratio equals the INVERSE of
 * the separation ratio, so spreading the fingers to twice their separation
 * halves the altitude. That is the gesture's whole contract, it holds
 * anywhere on the glass because only the ratio is read, and it composes: a
 * spread delivered over fifty pointermove events lands exactly where the same
 * spread delivered in one does, since logarithms of a product add.
 *
 * Sign convention follows the wheel's: positive ticks descend, and spreading
 * apart (`nextPx > prevPx`) is the universal "zoom in" gesture.
 *
 * Degenerate separations return zero rather than +-Infinity or NaN. Two
 * pointers CAN report identical coordinates — a browser coalescing touches,
 * or a synthesised event — and a single Infinity reaching `scrollBy` would
 * slam the altitude to a clamp with no way back for the user.
 */
export function pinchDistanceToTicks(
  prevPx: number,
  nextPx: number,
  altRatioPerTick: number = ALT_RATIO_PER_TICK,
): number {
  if (!(prevPx > 0) || !(nextPx > 0)) return 0
  return Math.log(nextPx / prevPx) / Math.log(altRatioPerTick)
}

/** Euclidean separation of two pointers, in client pixels. */
export function pointerSeparation(
  a: { x: number, y: number },
  b: { x: number, y: number },
): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

/**
 * "Up" direction for an orbit camera looking toward `direction` (need not be
 * normalized) from outside a sphere whose polar axis is `polarAxis` (default
 * `[0,0,1]`, matching this app's Z-polar convention - see `math/wgs84.ts`,
 * `regimes/Shell.ts`'s `invB`, and `field/synthetic.ts`'s `absLat`, all of
 * which treat Z as the polar axis).
 *
 * Computed as the polar axis projected onto the plane tangent to the sphere
 * at `direction`, then renormalized - the standard technique for a
 * singularity-free orbit-camera up vector (unlike a FIXED world-up, which
 * degenerates into visible roll-flipping as the view direction approaches
 * either pole). Well-defined everywhere except exactly at a pole, where the
 * tangent-plane projection is the zero vector; `ScaleCamera.orbitBy` already
 * clamps latitude to `+-(pi/2 - 1e-6)`, so that exact case cannot arise from
 * normal use - the fallback below exists only so this function has no
 * discontinuity if called directly with a literal pole vector.
 */
/**
 * Idle-drift longitude delta for the auto-rotating camera (see viewer.ts's
 * `autoRotate`), expressed per second and scaled by `dtSec` so the drift is
 * frame-rate independent. One full turn takes about 12 minutes; this is
 * enough movement to show a living globe without moving a region away
 * while a visitor reads the opening.
 */
export const IDLE_AUTOROTATE_RAD_PER_SEC = 2 * Math.PI / (12 * 60)

export function autoRotateDeltaLon(dtSec: number, ratePerSec = IDLE_AUTOROTATE_RAD_PER_SEC): number {
  return ratePerSec * dtSec
}

export function projectedUp(direction: Vec3, polarAxis: Vec3 = [0, 0, 1]): Vec3 {
  const n = Math.hypot(direction[0], direction[1], direction[2])
  if (n < 1e-12) return polarAxis
  const d: Vec3 = [direction[0] / n, direction[1] / n, direction[2] / n]
  const dot = d[0] * polarAxis[0] + d[1] * polarAxis[1] + d[2] * polarAxis[2]
  const proj: Vec3 = [
    polarAxis[0] - dot * d[0],
    polarAxis[1] - dot * d[1],
    polarAxis[2] - dot * d[2],
  ]
  const projLen = Math.hypot(proj[0], proj[1], proj[2])
  // Unreachable via ScaleCamera's clamped latitude - see doc above.
  if (projLen < 1e-9) return polarAxis
  return [proj[0] / projLen, proj[1] / projLen, proj[2] / projLen]
}
