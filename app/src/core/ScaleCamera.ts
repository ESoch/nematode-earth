import { WGS84 } from '../math/wgs84'
import { descentAt, type DescentState } from '../math/descent'
import type { Vec3 } from '../math/raySpheroid'

export const ALT_MIN = 0.01
export const ALT_MAX = 4e7
/**
 * One scroll tick changes altitude by this factor. ~26 ticks per decade.
 *
 * Exported because `scrollBy` is now the ONLY way anything reaches the
 * altitude — wheel and pinch alike — and `orbitInput.ts` has to convert a
 * pinch's distance ratio into this unit to get there. A pinch that computed
 * its own altitude directly would be a second altitude path, free to drift
 * away from the wheel's (and from the ground floor below).
 */
export const ALT_RATIO_PER_TICK = 1.09
const TICK = ALT_RATIO_PER_TICK
const QUALITY_APPROACH_LOG_BAND = Math.log(1.25)

/**
 * How close the camera may stand to the displaced mesh, in metres.
 *
 * Set by `viewer.ts`'s PerspectiveCamera near plane, which is 1 m: at any
 * smaller clearance the ground directly beneath the camera is nearer than
 * the near plane, gets clipped away, and — because backface culling means a
 * clipped or culled surface occludes nothing — the viewer looks straight
 * THROUGH the ground at whatever lower terrain lies beyond. The frame is
 * not black, so a "did the screen go dark" check passes while the defect is
 * fully present. 2 m is the smallest round figure that clears it.
 *
 * Deliberately not larger. The piece's closest shot is a hand's breadth off
 * the ground (spec 8.3's count-in-view falling to a handful), and a floor
 * generous enough to feel safe would delete that shot over every land
 * surface on Earth. Over water this constant does not bind at all: the
 * drawn ocean surface sags BELOW datum between vertices (see
 * `field/groundHeight.ts`), so the floor there is zero and `ALT_MIN`
 * remains reachable.
 */
export const GROUND_CLEARANCE_M = 2

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

/**
 * Height of the drawn surface above the sphere at `dir`, in metres — see
 * `field/groundHeight.ts`'s `cellMaxDisplacementM`, which is what
 * `viewer.ts` supplies. Routinely NEGATIVE, and legitimately so: over water
 * every corner of a cell sits at datum or below, so the bound is zero or
 * less and the floor is inert there.
 *
 * `null` means "there is no ground here to stand on" — no elevation atlas
 * has loaded yet, or the sweep has taken matter away. That is a different
 * statement from "the ground is at zero", and conflating the two silently
 * imposes the clearance as a global minimum altitude.
 */
export type GroundHeightAt = (dir: Vec3) => number | null

/**
 * The composition the piece opens on: a full-globe framing centred on the
 * Atlantic at approximately 10°N, 10°W: Africa, Europe and the Atlantic.
 * This is the candidate composition for visual comparison, at the existing
 * full-globe altitude. Responsive framing changes projection, not altitude.
 *
 * Exported rather than inlined in `viewer.ts` because it is half of a PAIR.
 * `sweep/Matter.ts`'s existing `SUN_DIR` remains unchanged. The opening
 * controller starts here, drifts slowly while idle, and holds the view
 * during playback or after an explicit gesture.
 */
export const OPENING_POSE = {
  lonRad: -10 * Math.PI / 180,
  latRad: 10 * Math.PI / 180,
  altM: 1.15e7,
} as const

/**
 * Scale-coupled camera with a floating origin.
 *
 * Two properties that cannot be retrofitted (spec section 8.1):
 *
 * 1. Every motion parameter derives from altitude, and scroll is LOGARITHMIC,
 *    so constant input gives constant perceptual approach across all ten orders
 *    of magnitude.
 * 2. The camera sits permanently at the render origin and the world translates
 *    around it. At millimetre scale, f32 world coordinates anchored at the
 *    planet centre lose catastrophic precision (ULP at 6371 km is 0.5 m).
 *
 * Altitude is a radius offset from `WGS84.a`, which is NOT the surface the
 * viewer sees: `sweep/Matter.ts` displaces its mesh by `elevation * k`, so
 * over high terrain the ground stands kilometres above the datum the
 * altitude is measured from. `groundHeightAt` closes that gap — see
 * `GROUND_CLEARANCE_M` and `field/groundHeight.ts`.
 */
export class ScaleCamera {
  /**
   * The altitude the USER asked for, before the ground floor. Kept separate
   * from the effective one so orbiting over a mountain and back off it is not
   * a one-way trip: the floor lifts the camera while it is over high ground
   * and releases it afterwards, rather than silently rewriting the request.
   */
  private _alt = 1e7
  private _qualityFloorM = ALT_MIN
  private _lon = 0
  private _lat = 0

  /**
   * Supplies the height of the drawn surface under the camera. Null (the
   * default) means no floor at all — every existing caller and every test
   * that predates the floor keeps its exact previous behaviour, and the
   * viewer only attaches a sampler once a real elevation atlas has landed.
   */
  groundHeightAt: GroundHeightAt | null = null

  /**
   * Public detail limit, supplied from the dataset and current CSS projection.
   * Independent of matter visibility. Standalone diagnostic cameras retain
   * ALT_MIN until configured; the viewer always supplies its quality floor.
   * Raising it consumes the old request, so a later resize/upgrade cannot
   * unexpectedly zoom in. It never weakens the physical ground floor.
   */
  get qualityFloorM(): number { return this._qualityFloorM }
  set qualityFloorM(m: number) {
    if (!Number.isFinite(m)) return
    this._qualityFloorM = clamp(m, ALT_MIN, ALT_MAX)
    this._alt = Math.max(this._alt, this._qualityFloorM)
  }

  /** Show the detail hint only when quality, rather than terrain, stops us. */
  get atQualityFloor(): boolean {
    return this._qualityFloorM > ALT_MIN && this.altitude <= this._qualityFloorM * (1 + 1e-9)
  }

  /** Effective altitude: never below the mesh under the camera. */
  get altitude(): number {
    return Math.max(this._alt, this.groundFloor)
  }

  /** What the user asked for, ignoring the ground floor. Diagnostics only. */
  get requestedAltitude(): number { return this._alt }

  /**
   * Lowest altitude the camera may occupy at its current direction. Zero
   * without a sampler, or wherever the sampler reports no ground at all.
   *
   * Ground at or below the datum imposes nothing at all, rather than
   * `max(0, h + clearance)`. Over open water the bound `viewer.ts` supplies
   * is exactly zero across whole cells — every corner sits at datum until
   * the bathymetry beat drops it — so adding the clearance there would
   * quietly raise the minimum altitude over every ocean on Earth from
   * `ALT_MIN` to the clearance, deleting a shot that works today for no
   * reason connected to the defect being fixed. The step this introduces at
   * a coastline is one clearance, it lands on a cell that contains land,
   * and it does not oscillate, because the bound does not cross zero
   * repeatedly the way a point-sampled surface would.
   */
  get groundFloor(): number {
    if (!this.groundHeightAt) return 0
    const h = this.groundHeightAt(this.direction)
    if (h === null || h <= 0) return 0
    return h + GROUND_CLEARANCE_M
  }

  get longitude(): number { return this._lon }
  get latitude(): number { return this._lat }

  /** Unit direction from the planet centre to the camera. */
  get direction(): Vec3 {
    const cl = Math.cos(this._lat)
    return [cl * Math.cos(this._lon), cl * Math.sin(this._lon), Math.sin(this._lat)]
  }

  get worldPosition(): Vec3 {
    const r = WGS84.a + this.altitude
    const d = this.direction
    return [r * d[0], r * d[1], r * d[2]]
  }

  get state(): DescentState {
    return descentAt(this.altitude, 0.047)
  }

  setAltitude(m: number): void {
    if (Number.isNaN(m)) return
    this._alt = clamp(m, this._qualityFloorM, ALT_MAX)
  }

  /**
   * Positive ticks descend. Multiplicative outside the quality approach band;
   * within 25% of that floor, inward movement softens and stops finitely.
   *
   * Measured from the EFFECTIVE altitude, not the requested one: a camera
   * held up by the ground floor should respond to a scroll outward
   * immediately, rather than first spending back however many ticks it
   * burned pushing into the mountain.
   */
  scrollBy(ticks: number): void {
    if (Number.isNaN(ticks)) return
    const current = this.altitude
    const floor = this._qualityFloorM
    if (ticks > 0 && floor > ALT_MIN) {
      const x = Math.log(current / floor)
      const band = QUALITY_APPROACH_LOG_BAND
      const s = Math.log(TICK) * ticks
      const distanceAboveBand = Math.max(0, x - band)
      if (s > distanceAboveBand) {
        // Consume ordinary log distance first, then integrate dx/ds =
        // -sqrt(x/band) exactly. Input fragments compose without storing a
        // hidden target or overscroll debt. Outward motion stays untapered.
        const root = Math.max(0, Math.sqrt(Math.min(x, band))
          - (s - distanceAboveBand) / (2 * Math.sqrt(band)))
        this.setAltitude(floor * Math.exp(root * root))
        return
      }
    }
    this.setAltitude(current / Math.pow(TICK, ticks))
  }

  orbitBy(dLon: number, dLat: number): void {
    this.setOrientation(this._lon + dLon, this._lat + dLat)
  }

  /** Absolute pose, in radians. Latitude is GEOCENTRIC, as in `worldPosition`. */
  setOrientation(lonRad: number, latRad: number): void {
    this._lon = lonRad
    this._lat = clamp(latRad, -Math.PI / 2 + 1e-6, Math.PI / 2 - 1e-6)
  }

  /** Subtract in f64, hand the small residual to the GPU. */
  worldToRender(p: Vec3): Vec3 {
    const c = this.worldPosition
    return [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
  }
}
