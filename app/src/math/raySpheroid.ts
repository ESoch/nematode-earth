import { WGS84 } from './wgs84'

export type Vec3 = readonly [number, number, number]

export interface ShellHit {
  hit: boolean
  /** Ray parameters in ORIGINAL space, ordered, entry/exit pairs. */
  segments: Array<[number, number]>
  /**
   * Outer-sphere chord computed directly as 2s, never by differencing roots.
   * Invariant: exactly 0 whenever `hit` is false. Callers must gate on `hit`,
   * never on `outerChord > 0` — a ray can mathematically intersect the outer
   * sphere behind the origin (negative t) while missing on the forward ray;
   * that case reports `hit: false` and this field is zeroed accordingly.
   */
  outerChord: number
}

/**
 * Affine map into sphere space: S = diag(1/a, 1/a, 1/b).
 *
 * Under S the reference ellipsoid becomes the unit sphere and every similar
 * ellipsoid (scale k) becomes a sphere of radius k, so the concentric-shell
 * structure survives and the shared-L^2 trick still applies. Spec section 7.1.
 */
export function toSphereSpace(v: Vec3): Vec3 {
  return [v[0] / WGS84.a, v[1] / WGS84.a, v[2] / WGS84.b]
}

/** Tangent chord through a uniform spherical shell: 2*sqrt(2RH + H^2). */
export function chordThroughShell(R: number, H: number): number {
  return 2 * Math.sqrt(2 * R * H + H * H)
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

/**
 * Ray/shell intersection for two concentric similar ellipsoids.
 *
 * Ray Tracing Gems Ch.7 method 6: the discriminant is formed from the
 * perpendicular vector fd = f - b2*d, so L2 = dot(fd, fd) is INDEPENDENT of
 * radius and is shared across both shell surfaces. Roots go through the c/q
 * branch to avoid catastrophic cancellation, and chord length is returned as
 * 2s directly rather than as a difference of two ~3e7 m values. Spec section 7.2.
 *
 * Method 6 fixes the discriminant, NOT the tangent singularity: d(chord)/dL
 * still diverges at grazing. The shader must pair this with the analytic
 * silhouette antialias and a pixel-footprint step-count floor.
 *
 * @param kOuter outer shell scale, 1 + H_max/a
 */
export function intersectShell(origin: Vec3, dir: Vec3, kOuter: number): ShellHit {
  const miss: ShellHit = { hit: false, segments: [], outerChord: 0 }

  const o = toSphereSpace(origin)
  const dRaw = toSphereSpace(dir)
  const dLen = Math.hypot(dRaw[0], dRaw[1], dRaw[2])
  if (dLen === 0) return miss
  const d: Vec3 = [dRaw[0] / dLen, dRaw[1] / dLen, dRaw[2] / dLen]

  // f is the vector from the sphere centre (origin) to the ray origin.
  const b2 = dot(o, d)
  const fd: Vec3 = [o[0] - b2 * d[0], o[1] - b2 * d[1], o[2] - b2 * d[2]]
  const L2 = dot(fd, fd) // shared by both radii

  const roots = (r: number): [number, number] | null => {
    const disc = r * r - L2
    if (disc <= 0) return null
    const s = Math.sqrt(disc)
    const c = dot(o, o) - r * r
    const q = b2 >= 0 ? -s - b2 : s - b2
    const t0 = c / q
    const t1 = q
    return t0 <= t1 ? [t0, t1] : [t1, t0]
  }

  const outer = roots(kOuter)
  if (!outer) return miss

  // Chord directly as 2s (in sphere space), converted back by /dLen.
  const outerChord = (2 * Math.sqrt(kOuter * kOuter - L2)) / dLen

  const toOriginal = (t: number) => t / dLen
  const inner = roots(1.0)

  const segments: Array<[number, number]> = []
  if (!inner) {
    segments.push([toOriginal(outer[0]), toOriginal(outer[1])])
  } else {
    segments.push([toOriginal(outer[0]), toOriginal(inner[0])])
    segments.push([toOriginal(inner[1]), toOriginal(outer[1])])
  }

  // Clip to the forward ray and drop degenerate segments.
  const forward = segments
    .map(([t0, t1]) => [Math.max(t0, 0), Math.max(t1, 0)] as [number, number])
    .filter(([t0, t1]) => t1 > t0)

  // Forward-clip found nothing: report a total miss, NOT the outerChord just
  // computed above - that value describes an intersection that may exist
  // only behind the ray origin, and must not leak out as if it were valid.
  if (forward.length === 0) return miss
  return { hit: true, segments: forward, outerChord }
}
