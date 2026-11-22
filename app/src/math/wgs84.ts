const A = 6378137.0
const INV_F = 298.257223563
const F = 1 / INV_F

export const WGS84 = {
  a: A,
  b: A * (1 - F),
  f: F,
  /** Scalar-radiometry radius only (Q, chord tables). Never used for geometry. */
  R_MEAN: 6371000.0,
} as const

/** (1 - f)^2, the factor relating the two latitude conventions. */
const K = (1 - F) * (1 - F)

/**
 * tan(phi_gc) = (1 - f)^2 * tan(phi_gd).
 *
 * Implemented via atan2 rather than tan/atan so the poles do not go through
 * infinity. Peaks at 11.545 arcmin near 45.1 degrees = 21.4 km of ground
 * displacement, which is a content error (continents in the wrong place) and
 * larger than the sphere-vs-spheroid silhouette error. Spec section 4.4.
 */
export function geodeticToGeocentricLat(latRad: number): number {
  return Math.atan2(K * Math.sin(latRad), Math.cos(latRad))
}

export function geocentricToGeodeticLat(latRad: number): number {
  return Math.atan2(Math.sin(latRad), K * Math.cos(latRad))
}

/** Maximum |phi_gd - phi_gc| in arcminutes, found by ternary search. */
export function maxLatOffsetArcmin(): number {
  const offset = (lat: number) => lat - geodeticToGeocentricLat(lat)
  let lo = 0
  let hi = Math.PI / 2
  for (let i = 0; i < 200; i++) {
    const m1 = lo + (hi - lo) / 3
    const m2 = hi - (hi - lo) / 3
    if (offset(m1) < offset(m2)) lo = m1
    else hi = m2
  }
  return (offset((lo + hi) / 2) * 180 * 60) / Math.PI
}
