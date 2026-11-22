import { geodeticToGeocentricLat } from '../math/wgs84'
import { sampleCubeDensity, type CubeProbe, type CubeMeta } from './loadCube'

/**
 * Direction on the unit sphere for a GEODETIC lat/lon — the convention
 * ordinary coordinates ("52.1N, 5.3E") are always given in — for probing
 * the baked cube the same way `pipeline/build/bake_cube.py` wrote it.
 *
 * The bake goes the OTHER way at build time (`bake_cube.py::bake_face`): a
 * cube direction is inherently geocentric (`asin(z)` has no ellipsoid
 * concept), so for every texel it computes the texel's geocentric latitude
 * and converts it to geodetic via `geocentric_to_geodetic_lat` *before*
 * sampling the geodetic (EPSG:4326) source raster — confirmed directly in
 * `pipeline/build/bake_cube.py`'s `bake_face`, not inferred from the
 * sidecar's prose alone (its `geodetic_to_geocentric.note` describes the
 * same step but is easy to misread as running the other direction).
 * Reproducing a texel's direction from a known geodetic lat/lon therefore
 * needs the INVERSE of that step — `geodeticToGeocentricLat` — applied to
 * the latitude before it's turned into a direction, not
 * `geocentricToGeodeticLat`.
 *
 * Longitude needs no such correction; it is identical in both conventions.
 * `x=cos(lat)cos(lon), y=cos(lat)sin(lon), z=sin(lat)` is this app's
 * Z-polar convention (`math/wgs84.ts`'s doc comment, `ScaleCamera.ts`'s
 * position formula, `pipeline/build/eac.py::lonlat_to_direction`).
 */
export function latLonToDirection(latDegGeodetic: number, lonDegGeodetic: number): [number, number, number] {
  const latGd = (latDegGeodetic * Math.PI) / 180
  const lonRad = (lonDegGeodetic * Math.PI) / 180
  const latGc = geodeticToGeocentricLat(latGd)
  const cl = Math.cos(latGc)
  return [cl * Math.cos(lonRad), cl * Math.sin(lonRad), Math.sin(latGc)]
}

export interface NamedProbe {
  name: string
  latDeg: number
  lonDeg: number
}

/**
 * Orientation-check probes (spec-mandated set): the source paper's measured
 * range spans desert floor to sub-Arctic peak, so these four should read
 * back with enormous contrast if — and only if — the cube's orientation
 * and log-decode are both correct. A mirrored or rotated face is invisible
 * on the synthetic (rotationally featureless) field; it is obvious here.
 */
export const KNOWN_PROBES: readonly NamedProbe[] = [
  { name: 'Netherlands', latDeg: 52.1, lonDeg: 5.3 },
  { name: 'Sahara', latDeg: 23, lonDeg: 13 },
  { name: 'Amazon', latDeg: -3, lonDeg: -60 },
  { name: 'Siberia', latDeg: 66, lonDeg: 100 },
]

export interface GeoProbeResult extends CubeProbe {
  name: string
  latDeg: number
  lonDeg: number
}

/** `latLonToDirection` + `sampleCubeDensity` in one call, for HUD/report use. */
export function probeGeoLocation(
  faces: readonly Uint16Array[],
  meta: CubeMeta,
  probe: NamedProbe,
): GeoProbeResult {
  const dir = latLonToDirection(probe.latDeg, probe.lonDeg)
  return { ...sampleCubeDensity(faces, meta, dir), name: probe.name, latDeg: probe.latDeg, lonDeg: probe.lonDeg }
}
