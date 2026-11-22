import { describe, it, expect } from 'vitest'
import { geocentricToGeodeticLat } from '../src/math/wgs84'
import { parseCubeMeta, type CubeMeta } from '../src/field/loadCube'
import { latLonToDirection, probeGeoLocation, KNOWN_PROBES } from '../src/field/geoProbe'

const deg = (rad: number) => (rad * 180) / Math.PI

describe('latLonToDirection', () => {
  it('produces a unit vector', () => {
    for (const [lat, lon] of [[0, 0], [52.1, 5.3], [-66.5, 100], [89, -179]] as const) {
      const [x, y, z] = latLonToDirection(lat, lon)
      expect(Math.hypot(x, y, z)).toBeCloseTo(1, 10)
    }
  })

  it('is identity-ish at the equator: z is 0 and (x,y) is (cos lon, sin lon)', () => {
    const [x, y, z] = latLonToDirection(0, 30)
    expect(z).toBeCloseTo(0, 10)
    expect(x).toBeCloseTo(Math.cos((30 * Math.PI) / 180), 10)
    expect(y).toBeCloseTo(Math.sin((30 * Math.PI) / 180), 10)
  })

  it('puts the north pole at +Z regardless of longitude', () => {
    const [, , z1] = latLonToDirection(90, 0)
    const [, , z2] = latLonToDirection(90, 173)
    expect(z1).toBeCloseTo(1, 10)
    expect(z2).toBeCloseTo(1, 10)
  })

  it('round-trips through the inverse (geocentricToGeodeticLat + atan2) for known points', () => {
    for (const [latDeg, lonDeg] of [[52.1, 5.3], [-3, -60], [66, 100], [23, 13], [-45.1, 179]] as const) {
      const [x, y, z] = latLonToDirection(latDeg, lonDeg)
      const latGcRecovered = Math.asin(Math.max(-1, Math.min(1, z)))
      const latGdRecovered = deg(geocentricToGeodeticLat(latGcRecovered))
      const lonRecovered = deg(Math.atan2(y, x))
      expect(latGdRecovered).toBeCloseTo(latDeg, 9)
      expect(lonRecovered).toBeCloseTo(lonDeg, 9)
    }
  })

  it('moves the direction slightly EQUATORWARD of the raw geodetic latitude in the north (uses geodeticToGeocentricLat, not its inverse)', () => {
    // If this used geocentricToGeodeticLat instead, the offset would run the
    // other way (poleward) -- this pins down which of the two conversions
    // is correct, per pipeline/build/bake_cube.py's bake_face (geocentric ->
    // geodetic, confirmed by reading the source directly).
    const [, , z] = latLonToDirection(45.1, 0)
    const latGcRad = Math.asin(z)
    expect(deg(latGcRad)).toBeLessThan(45.1)
  })
})

describe('KNOWN_PROBES', () => {
  it('has the four spec-mandated probe locations', () => {
    expect(KNOWN_PROBES.map((p) => p.name)).toEqual(['Netherlands', 'Sahara', 'Amazon', 'Siberia'])
  })
})

describe('probeGeoLocation', () => {
  it('composes latLonToDirection and sampleCubeDensity', () => {
    const size = 2
    const meta: CubeMeta = parseCubeMeta({
      projection: 'plain',
      face_size: size,
      bytes_per_face: size * size * 4 * 2,
      total_bytes: size * size * 4 * 2 * 6,
      face_order: ['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
    })
    // (0,0) geodetic -> equator, prime meridian -> direction (1,0,0) -> +X face.
    const faces = [
      new Uint16Array(size * size * 4),
      new Uint16Array(size * size * 4),
      new Uint16Array(size * size * 4),
      new Uint16Array(size * size * 4),
      new Uint16Array(size * size * 4),
      new Uint16Array(size * size * 4),
    ]
    const result = probeGeoLocation(faces, meta, { name: 'Null Island', latDeg: 0, lonDeg: 0 })
    expect(result.name).toBe('Null Island')
    expect(result.face).toBe(0)
  })
})
