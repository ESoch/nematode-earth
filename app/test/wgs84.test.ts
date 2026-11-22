import { describe, it, expect } from 'vitest'
import {
  WGS84, geodeticToGeocentricLat, geocentricToGeodeticLat, maxLatOffsetArcmin,
} from '../src/math/wgs84'

const deg = (d: number) => (d * Math.PI) / 180
const toArcmin = (rad: number) => (rad * 180 * 60) / Math.PI

describe('WGS84 constants', () => {
  it('matches the published ellipsoid', () => {
    expect(WGS84.a).toBe(6378137.0)
    expect(WGS84.b).toBeCloseTo(6356752.314245, 6)
    expect(1 / WGS84.f).toBeCloseTo(298.257223563, 9)
  })

  it('derives b from a and f consistently', () => {
    expect(WGS84.a * (1 - WGS84.f)).toBeCloseTo(WGS84.b, 6)
  })
})

describe('geodetic to geocentric latitude', () => {
  it('is identity at the equator and both poles', () => {
    expect(geodeticToGeocentricLat(0)).toBeCloseTo(0, 12)
    expect(geodeticToGeocentricLat(deg(90))).toBeCloseTo(deg(90), 12)
    expect(geodeticToGeocentricLat(deg(-90))).toBeCloseTo(deg(-90), 12)
  })

  it('is odd-symmetric', () => {
    const p = geodeticToGeocentricLat(deg(37))
    const n = geodeticToGeocentricLat(deg(-37))
    expect(p).toBeCloseTo(-n, 12)
  })

  it('moves geocentric equatorward of geodetic in the north', () => {
    expect(geodeticToGeocentricLat(deg(45))).toBeLessThan(deg(45))
  })

  it('peaks at 11.545 arcmin near 45.1 degrees (spec section 4.4)', () => {
    expect(maxLatOffsetArcmin()).toBeCloseTo(11.545, 2)
  })

  it('gives 21.4 km of ground displacement at the peak', () => {
    const km = (maxLatOffsetArcmin() / 60) * 111.32
    expect(km).toBeCloseTo(21.4, 1)
  })

  it('round-trips through the inverse', () => {
    for (const d of [-88, -45.1, -12, 0, 7, 45.1, 66.5, 88]) {
      const gd = deg(d)
      expect(geocentricToGeodeticLat(geodeticToGeocentricLat(gd))).toBeCloseTo(gd, 12)
    }
  })

  it('offset at 45.1 degrees is within a hair of the maximum', () => {
    const off = toArcmin(deg(45.1) - geodeticToGeocentricLat(deg(45.1)))
    expect(off).toBeCloseTo(maxLatOffsetArcmin(), 3)
  })
})
