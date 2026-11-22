import { describe, it, expect } from 'vitest'
import { intersectShell, chordThroughShell, toSphereSpace, type Vec3 } from '../src/math/raySpheroid'
import { WGS84 } from '../src/math/wgs84'

const H = 192000
const K_OUTER = 1 + H / WGS84.a

describe('toSphereSpace', () => {
  it('maps a point on the equatorial surface to the unit sphere', () => {
    const [x, y, z] = toSphereSpace([WGS84.a, 0, 0])
    expect(Math.hypot(x, y, z)).toBeCloseTo(1, 12)
  })

  it('maps the north pole to the unit sphere', () => {
    const [x, y, z] = toSphereSpace([0, 0, WGS84.b])
    expect(Math.hypot(x, y, z)).toBeCloseTo(1, 12)
  })
})

describe('chordThroughShell', () => {
  // Verified table from the spec, section 7.4: chord = 2*sqrt(2RH + H^2)
  const R = 6371000
  it.each([
    [1, 7139],
    [1000, 225770],
    [10000, 714199],
    [32000, 1278699],
    [64000, 1810618],
    [100000, 2266451],
  ])('H=%i m gives a chord of about %i m', (h, expected) => {
    expect(chordThroughShell(R, h)).toBeCloseTo(expected, -1)
  })

  it('gives a limb ratio of 28.29 at H=64 km', () => {
    expect(chordThroughShell(R, 64000) / 64000).toBeCloseTo(28.29, 2)
  })
})

describe('intersectShell', () => {
  const outward = (alt: number): Vec3 => [WGS84.a + alt, 0, 0]

  it('misses entirely when the ray points away from the planet', () => {
    const r = intersectShell(outward(1e6), [1, 0, 0], K_OUTER)
    expect(r.hit).toBe(false)
    expect(r.segments).toHaveLength(0)
  })

  it('produces two segments for a ray through the whole planet', () => {
    const r = intersectShell(outward(1e6), [-1, 0, 0], K_OUTER)
    expect(r.hit).toBe(true)
    expect(r.segments).toHaveLength(2)
  })

  it('produces one segment for a ray that misses the inner sphere', () => {
    // Aim so the impact parameter lands between the inner and outer radii.
    const b = WGS84.a + H / 2
    const d = 1e8
    const dir: Vec3 = [-1, b / d, 0]
    const n = Math.hypot(dir[0], dir[1], dir[2])
    const r = intersectShell([d, 0, 0], [dir[0] / n, dir[1] / n, dir[2] / n], K_OUTER)
    expect(r.hit).toBe(true)
    expect(r.segments).toHaveLength(1)
  })

  it('segments are ordered and non-degenerate', () => {
    const r = intersectShell(outward(1e6), [-1, 0, 0], K_OUTER)
    expect(r.segments.length).toBeGreaterThan(0)
    for (const [t0, t1] of r.segments) {
      expect(t1).toBeGreaterThan(t0)
      expect(t0).toBeGreaterThan(0)
    }
  })

  it('nadir path length through the shell equals the shell thickness', () => {
    const r = intersectShell(outward(1e6), [-1, 0, 0], K_OUTER)
    const first = r.segments[0]!
    expect(first[1] - first[0]).toBeCloseTo(H, 0)
  })

  it('reports the outer chord directly rather than by differencing roots', () => {
    const r = intersectShell(outward(1e6), [-1, 0, 0], K_OUTER)
    const rOut = WGS84.a * K_OUTER
    // Head-on through the centre in sphere space: chord is exactly the full
    // diameter (2*rOut), not merely "some positive value under a loose cap" -
    // that looser check would pass for a function returning a constant like 1.
    expect(r.outerChord).toBeCloseTo(2 * rOut, 3)
  })

  it('zeroes the outer chord on a total miss, never leaking the backward-ray value', () => {
    // Points directly AWAY from the planet: mathematically intersects the
    // outer sphere only behind the origin (negative t), so hit is false.
    const r = intersectShell(outward(1e6), [1, 0, 0], K_OUTER)
    expect(r.hit).toBe(false)
    expect(r.outerChord).toBe(0)
  })

  it('is stable for a grazing ray (no NaN, no negative length)', () => {
    const rOut = WGS84.a * K_OUTER
    const d = 3e7
    const dir: Vec3 = [-1, (rOut * 0.999999) / d, 0]
    const n = Math.hypot(dir[0], dir[1], dir[2])
    const r = intersectShell([d, 0, 0], [dir[0] / n, dir[1] / n, dir[2] / n], K_OUTER)

    // Assert the precondition rather than guarding on it: if this ray ever stops
    // hitting, the test must fail loudly rather than silently asserting nothing.
    expect(r.hit).toBe(true)
    expect(r.segments.length).toBeGreaterThan(0)

    for (const [t0, t1] of r.segments) {
      expect(Number.isFinite(t0)).toBe(true)
      expect(Number.isFinite(t1)).toBe(true)
      expect(t1).toBeGreaterThanOrEqual(t0)
    }
  })

  it('detects the oblateness: a polar ray hits sooner than an equatorial one', () => {
    const dist = 1e7
    const eq = intersectShell([dist, 0, 0], [-1, 0, 0], K_OUTER)
    const po = intersectShell([0, 0, dist], [0, 0, -1], K_OUTER)
    // Entry lands on the OUTER shell (radius scaled by K_OUTER), not the bare
    // reference ellipsoid, so the pole-vs-equator offset is K_OUTER*(a-b)
    // (~22.03 km here), not the unscaled flattening amount a-b (~21.4 km).
    // Verified independently: K_OUTER*(a-b) = 22028.4254024522...,
    // matching the implementation's 22028.4254024527... to 5e-7 m.
    expect(po.segments[0]![0] - eq.segments[0]![0]).toBeCloseTo(K_OUTER * (WGS84.a - WGS84.b), 0)
  })
})
