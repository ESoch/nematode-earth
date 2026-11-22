import { describe, it, expect } from 'vitest'
import { intersectShell, toSphereSpace, type Vec3 } from '../src/math/raySpheroid'
import { WGS84 } from '../src/math/wgs84'

const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

const H = 192000
const K_OUTER = 1 + H / WGS84.a

/**
 * Plain-JS mirror of the far-wall entry point `regimes/Shell.ts`'s TSL march
 * computes: the SAME c/q robust-root method `intersectShell`'s `roots()`
 * uses for the near root (already what `tEnter` reproduces), generalized to
 * the inner sphere's FAR root instead. Written and checked against
 * `intersectShell` - the tested, but until now unused, ground truth - BEFORE
 * transcribing the identical arithmetic into TSL, so the shader's algebra is
 * proven correct on the CPU first (TSL node graphs cannot be executed
 * numerically in this environment - see webgpu.test.ts).
 *
 * Deliberately duplicates `intersectShell`'s `roots(1.0)` far branch rather
 * than calling it, because the point of this test IS to pin the exact
 * sequence of operations Shell.ts's TSL will perform (same variable names,
 * same order) - calling `intersectShell` directly would test the reference
 * against itself and prove nothing about the port.
 */
function farWallEntryT(origin: Vec3, dir: Vec3): number {
  const o = toSphereSpace(origin)
  const dRaw = toSphereSpace(dir)
  const dLen = Math.hypot(dRaw[0], dRaw[1], dRaw[2])
  const d: Vec3 = [dRaw[0] / dLen, dRaw[1] / dLen, dRaw[2] / dLen]

  const b2 = dot(o, d)
  const fd: Vec3 = [o[0] - b2 * d[0], o[1] - b2 * d[1], o[2] - b2 * d[2]]
  const L2 = dot(fd, fd)

  const discIn = 1 - L2
  if (discIn <= 0) return NaN

  const sIn = Math.sqrt(discIn)
  const cQuadIn = dot(o, o) - 1
  const qIn = b2 >= 0 ? -sIn - b2 : sIn - b2
  const t0 = cQuadIn / qIn
  const t1 = qIn
  return Math.max(t0, t1) / dLen
}

describe('farWallEntryT (the formula Shell.ts transcribes into TSL)', () => {
  const outward = (alt: number): Vec3 => [WGS84.a + alt, 0, 0]

  it('matches intersectShell.segments[1][0] exactly for a ray through the whole planet', () => {
    const origin = outward(1e6)
    const dir: Vec3 = [-1, 0, 0]
    const r = intersectShell(origin, dir, K_OUTER)
    expect(r.segments).toHaveLength(2)
    expect(farWallEntryT(origin, dir)).toBeCloseTo(r.segments[1]![0], 6)
  })

  it('matches for an oblate polar ray too (checks the geodetic asymmetry carries through)', () => {
    const dist = 1e7
    const origin: Vec3 = [0, 0, dist]
    const dir: Vec3 = [0, 0, -1]
    const r = intersectShell(origin, dir, K_OUTER)
    expect(r.segments).toHaveLength(2)
    expect(farWallEntryT(origin, dir)).toBeCloseTo(r.segments[1]![0], 6)
  })

  it('matches for an off-axis, non-nadir ray through the whole planet', () => {
    const origin: Vec3 = [1.2e7, 3e6, -1.5e6]
    const toCentre = Math.hypot(origin[0], origin[1], origin[2])
    const dir: Vec3 = [-origin[0] / toCentre, -origin[1] / toCentre, -origin[2] / toCentre]
    const r = intersectShell(origin, dir, K_OUTER)
    expect(r.segments).toHaveLength(2)
    expect(farWallEntryT(origin, dir)).toBeCloseTo(r.segments[1]![0], 5)
  })

  it('is NaN (no far wall) for a ray that only grazes the outer shell, one segment', () => {
    const b = WGS84.a + H / 2
    const d = 1e8
    const dir: Vec3 = [-1, b / d, 0]
    const n = Math.hypot(dir[0], dir[1], dir[2])
    const normalizedDir: Vec3 = [dir[0] / n, dir[1] / n, dir[2] / n]
    const origin: Vec3 = [d, 0, 0]
    const r = intersectShell(origin, normalizedDir, K_OUTER)
    expect(r.segments).toHaveLength(1)
    expect(Number.isNaN(farWallEntryT(origin, normalizedDir))).toBe(true)
  })

  it('the far-wall chord is exactly as long as the near-wall chord (justifies reusing the same dt)', () => {
    const origin = outward(1e6)
    const dir: Vec3 = [-1, 0, 0]
    const r = intersectShell(origin, dir, K_OUTER)
    const nearLen = r.segments[0]![1] - r.segments[0]![0]
    const farLen = r.segments[1]![1] - r.segments[1]![0]
    expect(farLen).toBeCloseTo(nearLen, 6)
  })
})
