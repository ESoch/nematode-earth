import { describe, it, expect } from 'vitest'
import * as THREE from 'three/webgpu'
import {
  sampleElevationCube, sphereCellCorners, interpolatedDisplacementM,
  cellMaxDisplacementM,
  MATTER_WIDTH_SEGMENTS, MATTER_HEIGHT_SEGMENTS,
} from '../src/field/groundHeight'
import { WGS84 } from '../src/math/wgs84'
import { terrainDisplacementM } from '../src/field/terrain'
import { directionToFaceUV, type CubeMeta } from '../src/field/loadCube'
import { latLonToDirection } from '../src/field/geoProbe'
import type { Vec3 } from '../src/math/raySpheroid'

const CHANNELS = 4

function meta(faceSize: number): CubeMeta {
  return {
    projection: 'plain',
    face_size: faceSize,
    bytes_per_face: faceSize * faceSize * CHANNELS * 2,
    total_bytes: faceSize * faceSize * CHANNELS * 2 * 6,
    face_order: ['+x', '-x', '+y', '-y', '+z', '-z'],
    field: 'elevation',
  }
}

/** Six faces, every texel R = `elevationM`, G (land mask) = `land`. */
function uniformFaces(faceSize: number, elevationM: number, land: number): Uint16Array[] {
  return Array.from({ length: 6 }, () => {
    const f = new Uint16Array(faceSize * faceSize * CHANNELS)
    for (let i = 0; i < faceSize * faceSize; i++) {
      f[i * CHANNELS] = THREE.DataUtils.toHalfFloat(elevationM)
      f[i * CHANNELS + 1] = THREE.DataUtils.toHalfFloat(land)
    }
    return f
  })
}

/**
 * Relief rough enough that cells are far from flat — a flat field would let
 * a wrong triangle choice, or a wrong cell index, agree with the truth by
 * accident.
 */
const ROUGH_SIZE = 64
function roughFaces(): Uint16Array[] {
  return Array.from({ length: 6 }, (_, f) => {
    const a = new Uint16Array(ROUGH_SIZE * ROUGH_SIZE * CHANNELS)
    for (let y = 0; y < ROUGH_SIZE; y++) {
      for (let x = 0; x < ROUGH_SIZE; x++) {
        const i = (x + y * ROUGH_SIZE) * CHANNELS
        a[i] = THREE.DataUtils.toHalfFloat(3000 * Math.sin(x * 0.7 + f) * Math.cos(y * 0.9 - f) + 2000)
        a[i + 1] = THREE.DataUtils.toHalfFloat(1)
      }
    }
    return a
  })
}

/** `uniformFaces`, then one texel (the one `dir` lands on) overwritten. */
function facesWithSpike(
  faceSize: number, baseM: number, land: number, dir: Vec3, spikeM: number,
): Uint16Array[] {
  const faces = uniformFaces(faceSize, baseM, land)
  const { face, u, v } = directionToFaceUV(dir[0], dir[1], dir[2])
  const px = Math.min(faceSize - 1, Math.max(0, Math.floor(((u + 1) / 2) * faceSize)))
  const py = Math.min(faceSize - 1, Math.max(0, Math.floor(((v + 1) / 2) * faceSize)))
  faces[face]![(px + py * faceSize) * CHANNELS] = THREE.DataUtils.toHalfFloat(spikeM)
  return faces
}

describe('sampleElevationCube', () => {
  it('reads the R channel as raw signed metres and G as the land mask', () => {
    const faces = uniformFaces(8, 3400, 1)
    const s = sampleElevationCube(faces, meta(8), latLonToDirection(30, 90))
    expect(s.elevationM).toBeCloseTo(3400, 0)
    expect(s.landFraction).toBeCloseTo(1, 3)
  })

  it('reads negative (bathymetric) elevations, not just land', () => {
    const faces = uniformFaces(8, -4200, 0)
    const s = sampleElevationCube(faces, meta(8), latLonToDirection(-20, -140))
    expect(s.elevationM).toBeCloseTo(-4200, 0)
    expect(s.landFraction).toBeCloseTo(0, 3)
  })

  it('interpolates BILINEARLY, matching the GPU texture\'s LinearFilter', () => {
    // A single raised texel must bleed into its neighbours. Probed exactly on
    // the boundary between the spike texel and the one beside it, where the
    // bilinear answer is the arithmetic mean of the two — a nearest-texel
    // reader would return one or the other, never the midpoint.
    const size = 16
    const dir = latLonToDirection(0, 0)
    const faces = facesWithSpike(size, 0, 1, dir, 8000)
    const { face, u, v } = directionToFaceUV(dir[0], dir[1], dir[2])
    expect(face).toBe(0) // +x, so u = -z/x and v = -y/x invert to dir = [1, -v, -u]
    const px = Math.floor(((u + 1) / 2) * size)
    const py = Math.floor(((v + 1) / 2) * size)
    const uBoundary = ((px + 1) / size) * 2 - 1 // between texel px and px+1
    const vCentre = ((py + 0.5) / size) * 2 - 1 // exactly on row py, no v blend
    const s = sampleElevationCube(faces, meta(size), [1, -vCentre, -uBoundary])
    expect(s.elevationM).toBeCloseTo(4000, 0)
  })
})

describe('sphereCellCorners — the tessellation the matter mesh is actually built on', () => {
  const tess = { widthSegments: MATTER_WIDTH_SEGMENTS, heightSegments: MATTER_HEIGHT_SEGMENTS }

  it('matches the geometry three.js actually generates, vertex for vertex', () => {
    // Not asserted against a re-derivation of the formula — asserted against
    // the real SphereGeometry's own position buffer, so a convention change
    // in three.js (or a sign slip here) fails rather than agreeing with
    // itself. A small sphere keeps the buffer cheap.
    const W = 8
    const H = 4
    const geo = new THREE.SphereGeometry(1, W, H)
    const pos = geo.getAttribute('position')
    const vertexAt = (ix: number, iy: number): Vec3 => {
      const i = iy * (W + 1) + ix
      return [pos.getX(i), pos.getY(i), pos.getZ(i)]
    }
    // A direction squarely inside cell (ix=2, iy=1): the midpoint of that
    // cell's four vertices, renormalised.
    const c = [vertexAt(2, 1), vertexAt(3, 1), vertexAt(2, 2), vertexAt(3, 2)]
    const mid = c.reduce<[number, number, number]>(
      (acc, v) => [acc[0] + v[0] / 4, acc[1] + v[1] / 4, acc[2] + v[2] / 4], [0, 0, 0],
    )
    const len = Math.hypot(...mid)
    const dir: Vec3 = [mid[0] / len, mid[1] / len, mid[2] / len]

    const corners = sphereCellCorners(dir, { widthSegments: W, heightSegments: H })
    expect(corners).toHaveLength(4)
    for (const corner of corners) {
      const expected = vertexAt(corner.ix, corner.iy)
      const elen = Math.hypot(...expected)
      // 6 digits, not more: three.js stores positions as Float32, so the
      // reference itself only carries ~7 significant decimal digits.
      expect(corner.dir[0]).toBeCloseTo(expected[0] / elen, 6)
      expect(corner.dir[1]).toBeCloseTo(expected[1] / elen, 6)
      expect(corner.dir[2]).toBeCloseTo(expected[2] / elen, 6)
    }
    expect(corners.map((k) => `${k.ix},${k.iy}`).sort()).toEqual(['2,1', '2,2', '3,1', '3,2'])
    geo.dispose()
  })

  it('weights sum to one, so the interpolant reproduces a flat field exactly', () => {
    const dir = latLonToDirection(30, 90)
    const corners = sphereCellCorners(dir, tess)
    const total = corners.reduce((s, k) => s + k.weight, 0)
    expect(total).toBeCloseTo(1, 12)
    for (const k of corners) expect(k.weight).toBeGreaterThanOrEqual(0)
  })

  it('wraps in longitude rather than running off the end of the vertex grid', () => {
    // Just short of the phi seam: the "next" column is column 0 again, not
    // column `widthSegments` (which is a duplicate vertex, and indexing past
    // it would be out of range on any smaller grid).
    const dir = latLonToDirection(0, -0.001)
    const corners = sphereCellCorners(dir, tess)
    for (const k of corners) {
      expect(k.ix).toBeGreaterThanOrEqual(0)
      expect(k.ix).toBeLessThanOrEqual(MATTER_WIDTH_SEGMENTS)
      expect(k.iy).toBeGreaterThanOrEqual(0)
      expect(k.iy).toBeLessThanOrEqual(MATTER_HEIGHT_SEGMENTS)
    }
  })
})

describe('displacement of the mesh that is actually drawn', () => {
  const tess = { widthSegments: MATTER_WIDTH_SEGMENTS, heightSegments: MATTER_HEIGHT_SEGMENTS }
  const dir = latLonToDirection(30, 90) // Tibet

  it('reproduces a uniform land field exactly at both the point and the cell max', () => {
    const faces = uniformFaces(64, 4800, 1)
    const opts = { ...tess, exaggeration: 1, bathymetryProgress: 0 }
    expect(interpolatedDisplacementM(faces, meta(64), dir, opts)).toBeCloseTo(4800, 0)
    expect(cellMaxDisplacementM(faces, meta(64), dir, opts)).toBeCloseTo(4800, 0)
  })

  it('applies the land exaggeration but never exaggerates ocean (spec 7.5)', () => {
    const land = uniformFaces(64, 2000, 1)
    const ocean = uniformFaces(64, -2000, 0)
    const opts = { ...tess, exaggeration: 3.6, bathymetryProgress: 1 }
    expect(interpolatedDisplacementM(land, meta(64), dir, opts)).toBeCloseTo(7200, 0)
    expect(interpolatedDisplacementM(ocean, meta(64), dir, opts)).toBeCloseTo(-2000, 0)
  })

  it('holds the ocean surface at datum until the bathymetry beat runs', () => {
    const ocean = uniformFaces(64, -5000, 0)
    const opts = { ...tess, exaggeration: 1, bathymetryProgress: 0 }
    expect(interpolatedDisplacementM(ocean, meta(64), dir, opts)).toBeCloseTo(0, 6)
  })

  // The full-mesh raycasts check geometry correctness even on slower CI runners.
  it('is never pierced by the mesh three.js actually builds', { timeout: 30_000 }, () => {
    // THE load-bearing claim: the camera floor is `cellMaxDisplacementM`, so
    // if any direction exists where the drawn surface stands above it, the
    // camera goes inside the mesh there and the frame goes black — the exact
    // defect the floor exists to stop. Asserted against a real
    // SphereGeometry, displaced vertex by vertex with the same function the
    // shader mirrors and then intersected with THREE.Raycaster, rather than
    // against a second derivation of the same algebra. Double-sided because
    // the ray starts at the planet centre and leaves through the back of
    // every triangle.
    const faces = roughFaces()
    const opts = {
      widthSegments: MATTER_WIDTH_SEGMENTS,
      heightSegments: MATTER_HEIGHT_SEGMENTS,
      exaggeration: 1,
      bathymetryProgress: 0,
    }
    const geo = new THREE.SphereGeometry(WGS84.a, opts.widthSegments, opts.heightSegments)
    const pos = geo.getAttribute('position')
    for (let i = 0; i < pos.count; i++) {
      const v = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i))
      const d = v.clone().normalize()
      const s = sampleElevationCube(faces, meta(ROUGH_SIZE), [d.x, d.y, d.z])
      v.addScaledVector(d, terrainDisplacementM(s.elevationM, s.landFraction, 1, 0))
      pos.setXYZ(i, v.x, v.y, v.z)
    }
    pos.needsUpdate = true
    geo.computeBoundingSphere()
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }))
    mesh.updateMatrixWorld()

    const rc = new THREE.Raycaster()
    rc.far = 1e9
    let checked = 0
    let tightest = Infinity
    for (let i = 0; i < 300; i++) {
      // Deterministic golden-ratio sweep of the whole sphere.
      const theta = Math.acos(1 - 2 * ((i * 0.6180339887) % 1))
      const phi = i * 2.399963
      const d = new THREE.Vector3(
        Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi),
      )
      rc.set(new THREE.Vector3(0, 0, 0), d)
      const hits = rc.intersectObject(mesh)
      if (hits.length === 0) continue
      checked++
      const drawn = hits[0]!.distance - WGS84.a
      const bound = cellMaxDisplacementM(faces, meta(ROUGH_SIZE), [d.x, d.y, d.z], opts)
      expect(bound).toBeGreaterThanOrEqual(drawn)
      tightest = Math.min(tightest, bound - drawn)
    }
    expect(checked).toBeGreaterThan(250)
    // Loose, but not unboundedly so: it is a bound on one cell, not on the planet.
    expect(tightest).toBeLessThan(2000)
    geo.dispose()
  })

  it('bounds the interpolated surface from above — that is what makes it a safe floor', () => {
    // A spike on ONE cell corner: the mesh between vertices is a linear
    // interpolation toward it, so the point value is strictly below the
    // corner maximum. A clamp built on the point value alone would let the
    // camera through the mesh anywhere off-vertex.
    const size = 512
    const corners = sphereCellCorners(dir, tess)
    const spikeCorner = corners[0]!
    const faces = facesWithSpike(size, 500, 1, spikeCorner.dir, 8800)
    const opts = { ...tess, exaggeration: 1, bathymetryProgress: 0 }
    const point = interpolatedDisplacementM(faces, meta(size), dir, opts)
    const cellMax = cellMaxDisplacementM(faces, meta(size), dir, opts)
    expect(cellMax).toBeGreaterThan(point)
    expect(cellMax).toBeGreaterThan(500)
    expect(cellMax).toBeGreaterThanOrEqual(point)
  })
})
