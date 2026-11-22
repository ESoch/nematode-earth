import * as THREE from 'three/webgpu'
import { directionToFaceUV, type CubeMeta } from './loadCube'
import { terrainDisplacementM } from './terrain'
import type { Vec3 } from '../math/raySpheroid'

/**
 * How high the matter mesh's SURFACE actually stands above the sphere at a
 * given direction — the CPU mirror of the vertex displacement
 * `sweep/Matter.ts` performs on the GPU, so `core/ScaleCamera.ts` can keep
 * the camera above it.
 *
 * `ScaleCamera.altitude` is a radius offset from `WGS84.a`, and nothing in
 * the camera ever knew that the mesh in front of it is displaced by
 * `elevation * k` (`field/terrain.ts`'s `terrainDisplacementM`). Over Tibet
 * that is ~5 km of rock between the camera's nominal altitude and the
 * ground, so descending anywhere on the Plateau put the camera INSIDE the
 * mesh: the matter material is opaque and front-faced, so from inside there
 * is nothing to draw at all and the frame goes exactly black. Measured at
 * 100 m over Tibet: 0.00% non-zero pixels, against a normally-rendering
 * frame at the same altitude over the Indian Ocean.
 *
 * THE MESH, NOT THE TERRAIN. `MATTER_GEOMETRY` is a fixed 256x128
 * `SphereGeometry`, and displacement is applied PER VERTEX; the surface
 * between vertices is the linear interpolation of those displaced corners,
 * not the elevation atlas resampled. Vertex spacing is 1.40625 degrees in
 * both parameters — ~156 km along the tessellation's own equator, less near
 * its poles (which sit at +-Y in this app's Z-polar frame, i.e. on the
 * equator at 90E/90W, NOT at the geographic poles). So the height an
 * approaching camera collides with can differ substantially from the true
 * elevation under it, and a clamp built on the atlas value alone would be
 * clamping to a surface that is not on screen. `interpolatedDisplacementM`
 * reads those displaced vertices at a point; `cellMaxDisplacementM` bounds
 * the whole cell from above, and that bound — not the point value — is what
 * the floor uses.
 *
 * The bound is deliberately loose. An exact model of the drawn surface has
 * to decide which of each quad's two TRIANGLES a direction lands on, and a
 * raycast against a real displaced `SphereGeometry` showed that decision
 * going wrong often enough, and by enough (one direction in ~120 was out by
 * 1136 m, in the direction that would let the camera through), that a
 * safety clamp cannot rest on it. The corner maximum needs no such decision:
 * every triangle of the cell is a plane through three of these four corners,
 * and a plane through three points never rises above the highest of them
 * anywhere inside their hull, so this bounds the drawn surface whatever the
 * triangulation does. Measured cost, against binary-searched crossovers at
 * three Himalayan sites: the camera is held 339 m, 1507 m and 2835 m higher
 * than strictly necessary. Measured benefit: it is never wrong.
 *
 * Sampling is BILINEAR, matching the `LinearFilter` the cube texture is
 * built with in `field/loadCube.ts`, so the vertex heights computed here are
 * the same ones the vertex shader reads. A nearest-texel mirror would drift
 * from the GPU by up to a full texel of relief.
 */

/** RGBA16Float, matching `loadCube.ts`. */
const CUBE_CHANNELS = 4

/** `viewer.ts`'s `MATTER_GEOMETRY` tessellation. Exported so the two cannot drift apart. */
export const MATTER_WIDTH_SEGMENTS = 256
export const MATTER_HEIGHT_SEGMENTS = 128

export interface ElevationSample {
  /** Raw signed metres — the elevation atlas's R channel carries no log encoding. */
  elevationM: number
  /** The atlas's G channel: land mask as a fraction, blended across coastlines. */
  landFraction: number
}

const clampIndex = (i: number, size: number): number => (i < 0 ? 0 : i > size - 1 ? size - 1 : i)

/**
 * Bilinear CPU sample of an elevation atlas's R (metres) and G (land mask)
 * channels along `dir`, mirroring `field/terrainNodes.ts`'s
 * `sampleElevationNode` / `sampleLandMaskNode`.
 *
 * Edge behaviour is `ClampToEdge` WITHIN a face, matching the texture's own
 * wrap mode. It does not reproduce the GPU's seamless cross-face filtering,
 * which differs only within half a texel of a face boundary — 5 km at the
 * 1024 tier, well inside the tessellation slack `cellMaxDisplacementM`
 * already carries.
 */
export function sampleElevationCube(
  faces: readonly Uint16Array[],
  meta: CubeMeta,
  dir: Vec3,
): ElevationSample {
  const { face, u, v } = directionToFaceUV(dir[0], dir[1], dir[2])
  const faceData = faces[face]
  if (!faceData) throw new Error(`sampleElevationCube: missing face ${face}`)

  const size = meta.face_size
  // Texel CENTRES sit at (i + 0.5) / size in [0,1], so the continuous texel
  // coordinate of a uv is uv*size - 0.5 — the same half-texel offset the
  // hardware sampler applies.
  const fx = ((u + 1) / 2) * size - 0.5
  const fy = ((v + 1) / 2) * size - 0.5
  const x0 = Math.floor(fx)
  const y0 = Math.floor(fy)
  const tx = fx - x0
  const ty = fy - y0

  let elevationM = 0
  let landFraction = 0
  for (let dy = 0; dy <= 1; dy++) {
    for (let dx = 0; dx <= 1; dx++) {
      const px = clampIndex(x0 + dx, size)
      const py = clampIndex(y0 + dy, size)
      const idx = (px + py * size) * CUBE_CHANNELS
      const w = (dx === 0 ? 1 - tx : tx) * (dy === 0 ? 1 - ty : ty)
      elevationM += w * THREE.DataUtils.fromHalfFloat(faceData[idx] ?? 0)
      landFraction += w * THREE.DataUtils.fromHalfFloat(faceData[idx + 1] ?? 0)
    }
  }
  return { elevationM, landFraction }
}

export interface SphereTessellation {
  widthSegments: number
  heightSegments: number
}

export interface CellCorner {
  /** Column in the `SphereGeometry` vertex grid, `0..widthSegments`. */
  ix: number
  /** Row in the `SphereGeometry` vertex grid, `0..heightSegments`. */
  iy: number
  /** Which corner of the cell this is, 0 or 1 in each parameter. */
  dx: number
  dy: number
  /** Unit direction of that vertex. */
  dir: Vec3
  /** Bilinear weight of this corner for the queried direction; the four sum to 1. */
  weight: number
}

export interface SphereCell {
  /** Ordered `(dx,dy)` = (0,0), (1,0), (0,1), (1,1). */
  corners: CellCorner[]
  /** Position within the cell, both in `[0,1)`. */
  tx: number
  ty: number
}

/**
 * The four `SphereGeometry` vertices bracketing `dir`, with the bilinear
 * weights the rasteriser effectively interpolates their displaced positions
 * by.
 *
 * three.js builds a sphere's vertices in a Y-POLAR parametrisation
 * (`SphereGeometry.js`): `phi = 2*pi*ix/W`, `theta = pi*iy/H`, and
 * `(x,y,z) = (-cos(phi) sin(theta), cos(theta), sin(phi) sin(theta))`. The
 * mesh is never rotated in `viewer.ts`, so those tessellation poles land on
 * this app's +-Y axis while the DATA is Z-polar. Inverting the formula, not
 * re-deriving a lat/lon grid, is what keeps the two conventions from being
 * quietly conflated — `groundHeight.test.ts` checks the inverse against a
 * real `SphereGeometry` position buffer rather than against itself.
 */
export function sphereCellCorners(dir: Vec3, tess: SphereTessellation): CellCorner[] {
  return sphereCell(dir, tess).corners
}

/** `sphereCellCorners` plus the within-cell parameters the triangulation needs. */
export function sphereCell(dir: Vec3, tess: SphereTessellation): SphereCell {
  const [x, y, z] = dir
  const len = Math.hypot(x, y, z) || 1
  const ny = y / len
  const theta = Math.acos(ny < -1 ? -1 : ny > 1 ? 1 : ny)
  // phi from x = -cos(phi) sin(theta), z = sin(phi) sin(theta). sin(theta)
  // is non-negative over [0, pi] so it scales both arguments identically and
  // atan2 is unaffected by it; at the poles both are 0 and phi is arbitrary,
  // which is correct — every column meets there.
  let phi = Math.atan2(z / len, -x / len)
  if (phi < 0) phi += 2 * Math.PI

  const fx = (phi / (2 * Math.PI)) * tess.widthSegments
  const fy = (theta / Math.PI) * tess.heightSegments
  const ix0 = Math.floor(fx)
  const iy0 = Math.floor(fy)
  const tx = fx - ix0
  const ty = fy - iy0

  const corners: CellCorner[] = []
  for (let dy = 0; dy <= 1; dy++) {
    for (let dx = 0; dx <= 1; dx++) {
      // Longitude WRAPS (column W is a duplicate of column 0, so either index
      // gives the same direction); latitude clamps at the poles.
      const ix = (ix0 + dx) % tess.widthSegments
      const iy = Math.min(iy0 + dy, tess.heightSegments)
      const vPhi = (2 * Math.PI * ix) / tess.widthSegments
      const vTheta = (Math.PI * iy) / tess.heightSegments
      const st = Math.sin(vTheta)
      corners.push({
        ix,
        iy,
        dx,
        dy,
        dir: [-Math.cos(vPhi) * st, Math.cos(vTheta), Math.sin(vPhi) * st],
        weight: (dx === 0 ? 1 - tx : tx) * (dy === 0 ? 1 - ty : ty),
      })
    }
  }
  return { corners, tx, ty }
}

export interface DisplacementOpts extends SphereTessellation {
  /** `math/descent.ts`'s `exaggerationAt(altitude)` — land only. */
  exaggeration: number
  /** `Timeline.ts`'s `SweepState.bathymetry`; 0 holds the ocean surface at datum. */
  bathymetryProgress: number
}

function cornerDisplacements(
  faces: readonly Uint16Array[],
  meta: CubeMeta,
  dir: Vec3,
  opts: DisplacementOpts,
): Array<{ displacementM: number, weight: number }> {
  return sphereCellCorners(dir, opts).map((corner) => {
    const { elevationM, landFraction } = sampleElevationCube(faces, meta, corner.dir)
    return {
      displacementM: terrainDisplacementM(elevationM, landFraction, opts.exaggeration, opts.bathymetryProgress),
      weight: corner.weight,
    }
  })
}

/**
 * Height of the DRAWN surface above the sphere at `dir`, in metres: the
 * displaced cell corners interpolated exactly as the mesh interpolates them.
 * This is the number to compare a measured camera crossover against when
 * asking whether the crossover tracks the mesh or the terrain.
 */
export function interpolatedDisplacementM(
  faces: readonly Uint16Array[],
  meta: CubeMeta,
  dir: Vec3,
  opts: DisplacementOpts,
): number {
  return cornerDisplacements(faces, meta, dir, opts)
    .reduce((sum, c) => sum + c.weight * c.displacementM, 0)
}

/**
 * Upper bound on the drawn surface over the whole tessellation cell
 * containing `dir` — the maximum of its four displaced corners.
 *
 * This, not `interpolatedDisplacementM`, is what a camera floor must use —
 * see the module doc for why the exact triangle model was measured and
 * rejected. Every triangle of the cell is a plane through three of these
 * four corners, so none of them rises above the maximum anywhere inside the
 * cell; the radial chord between displaced vertices sags further below it
 * still, so the bound has slack in the safe direction at every interior
 * point.
 */
export function cellMaxDisplacementM(
  faces: readonly Uint16Array[],
  meta: CubeMeta,
  dir: Vec3,
  opts: DisplacementOpts,
): number {
  return cornerDisplacements(faces, meta, dir, opts)
    .reduce((max, c) => Math.max(max, c.displacementM), -Infinity)
}
