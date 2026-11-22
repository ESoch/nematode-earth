import * as THREE from 'three/webgpu'
import { WGS84 } from '../math/wgs84'
import { latLonToDirection } from '../field/geoProbe'

/**
 * Plots the 6,825 soil sample records in the 2020 Figshare release. Uniform
 * screen-space markers show sampling locations at every zoom. Cyan marks the
 * Antarctic biome code, and amber marks every other code. This geographic
 * grouping supplies no per-record information about the 2019 model's inputs.
 *
 * A CPU horizon test removes far-side points. Positions are written relative
 * to the camera before conversion to float32 to preserve precision.
 */

// ---------------------------------------------------------------------------
// Loader: samples_meta.json + samples.bin
// (pipeline/build/bake_nematode_samples.py's output; see its module
// docstring for the authoritative format description this mirrors.)
// ---------------------------------------------------------------------------

/** Byte layout `pipeline/build/bake_nematode_samples.py` bakes: `<fffB`, no padding. */
export const SAMPLE_RECORD_BYTES = 13

const EXPECTED_FIELD_ORDER = [
  'latitude_deg', 'longitude_deg', 'total_per_100g_dry_soil', 'biome_index',
] as const

export interface SamplesMeta {
  n_samples: number
  record_bytes: number
  total_bytes: number
}

/**
 * Validates `json` (parsed `samples_meta.json`) has the shape this module
 * depends on and that its self-reported byte counts are internally
 * consistent -- same discipline as `field/loadCube.ts`'s `parseCubeMeta`:
 * throw a specific, field-level error on any mismatch rather than let a
 * malformed or re-baked-with-a-different-layout sidecar surface later as a
 * silently mis-strided decode.
 */
export function parseSamplesMeta(json: unknown): SamplesMeta {
  if (typeof json !== 'object' || json === null) {
    throw new Error('samples_meta.json: expected a JSON object')
  }
  const obj = json as Record<string, unknown>

  const { n_samples: nSamples } = obj
  if (typeof nSamples !== 'number' || !Number.isInteger(nSamples) || nSamples < 0) {
    throw new Error('samples_meta.json: "n_samples" must be a non-negative integer')
  }

  const { record_bytes: recordBytes } = obj
  if (recordBytes !== SAMPLE_RECORD_BYTES) {
    throw new Error(
      `samples_meta.json: "record_bytes" is ${JSON.stringify(recordBytes)}, expected `
      + `${SAMPLE_RECORD_BYTES} -- decodeSamples only implements the current `
      + '<fffB (lat,lon,total,biome) layout; a re-bake with a different layout '
      + 'must not be silently mis-decoded here',
    )
  }

  const { total_bytes: totalBytes } = obj
  const expectedTotalBytes = recordBytes * nSamples
  if (totalBytes !== expectedTotalBytes) {
    throw new Error(
      `samples_meta.json: "total_bytes" (${JSON.stringify(totalBytes)}) does not match `
      + `record_bytes * n_samples (${expectedTotalBytes})`,
    )
  }

  const { field_order: fieldOrder } = obj
  if (fieldOrder !== undefined) {
    const ok = Array.isArray(fieldOrder)
      && fieldOrder.length === EXPECTED_FIELD_ORDER.length
      && EXPECTED_FIELD_ORDER.every((name, i) => fieldOrder[i] === name)
    if (!ok) {
      throw new Error(
        `samples_meta.json: "field_order" is ${JSON.stringify(fieldOrder)}, expected `
        + `${JSON.stringify(EXPECTED_FIELD_ORDER)} -- decodeSamples assumes this exact `
        + 'field order and offsets',
      )
    }
  }

  return { n_samples: nSamples, record_bytes: recordBytes, total_bytes: totalBytes }
}

export interface DecodedSample {
  latDeg: number
  lonDeg: number
  /** Nematode individuals per 100g dry soil (raw `Total_Number`) -- see this module's doc for why nothing here scales a marker by it. */
  totalPer100g: number
  /** Cleaned WWF biome code (round + Antarctica override), unfiltered -- see `pipeline/build/bake_nematode_samples.py`. */
  biomeIndex: number
}

/**
 * Decodes `buffer` (the fetched `samples.bin`) per `meta`'s record layout:
 * little-endian float32 lat, float32 lon, float32 total, uint8 biome index,
 * 13 bytes/record, no padding -- read with `DataView` rather than a typed
 * array, since 13 is not a multiple of 4 and a strided `Float32Array` view
 * would misalign every record after the first (the same reason
 * `samples_meta.json`'s own `decode_note` calls this out).
 *
 * Throws unless `buffer.byteLength` exactly matches `meta.total_bytes` --
 * mirrors `loadCube.ts`'s `validateFaceBytes`: a short or truncated fetch
 * must fail loudly, not silently decode a partial/garbage last record.
 */
export function decodeSamples(buffer: ArrayBuffer, meta: SamplesMeta): DecodedSample[] {
  if (buffer.byteLength !== meta.total_bytes) {
    throw new Error(
      `samples.bin: expected ${meta.total_bytes} bytes (n_samples=${meta.n_samples} * `
      + `record_bytes=${meta.record_bytes}) but received ${buffer.byteLength}`,
    )
  }
  const view = new DataView(buffer)
  const out: DecodedSample[] = []
  for (let i = 0; i < meta.n_samples; i++) {
    const base = i * meta.record_bytes
    out.push({
      latDeg: view.getFloat32(base + 0, true),
      lonDeg: view.getFloat32(base + 4, true),
      totalPer100g: view.getFloat32(base + 8, true),
      biomeIndex: view.getUint8(base + 12),
    })
  }
  return out
}

/** The dataset authors assign this code to samples south of 60° S. */
export const ANTARCTIC_BIOME_INDEX = 25

export function isAntarcticBiome(biomeIndex: number): boolean {
  return biomeIndex === ANTARCTIC_BIOME_INDEX
}

// ---------------------------------------------------------------------------
// Geometry: position on the true WGS84 ellipsoid, and self-occlusion
// ---------------------------------------------------------------------------

/**
 * WGS 84 surface position in metres. The atlas uses geocentric directions
 * with Z at the north pole. Intersect that same ray with the ellipsoid so
 * the marker stays aligned with the geographic texture at its source latitude.
 */
export function samplePosition(latDeg: number, lonDeg: number): [number, number, number] {
  const [dx, dy, dz] = latLonToDirection(latDeg, lonDeg)
  const radius = 1 / Math.hypot(dx / WGS84.a, dy / WGS84.a, dz / WGS84.b)
  return [dx * radius, dy * radius, dz * radius]
}

/**
 * Unit outward normal to the ellipsoid at `pos` (assumed to already be ON
 * the surface, e.g. from `samplePosition`) -- the gradient of
 * `x^2/a^2 + y^2/a^2 + z^2/b^2`, normalized.
 */
export function surfaceNormal(pos: readonly [number, number, number]): [number, number, number] {
  const a2 = WGS84.a * WGS84.a
  const b2 = WGS84.b * WGS84.b
  const nx = pos[0] / a2
  const ny = pos[1] / a2
  const nz = pos[2] / b2
  const len = Math.hypot(nx, ny, nz)
  return [nx / len, ny / len, nz / len]
}

/**
 * Exact horizon test for a point ON the ellipsoid surface: visible from
 * `camWorld` iff the camera sits on the outward side of the tangent plane
 * at `pointWorld`, i.e. `dot(camWorld - pointWorld, surfaceNormal) > 0`.
 *
 * This is the whole occlusion mechanism this module needs. `Shell.ts`'s
 * raymarch has `depthWrite = false` (it is a volumetric pass, not opaque
 * geometry), so there is no populated GPU depth buffer to depth-test
 * these points against -- a horizon test against the ellipsoid itself is
 * both necessary (the GPU depth buffer cannot do this job here) and
 * sufficient (it is exact for self-occlusion by the planet's own convex
 * body, for any camera above the surface, which every altitude in
 * `ScaleCamera`'s range satisfies). It does not account for terrain relief
 * or atmosphere -- points sit at height 0 on the reference ellipsoid, not
 * on the shaded terrain surface, so there is no elevation for it to occlude
 * against.
 */
export function isPointVisible(
  pointWorld: readonly [number, number, number],
  camWorld: readonly [number, number, number],
): boolean {
  const n = surfaceNormal(pointWorld)
  const dx = camWorld[0] - pointWorld[0]
  const dy = camWorld[1] - pointWorld[1]
  const dz = camWorld[2] - pointWorld[2]
  return dx * n[0] + dy * n[1] + dz * n[2] > 0
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

export interface SamplePointsOpts {
  /** Fixed on-screen size in pixels. Default 3. */
  pointSizePx?: number
  /** Colour for samples with any other biome code. Default amber. */
  otherColor?: THREE.ColorRepresentation
  /** Colour for samples with the Antarctic biome code. Default cyan. */
  antarcticColor?: THREE.ColorRepresentation
  opacity?: number
}

const DEFAULT_POINT_SIZE_PX = 3
const DEFAULT_OTHER_COLOR: THREE.ColorRepresentation = 0xffcc66
const DEFAULT_ANTARCTIC_COLOR: THREE.ColorRepresentation = 0x4fd1ff
const DEFAULT_OPACITY = 0.85

export interface SamplePointsLayer {
  /** Add this to the scene. Stays at the render origin -- see `update()`'s doc for why. */
  points: THREE.Points
  nSamples: number
  nOther: number
  nAntarctic: number
  /**
   * Recomputes which points are visible from `camWorld` (the camera's TRUE
   * planet-relative world position -- `ScaleCamera.worldPosition`, the
   * same value `Shell.ts`'s `cameraWorldPosition` uniform gets) and
   * rewrites the geometry's position/colour buffers, compacted so every
   * visible point sits contiguously at the front -- `geometry.drawRange`
   * is then narrowed to exactly that count, so occluded points are not
   * drawn at all (not merely hidden), with no shader-side discard needed.
   *
   * Every sample's TRUE world position is precomputed once at
   * construction; only the cheap per-frame parts (the horizon test and the
   * small `worldPos - camWorld` subtraction) run here, in plain JS, once
   * per point -- 6,825 iterations is trivial. The subtraction is done in
   * JS's native f64 BEFORE the small residual is written into the f32
   * geometry buffer, matching `ScaleCamera.worldToRender`'s doc ("subtract
   * in f64, hand the small residual to the GPU") -- the two operands here
   * (~6.4e6 m) individually carry sub-metre f32 error, so this subtraction
   * must not happen on the GPU.
   *
   * Call this once per frame before rendering (only when this layer is
   * visible and the camera has moved -- both left to the caller).
   */
  update(camWorld: readonly [number, number, number]): void
  dispose(): void
}

/**
 * Builds the point cloud from already-decoded samples -- the pure,
 * network-free core (`loadSamplePoints` below is the thin fetch wrapper
 * around it, matching `field/loadCube.ts`'s `loadCubeTexture` shape).
 */
export function buildSamplePoints(
  samples: readonly DecodedSample[],
  opts: SamplePointsOpts = {},
): SamplePointsLayer {
  const n = samples.length
  const worldPositions = new Float64Array(n * 3)
  const sourceColors = new Float32Array(n * 3)

  const otherColor = new THREE.Color(opts.otherColor ?? DEFAULT_OTHER_COLOR)
  const antarcticColor = new THREE.Color(opts.antarcticColor ?? DEFAULT_ANTARCTIC_COLOR)

  let nAntarctic = 0
  for (let i = 0; i < n; i++) {
    const s = samples[i]
    if (!s) throw new Error(`buildSamplePoints: missing sample at index ${i}`)
    const [x, y, z] = samplePosition(s.latDeg, s.lonDeg)
    worldPositions[i * 3 + 0] = x
    worldPositions[i * 3 + 1] = y
    worldPositions[i * 3 + 2] = z

    const antarctic = isAntarcticBiome(s.biomeIndex)
    if (antarctic) nAntarctic++
    const c = antarctic ? antarcticColor : otherColor
    sourceColors[i * 3 + 0] = c.r
    sourceColors[i * 3 + 1] = c.g
    sourceColors[i * 3 + 2] = c.b
  }

  const geometry = new THREE.BufferGeometry()
  const positionAttr = new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3)
  const colorAttr = new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3)
  positionAttr.setUsage(THREE.DynamicDrawUsage)
  colorAttr.setUsage(THREE.DynamicDrawUsage)
  geometry.setAttribute('position', positionAttr)
  geometry.setAttribute('color', colorAttr)
  geometry.setDrawRange(0, 0) // nothing drawn until the first update()

  const material = new THREE.PointsMaterial({
    size: opts.pointSizePx ?? DEFAULT_POINT_SIZE_PX,
    sizeAttenuation: false,
    vertexColors: true,
    transparent: true,
    opacity: opts.opacity ?? DEFAULT_OPACITY,
    depthWrite: false, // standard for blended sprites; avoids self-occlusion among overlapping points
    /**
     * Depth testing OFF, and the reason is a correction to this module's own
     * reasoning above.
     *
     * `update()`'s doc argues that no GPU depth test is available because
     * `Shell.ts` runs with `depthWrite = false`. That is true of the shell
     * and irrelevant, because the shell is not the occluder. `sweep/Matter.ts`
     * is OPAQUE (`material.transparent = false`, default depth write) and
     * writing depth is precisely how the sweep occludes the veil — it is the
     * mechanism spec 6.1's governing rule is built on.
     *
     * These points sit at exactly the reference ellipsoid radius, while
     * matter's `positionNode` displaces land outward by `exaggerationAt()`,
     * up to 3.6x real relief. Every sample on land is therefore BURIED inside
     * the terrain and depth-rejected, which is why the layer rendered nothing
     * at all until this was found. Nudging the points outward would need a
     * fudge larger than Everest-times-3.6 and would still lose them in the
     * Himalaya.
     *
     * Disabling the test is also the right behaviour on its own terms: this
     * overlay answers "where did anyone actually sample", and a site does not
     * stop being a site because a mountain is drawn in front of it. Far-side
     * points are already removed by the exact horizon test, which is the cull
     * that carries real meaning here.
     */
    depthTest: false,
  })

  const points = new THREE.Points(geometry, material)
  // Geometry positions are written ALREADY camera-relative (render-space)
  // by update() below -- the object itself never moves from the render
  // origin, the same floating-origin convention Particles.ts's per-instance
  // `records` buffer uses (see that file's doc: "xyz = render-space
  // (camera-relative) position").
  points.frustumCulled = false // this module's own horizon test is the cull; a stale/absent bounding sphere must not additionally hide points

  function update(camWorld: readonly [number, number, number]): void {
    const posArr = positionAttr.array as Float32Array
    const colArr = colorAttr.array as Float32Array
    let visible = 0
    for (let i = 0; i < n; i++) {
      const wx = worldPositions[i * 3 + 0] ?? 0
      const wy = worldPositions[i * 3 + 1] ?? 0
      const wz = worldPositions[i * 3 + 2] ?? 0
      if (!isPointVisible([wx, wy, wz], camWorld)) continue

      const slot = visible++
      posArr[slot * 3 + 0] = wx - camWorld[0]
      posArr[slot * 3 + 1] = wy - camWorld[1]
      posArr[slot * 3 + 2] = wz - camWorld[2]
      colArr[slot * 3 + 0] = sourceColors[i * 3 + 0] ?? 0
      colArr[slot * 3 + 1] = sourceColors[i * 3 + 1] ?? 0
      colArr[slot * 3 + 2] = sourceColors[i * 3 + 2] ?? 0
    }
    geometry.setDrawRange(0, visible)
    positionAttr.needsUpdate = true
    colorAttr.needsUpdate = true
  }

  function dispose(): void {
    geometry.dispose()
    material.dispose()
  }

  return {
    points,
    nSamples: n,
    nOther: n - nAntarctic,
    nAntarctic,
    update,
    dispose,
  }
}

async function fetchJson(url: string): Promise<unknown> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} ${res.statusText}`)
  return res.json() as Promise<unknown>
}

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} ${res.statusText}`)
  return res.arrayBuffer()
}

/**
 * Fetches `${baseUrl}/samples_meta.json` and `${baseUrl}/samples.bin`,
 * validates and decodes them, and builds the point cloud -- the network
 * counterpart to `buildSamplePoints`, matching `loadCube.ts`'s
 * `loadCubeTexture` shape (validate-then-decode, specific errors on any
 * mismatch, no silent mis-stride).
 */
export async function loadSamplePoints(
  baseUrl: string,
  opts: SamplePointsOpts = {},
): Promise<SamplePointsLayer> {
  const metaJson = await fetchJson(`${baseUrl}/samples_meta.json`)
  const meta = parseSamplesMeta(metaJson)
  const buffer = await fetchBuffer(`${baseUrl}/samples.bin`)
  const samples = decodeSamples(buffer, meta)
  return buildSamplePoints(samples, opts)
}
