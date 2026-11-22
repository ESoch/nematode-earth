import * as THREE from 'three/webgpu'
import { E_A } from '../math/optical'

/**
 * Fetches the real baked atlas (`pipeline/build/bake_cube.py`'s output —
 * `app/public/data/cube_plain_128/{cube_meta.json,face_{0..5}.bin}`) and
 * assembles it into a `THREE.CubeTexture` the existing shader path can
 * consume identically to `bakeCube.ts`'s synthetic one.
 *
 * The one real difference from `bakeCube.ts`: the baked atlas is
 * RGBA16Float (log-density in R, composition in G, CV in B, density support
 * in A), not R16Float. Soil v2 stores G as a support-weighted numerator
 * with six independent scalar support faces; it does not infer G validity
 * from density support. Shell currently consumes R/B/A only.
 *
 * This module only speaks the PLAIN (gnomonic) projection, matching
 * `Shell.ts`'s `ShellOpts.cube` sampling, which feeds `cubeTexture(tex, dir)`
 * directly with no EAC unwarp. EAC atlases used to be baked alongside the
 * plain ones and are no longer shipped: nothing here could ever have loaded
 * one correctly (the legacy metadata path accepts any non-empty projection
 * string, while soil v2 explicitly requires `"plain"`), and the projection's own
 * measured benefit — worst-case 1.27x texel improvement, identical RMS —
 * never justified the extra runtime ALU. `pipeline/build/eac.py` and its
 * tests remain; the warp math is still correct and still worth keeping, it
 * simply has no consumer. Re-bake with `--projections eac` if that changes.
 */

export const CUBE_FACE_COUNT = 6
/** RGBA16Float: 4 channels, 2 bytes each. */
const CUBE_CHANNELS = 4
const BYTES_PER_CHANNEL = 2
const SOIL_V2 = 'soil-cube-candidate-v2'
const FACE_ORDER = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'] as const

export interface GroupSupportMeta {
  format: 'R16Float'
  bytes_per_face: number
  total_bytes: number
  face_order: readonly string[]
  sha256: Readonly<Record<string, string>>
}

export interface CubeMeta {
  projection: string
  face_size: number
  bytes_per_face: number
  total_bytes: number
  face_order: readonly string[]
  /** Present on the elevation atlases (not the density one) — ground size of one texel, kilometres. */
  km_per_texel?: number
  /**
   * What the R channel actually holds — `'elevation'`, `'built_up_fraction'`,
   * or absent/`'density'`. `needsTauEncoding` dispatches on it; see that
   * function for why guessing is not an option.
   */
  field?: string
  schema?: 'soil-cube-candidate-v1' | typeof SOIL_V2
  /** Only v2 declares independent composition coverage. Legacy G validity is unknown. */
  groups_policy?: 'separate-coverage'
  group_support?: GroupSupportMeta
  face_sha256?: Readonly<Record<string, string>>
}

function assertFaceOrder(value: unknown, name: string): asserts value is string[] {
  if (!Array.isArray(value) || value.length !== CUBE_FACE_COUNT
    || !FACE_ORDER.every((face, i) => value[i] === face)) {
    throw new Error(`cube_meta.json: ${name} must be ${FACE_ORDER.join(',')}`)
  }
}

function parseFaceHashes(value: unknown, prefix: string, name: string): Readonly<Record<string, string>> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`cube_meta.json: ${name} must contain six SHA-256 checksums`)
  }
  const hashes = value as Record<string, unknown>
  const names = Array.from({ length: CUBE_FACE_COUNT }, (_, i) => `${prefix}_${i}.bin`)
  if (Object.keys(hashes).length !== CUBE_FACE_COUNT || !names.every((file) =>
    Object.hasOwn(hashes, file) && typeof hashes[file] === 'string' && /^[a-f0-9]{64}$/i.test(hashes[file]))) {
    throw new Error(`cube_meta.json: ${name} must have exactly ${names.join(',')} with SHA-256 hex digests`)
  }
  return Object.fromEntries(names.map((file) => [file, (hashes[file] as string).toLowerCase()]))
}

function parseGroupSupport(value: unknown, faceSize: number): GroupSupportMeta {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('cube_meta.json: group_support must be an R16Float descriptor')
  }
  const support = value as Record<string, unknown>
  if (support.format !== 'R16Float') throw new Error('cube_meta.json: group_support.format must be R16Float')
  const bytesPerFace = faceSize * faceSize * BYTES_PER_CHANNEL
  if (support.bytes_per_face !== bytesPerFace) {
    throw new Error('cube_meta.json: group_support.bytes_per_face must match face_size² * 2')
  }
  if (support.total_bytes !== bytesPerFace * CUBE_FACE_COUNT) {
    throw new Error('cube_meta.json: group_support.total_bytes must match six scalar faces')
  }
  assertFaceOrder(support.face_order, 'group_support.face_order')
  return {
    format: 'R16Float', bytes_per_face: bytesPerFace, total_bytes: bytesPerFace * CUBE_FACE_COUNT,
    face_order: [...support.face_order],
    sha256: parseFaceHashes(support.sha256, 'groups_support', 'group_support.sha256'),
  }
}

/**
 * Validates `json` (parsed `cube_meta.json`) has the shape this module
 * depends on, and that its self-reported byte counts are internally
 * consistent — throws a specific, field-level error on any mismatch rather
 * than letting a malformed sidecar surface later as a cryptic TypeError (or
 * worse, an aliased/garbage texture) deep in the fetch path.
 */
export function parseCubeMeta(json: unknown): CubeMeta {
  if (typeof json !== 'object' || json === null) {
    throw new Error('cube_meta.json: expected a JSON object')
  }
  const obj = json as Record<string, unknown>

  const { projection } = obj
  if (typeof projection !== 'string' || projection.length === 0) {
    throw new Error('cube_meta.json: "projection" must be a non-empty string')
  }

  const { face_size: faceSize } = obj
  if (typeof faceSize !== 'number' || !Number.isInteger(faceSize) || faceSize <= 0) {
    throw new Error('cube_meta.json: "face_size" must be a positive integer')
  }

  const { bytes_per_face: bytesPerFace } = obj
  if (typeof bytesPerFace !== 'number' || !Number.isInteger(bytesPerFace) || bytesPerFace <= 0) {
    throw new Error('cube_meta.json: "bytes_per_face" must be a positive integer')
  }

  const { total_bytes: totalBytes } = obj
  if (typeof totalBytes !== 'number' || !Number.isInteger(totalBytes) || totalBytes <= 0) {
    throw new Error('cube_meta.json: "total_bytes" must be a positive integer')
  }

  const { face_order: faceOrder } = obj
  if (
    !Array.isArray(faceOrder)
    || faceOrder.length !== CUBE_FACE_COUNT
    || !faceOrder.every((f) => typeof f === 'string')
  ) {
    throw new Error(`cube_meta.json: "face_order" must be an array of ${CUBE_FACE_COUNT} strings`)
  }

  const expectedBytesPerFace = faceSize * faceSize * CUBE_CHANNELS * BYTES_PER_CHANNEL
  if (bytesPerFace !== expectedBytesPerFace) {
    throw new Error(
      `cube_meta.json: bytes_per_face (${bytesPerFace}) does not match face_size `
      + `(${faceSize}) — expected ${expectedBytesPerFace} for RGBA16Float`,
    )
  }
  const expectedTotalBytes = bytesPerFace * CUBE_FACE_COUNT
  if (totalBytes !== expectedTotalBytes) {
    throw new Error(
      `cube_meta.json: total_bytes (${totalBytes}) does not match `
      + `bytes_per_face * ${CUBE_FACE_COUNT} (${expectedTotalBytes})`,
    )
  }

  const { km_per_texel: kmPerTexel } = obj
  if (kmPerTexel !== undefined && (typeof kmPerTexel !== 'number' || kmPerTexel <= 0)) {
    throw new Error('cube_meta.json: "km_per_texel", when present, must be a positive number')
  }

  const { field } = obj
  if (field !== undefined && (typeof field !== 'string' || field.length === 0)) {
    throw new Error('cube_meta.json: "field", when present, must be a non-empty string')
  }

  const meta: CubeMeta = {
    projection,
    face_size: faceSize,
    bytes_per_face: bytesPerFace,
    total_bytes: totalBytes,
    face_order: faceOrder,
    ...(kmPerTexel !== undefined ? { km_per_texel: kmPerTexel } : {}),
    ...(field !== undefined ? { field } : {}),
  }
  const { schema } = obj
  if (schema !== undefined && schema !== 'soil-cube-candidate-v1' && schema !== SOIL_V2) {
    throw new Error('cube_meta.json: unsupported schema')
  }
  if (schema !== undefined) meta.schema = schema
  if (schema === SOIL_V2) {
    if (projection !== 'plain') throw new Error('cube_meta.json: v2 projection must be plain')
    if (field !== 'density') throw new Error('cube_meta.json: v2 field must be density')
    if (!Number.isSafeInteger(totalBytes)) throw new Error('cube_meta.json: v2 dimensions exceed safe byte counts')
    assertFaceOrder(faceOrder, 'face_order')
    if (obj.groups_policy !== 'separate-coverage') {
      throw new Error('cube_meta.json: v2 groups_policy must be separate-coverage')
    }
    meta.groups_policy = 'separate-coverage'
    meta.group_support = parseGroupSupport(obj.group_support, faceSize)
    meta.face_sha256 = parseFaceHashes(obj.face_sha256, 'face', 'face_sha256')
  } else if (obj.group_support !== undefined || obj.groups_policy !== undefined) {
    throw new Error('cube_meta.json: group_support/groups_policy require schema soil-cube-candidate-v2')
  }
  return meta
}

/**
 * Whether this atlas's R channel is a `log(density)` that
 * `densityFaceToTauFace` should re-encode to `log(tau)`.
 *
 * This dispatch exists because `loadCubeTexture` used to apply that re-encode
 * to EVERY atlas it loaded, and the re-encode is an unconditional
 * `R += ln(E_A)`, i.e. `R -= 18.02`. On the density atlas that is exactly
 * right. On the other two it was silently destructive:
 *
 *  - The urban atlas holds a built-up FRACTION in `[0, 1]`. Every texel on
 *    Earth became about -18, and `cityLightIntensityNode` clamps to `[0, 1]`
 *    before its gamma — so city lights evaluated to exactly zero on every
 *    fragment, in every frame, from the commit that introduced them. Nothing
 *    threw. The feature was multiplied out of existence and looked like a
 *    tuning problem.
 *  - The elevation atlas holds metres, and carried a flat -18.02 m bias:
 *    small against 8849 m peaks, but it moves the shoreline off the datum
 *    the colour ramp and the land mask are both defined against.
 *
 * An unrecognised field THROWS rather than falling through to either branch.
 * A silent default is precisely what let this live, so a newly-baked atlas
 * has to state its encoding here before it can be loaded at all.
 */
export function needsTauEncoding(meta: CubeMeta): boolean {
  switch (meta.field) {
    // The density atlas predates the `field` key, so absent means density.
    case undefined:
    case 'density':
      return true
    case 'elevation':
    case 'built_up_fraction':
      return false
    default:
      throw new Error(
        `cube_meta.json: field "${meta.field}" has no R-channel encoding rule in `
        + 'loadCube.ts::needsTauEncoding. Add one — do not let it default, since '
        + 'the wrong branch corrupts every texel without erroring.',
      )
  }
}

/**
 * Throws unless `buffer`'s byte length exactly matches `meta.bytes_per_face`.
 * A short, truncated, or wrong-size fetch would otherwise silently
 * misalign every texel after the gap (or read past the buffer) instead of
 * failing loudly — indistinguishable from a shader bug without this check.
 */
export function validateFaceBytes(faceIndex: number, byteLength: number, meta: CubeMeta): void {
  if (byteLength !== meta.bytes_per_face) {
    throw new Error(
      `face_${faceIndex}.bin: expected ${meta.bytes_per_face} bytes `
      + `(face_size=${meta.face_size}, RGBA16Float) but received ${byteLength}`,
    )
  }
}

export interface LoadedCube {
  texture: THREE.CubeTexture<THREE.DataTexture>
  meta: CubeMeta
  /**
   * Raw per-face texel data (RGBA16Float bit patterns as uint16, row-major,
   * `(px + py*size)*4 + channel`), same face order as `texture.images`.
   * Kept alongside the GPU texture so the HUD can CPU-decode a probe point
   * without a readback — see `geoProbe.ts`.
   */
  faces: readonly Uint16Array[]
  /** V2 composition coverage (R16Float bits); independent of density's A. */
  groupSupportFaces?: readonly Uint16Array[]
}

const LOG_E_A = Math.log(E_A)

/**
 * Converts one fetched face's R channel from log(density_per_sqm) — what
 * the pipeline bakes, per `cube_meta.json`'s `channels.R` ("decode with
 * exp(R)" gives density, individuals/m^2) — to log(tau), tau = E_A *
 * density_per_sqm (`math/optical.ts`'s `tauNadir`). That is the quantity
 * `regimes/Shell.ts`'s cube-sampling branch actually expects: it decodes
 * with a bare `exp(R)` and feeds the result straight into the optical-depth
 * march, mirroring `bakeCube.ts`'s synthetic bake, where R already IS
 * log(tau) (`encodeLogTau`). Skipping this conversion would hand the
 * shader raw individual counts (~1e6-1e7) as if they were already an
 * optical depth — many orders of magnitude past `tauCap` — so `min(tau,
 * tauCap)` clamps every land AND ocean texel to the same constant,
 * rendering uniform despite real density spanning orders of magnitude.
 *
 * Returns a NEW array — `loadCubeTexture`'s returned `faces` (used for the
 * CPU-side density probes in `geoProbe.ts`) must keep decoding to actual
 * density, so the raw fetched bytes are never mutated in place.
 *
 * G, B, A pass through unchanged. In v2 G remains a weighted numerator;
 * only a composition consumer divides by its separate support, never A.
 */
export function densityFaceToTauFace(faceData: Uint16Array): Uint16Array {
  const out = new Uint16Array(faceData.length)
  for (let i = 0; i < faceData.length; i += CUBE_CHANNELS) {
    const rBits = faceData[i]
    if (rBits === undefined) throw new Error(`densityFaceToTauFace: texel index ${i} out of range`)
    const logDensity = THREE.DataUtils.fromHalfFloat(rBits)
    out[i] = THREE.DataUtils.toHalfFloat(logDensity + LOG_E_A)
    out[i + 1] = faceData[i + 1] ?? 0
    out[i + 2] = faceData[i + 2] ?? 0
    out[i + 3] = faceData[i + 3] ?? 0
  }
  return out
}

/**
 * Stride between sampled texels in `meanLandMaskFraction`. A 1024-face
 * atlas is 6.3M texels and the answer wanted is a mean, so every 16th texel
 * still gives ~390k samples — far more than enough to separate "populated"
 * from "identically zero", at a sixteenth of the load-time cost.
 */
const LAND_MASK_SAMPLE_STRIDE = 16

/** Mean of the elevation atlas's G (land-mask) channel over a strided sample of every face. */
export function meanLandMaskFraction(faces: readonly Uint16Array[]): number {
  let total = 0
  let count = 0
  for (const face of faces) {
    for (let i = 1; i < face.length; i += CUBE_CHANNELS * LAND_MASK_SAMPLE_STRIDE) {
      total += THREE.DataUtils.fromHalfFloat(face[i] ?? 0)
      count++
    }
  }
  return count === 0 ? 0 : total / count
}

/**
 * Loosest band the elevation atlas's mean land mask can fall in and still be
 * a real mask. Earth's land fraction is 0.292 and the pipeline gate holds
 * the bake to [0.285, 0.299], but that gate is AREA-weighted and cube faces
 * are not — texels near a face corner subtend far less solid angle than
 * texels at its centre — so a raw per-texel mean legitimately drifts from
 * the area-weighted figure. This band is not trying to re-run the pipeline's
 * gate; it only has to catch an atlas that carries no mask at all.
 */
const LAND_MASK_MIN_MEAN = 0.10
const LAND_MASK_MAX_MEAN = 0.60

/**
 * Throws unless an elevation atlas's G channel actually holds a land mask.
 *
 * `sweep/Matter.ts` shades land against ocean from this channel, and seven
 * of the nine elevation tiers on disk were baked before it held anything —
 * their sidecars say `G = "reserved (0)"` and they are identically zero.
 * Pointing the viewer at one is a one-word edit that renders the entire
 * planet as ocean, with nothing anywhere reporting a problem.
 *
 * Three separate defects in this atlas pipeline have now been silent
 * zeroing rather than a crash (the tau re-encode applied to every atlas, a
 * sun latitude that put every northern city in permanent day, and this), so
 * the check reads the BYTES rather than trusting the sidecar's prose — a
 * stale tier's channel documentation is exactly the thing that would be
 * wrong. The upper bound matters as much as the lower: a saturated
 * all-land mask is just as broken and just as quiet.
 */
export function assertLandMaskPopulated(faces: readonly Uint16Array[], baseUrl: string): void {
  const mean = meanLandMaskFraction(faces)
  if (mean < LAND_MASK_MIN_MEAN || mean > LAND_MASK_MAX_MEAN) {
    throw new Error(
      `${baseUrl}: elevation atlas G channel (land mask) has mean ${mean.toFixed(4)}, outside `
      + `[${LAND_MASK_MIN_MEAN}, ${LAND_MASK_MAX_MEAN}] — Earth's land fraction is 0.292. `
      + 'This tier was almost certainly baked before the mask existed (its sidecar will say '
      + 'G = "reserved (0)"), and rendering it would show a planet with no land at all. '
      + 'Re-bake it: uv run python build/bake_elevation_cube.py --face-sizes <N> --projections plain',
    )
  }
}

async function fetchJson(url: string, signal: AbortSignal): Promise<unknown> {
  const res = await fetch(url, { signal })
  signal.throwIfAborted()
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} ${res.statusText}`)
  const json: unknown = await res.json()
  signal.throwIfAborted()
  return json
}

async function fetchFace(baseUrl: string, index: number, meta: CubeMeta, signal: AbortSignal): Promise<Uint16Array> {
  const url = `${baseUrl}/face_${index}.bin`
  const res = await fetch(url, { signal })
  signal.throwIfAborted()
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} ${res.statusText}`)
  const buffer = await res.arrayBuffer()
  signal.throwIfAborted()
  validateFaceBytes(index, buffer.byteLength, meta)
  if (meta.schema === SOIL_V2) await assertFaceHash(buffer, meta.face_sha256![`face_${index}.bin`]!, url)
  signal.throwIfAborted()
  return new Uint16Array(buffer)
}

async function assertFaceHash(buffer: ArrayBuffer, expected: string, url: string): Promise<void> {
  if (!globalThis.crypto?.subtle) throw new Error(`${url}: SHA-256 verification requires WebCrypto`)
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', buffer))
  const actual = Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('')
  if (actual !== expected) throw new Error(`${url}: SHA-256 checksum mismatch`)
}

async function fetchGroupSupport(baseUrl: string, index: number, meta: GroupSupportMeta, signal: AbortSignal): Promise<Uint16Array> {
  const name = `groups_support_${index}.bin`
  const url = `${baseUrl}/${name}`
  const res = await fetch(url, { signal })
  signal.throwIfAborted()
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} ${res.statusText}`)
  const buffer = await res.arrayBuffer()
  signal.throwIfAborted()
  if (buffer.byteLength !== meta.bytes_per_face) {
    throw new Error(`${url}: expected ${meta.bytes_per_face} bytes (R16Float) but received ${buffer.byteLength}`)
  }
  await assertFaceHash(buffer, meta.sha256[name]!, url)
  signal.throwIfAborted()
  const face = new Uint16Array(buffer)
  for (let pixel = 0; pixel < face.length; pixel++) {
    const support = THREE.DataUtils.fromHalfFloat(face[pixel]!)
    if (!Number.isFinite(support) || support < 0 || support > 1) {
      throw new Error(`${url}: texel ${pixel} support must be finite and in [0,1]`)
    }
  }
  return face
}

/** Validate the integrity-checked v2 bytes before constructing any textures. */
function validateSoilFace(face: Uint16Array, support: Uint16Array, index: number): void {
  for (let i = 0; i < face.length; i += CUBE_CHANNELS) {
    const r = THREE.DataUtils.fromHalfFloat(face[i]!)
    const g = THREE.DataUtils.fromHalfFloat(face[i + 1]!)
    const b = THREE.DataUtils.fromHalfFloat(face[i + 2]!)
    const a = THREE.DataUtils.fromHalfFloat(face[i + 3]!)
    const gs = THREE.DataUtils.fromHalfFloat(support[i / CUBE_CHANNELS]!)
    const at = `face_${index}.bin: texel ${i / CUBE_CHANNELS}`
    if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b) || !Number.isFinite(a)) {
      throw new Error(`${at} RGBA channels must be finite`)
    }
    if (a < 0 || a > 1) throw new Error(`${at} A must be in [0,1]`)
    if (b < 0) throw new Error(`${at} B (bootstrap CV) must be nonnegative`)
    if (g < 0 || g > gs) throw new Error(`${at} G numerator must be in [0, own group support]`)
  }
}

/**
 * Fetches `${baseUrl}/cube_meta.json`, then its six `face_N.bin` faces and
 * (v2 only) six independent group-support faces in parallel. Builds a `THREE.CubeTexture` matching
 * `bakeCube.ts`'s output in every respect except channel count: same
 * `CubeReflectionMapping`, `ClampToEdgeWrapping`, `LinearFilter` min/mag,
 * `HalfFloatType`, `NoColorSpace`, `generateMipmaps = false` — AND the same
 * `exp(R) == tau` contract, via `densityFaceToTauFace` (see its doc): the
 * GPU texture's R channel is tau-encoded, while the `faces` returned
 * alongside it stay density-encoded for CPU-side probes.
 */
export async function loadCubeTexture(baseUrl: string): Promise<LoadedCube> {
  const controller = new AbortController()
  let timeout!: ReturnType<typeof setTimeout>
  // One deadline covers headers, bodies, validation and settlement. WebCrypto
  // cannot be cancelled; a late digest is quarantined by the signal checks,
  // so a timed-out attempt can never construct or bind a texture afterward.
  const deadline = new Promise<never>((_, reject) => {
    timeout = setTimeout(() => {
      const error = new Error(`${baseUrl}: data loading timed out after 120 seconds`)
      controller.abort(error)
      reject(error)
    }, 120_000)
  })
  try {
    return await Promise.race([loadCubeAttempt(baseUrl, controller), deadline])
  } catch (error) {
    controller.abort(error)
    throw error
  } finally {
    clearTimeout(timeout)
  }
}

async function loadCubeAttempt(baseUrl: string, controller: AbortController): Promise<LoadedCube> {
  const { signal } = controller
  const metaJson = await fetchJson(`${baseUrl}/cube_meta.json`, signal)
  const meta = parseCubeMeta(metaJson)

  const groupMeta = meta.group_support
  const jobs = Array.from({ length: CUBE_FACE_COUNT }, (_, i) => fetchFace(baseUrl, i, meta, signal))
  if (groupMeta) jobs.push(...Array.from({ length: CUBE_FACE_COUNT }, (_, i) => fetchGroupSupport(baseUrl, i, groupMeta, signal)))
  // Abort on the FIRST error, but settle every body reader/hash job before
  // allowing a retry. The outer deadline also bounds uncooperative work.
  const results = await Promise.allSettled(jobs.map((job) => job.catch((error: unknown) => {
    controller.abort(error)
    throw error
  })))
  signal.throwIfAborted()
  const values = results.map((result) => {
    if (result.status === 'rejected') throw result.reason
    return result.value
  })
  const faces = values.slice(0, CUBE_FACE_COUNT)
  const groupSupportFaces = groupMeta ? values.slice(CUBE_FACE_COUNT) : undefined
  if (groupSupportFaces) faces.forEach((face, i) => validateSoilFace(face, groupSupportFaces[i]!, i))
  // Only the DENSITY atlas gets the tau re-encode; see `needsTauEncoding`
  // for what applying it to the other two silently destroyed.
  const tauFaces = needsTauEncoding(meta) ? faces.map(densityFaceToTauFace) : faces

  // Elevation atlases carry the land mask Matter shades against; refuse one
  // that does not actually have it (see assertLandMaskPopulated).
  if (meta.field === 'elevation') assertLandMaskPopulated(faces, baseUrl)

  const faceTextures = tauFaces.map((tauData) => {
    const tex = new THREE.DataTexture(tauData, meta.face_size, meta.face_size, THREE.RGBAFormat, THREE.HalfFloatType)
    tex.needsUpdate = true
    return tex
  })

  const texture = new THREE.CubeTexture(
    faceTextures,
    THREE.CubeReflectionMapping,
    THREE.ClampToEdgeWrapping,
    THREE.ClampToEdgeWrapping,
    THREE.LinearFilter,
    THREE.LinearFilter,
    THREE.RGBAFormat,
    THREE.HalfFloatType,
    1,
    THREE.NoColorSpace,
  )
  texture.generateMipmaps = false
  texture.needsUpdate = true

  return { texture, meta, faces, ...(groupSupportFaces ? { groupSupportFaces } : {}) }
}

/**
 * Face size (texels per side) of an already-built cube texture, read off the
 * texture itself rather than passed alongside it.
 *
 * `smoothstepCubeDirection` and its TSL mirror need the texel pitch of the
 * exact atlas being sampled, and that number has to reach `regimes/Shell.ts`
 * — which receives only a `THREE.CubeTexture`, from a caller
 * (`src/viewer.ts`) this change is not allowed to touch. Deriving it from
 * the texture closes that gap without a new plumbing parameter, and, more
 * importantly, makes it impossible for the warp's assumed pitch to disagree
 * with the atlas actually bound: swapping in a different tier re-derives it.
 *
 * THROWS rather than defaulting. A wrong face size does not produce a
 * visibly wrong image — it produces a warp at the wrong spatial frequency,
 * which looks like a slightly different flavour of the same blockiness this
 * whole change exists to remove, and would be near-impossible to attribute.
 * `@types/three` types `CubeTexture.images` as `unknown[]` (the class is
 * generic with an `unknown` default), so the narrowing is explicit.
 */
export function cubeFaceSize(texture: THREE.CubeTexture): number {
  const first = (texture.images as unknown[])[0] as { image?: { width?: unknown } } | undefined
  const width = first?.image?.width
  if (typeof width !== 'number' || !Number.isInteger(width) || width <= 0) {
    throw new Error(
      'cubeFaceSize: cube texture face 0 has no integer image width — cannot derive the '
      + 'texel pitch smoothstepCubeDirection needs. Every cube this app builds '
      + '(loadCube.ts, bakeCube.ts) uses DataTexture faces, which always carry one.',
    )
  }
  return width
}

/**
 * Smoothstep-warped texel coordinate, on ONE cube-face axis.
 *
 * `c` is a face coordinate in `[-1, 1]` (the standard cube-map
 * `component / |major axis|`), and the return is another face coordinate to
 * sample AT instead. Feeding a hardware bilinear sampler the warped
 * coordinate turns its piecewise-bilinear reconstruction into a C1
 * (smoothstep-weighted) one at the cost of a few ALU, keeping the filtering
 * itself in the texture unit.
 *
 * WHY, precisely: `cube_plain_128` is 78.2 km/texel and is drawn at roughly
 * 31 screen pixels per texel, so its reconstruction filter is what the
 * viewer actually sees, not its resolution. Bilinear is C0 — its gradient
 * jumps across every texel-cell boundary — so EVERY crease in the magnified
 * field lies exactly on a texel edge, which is the polygonal faceting the
 * veil shows. Magnify the shipped `exp(R)*A` field to that scale and 100.0%
 * of its second-derivative mass sits on the texel-boundary lines under
 * bilinear. `test/cubeFilter.test.ts` reproduces that instrument on
 * `field/synthetic.ts`'s analytic stand-in for the atlas (which is what the
 * test environment can reach without loading the shipped .bin faces) and
 * measures 100.0% -> 10.8% for this warp, against a 6.1% uniform baseline
 * that a texel-free control arm confirms.
 *
 * The map in texel-centre space `p` (texel `j`'s centre is `p = j`) is
 * `p -> floor(p) + smoothstep(frac(p))`, which fixes every texel centre and
 * has zero derivative on either side of every cell boundary — so the
 * reconstructed field passes through the same texel values as before and no
 * longer creases where the cells meet.
 *
 * Rebaking finer was measured and ruled out before this: the cube is baked
 * from a 512x256 equirect source already at ~78 km/cell, and the same
 * instrument that grows monotonically on the terrain atlases saturates at
 * 5.6% on this one, so there is nothing finer in the source to recover.
 */
export function smoothstepTexelWarp(c: number, faceSize: number): number {
  const p = ((c + 1) * 0.5) * faceSize - 0.5
  const i = Math.floor(p)
  const f = p - i
  return ((i + f * f * (3 - 2 * f) + 0.5) / faceSize) * 2 - 1
}

/**
 * `smoothstepTexelWarp` applied to a whole cube-map DIRECTION, and the
 * reason this fix needs no per-face branching and cannot open a seam.
 *
 * Dividing by the largest-magnitude component gives a vector whose major
 * component is exactly +/-1 and whose other two are exactly +/- the face
 * coordinates `(u, v)` the hardware will derive — which pair, and with which
 * signs, depends on the face, and `three/src/nodes/accessors/CubeTextureNode.js`
 * negates a component again on top of that. None of that has to be
 * enumerated here, because `smoothstepTexelWarp` is ODD (`W(-c) = -W(c)`, a
 * consequence of `smoothstep(1-f) = 1 - smoothstep(f)` and the mirror
 * symmetry of the texel grid), so applying it component-wise commutes with
 * every signed permutation a cube-map convention can apply. Whatever face
 * the direction lands on, and whichever way round that face's axes run, the
 * warp that reaches `(u, v)` is the same one.
 *
 * FACE SEAMS, the failure this had to avoid. `W(+/-1) = +/-1` EXACTLY —
 * `c = 1` gives `p = N - 0.5`, whose fractional part is 0.5, and 0.5 is a
 * fixed point of smoothstep — and `|W(c)| <= 1` elsewhere. Two consequences,
 * both checked numerically in `test/cubeFilter.test.ts`:
 *
 *  1. The major axis stays the major axis (its component is +/-1 before and
 *     after), so the warp never moves a direction onto a different face.
 *  2. A direction exactly ON a seam has TWO components of magnitude 1; both
 *     are fixed, so it stays exactly on that seam, and the in-seam
 *     coordinate is warped identically as approached from either side.
 *
 * So the warp is the identity on the seam set itself: whatever the sampler
 * does where faces meet, it does bit-identically to before this change. The
 * direction is returned unnormalized — cube sampling uses direction only.
 */
export function smoothstepCubeDirection(
  x: number, y: number, z: number, faceSize: number,
): readonly [number, number, number] {
  const m = Math.max(Math.abs(x), Math.abs(y), Math.abs(z))
  if (m === 0) throw new Error('smoothstepCubeDirection: zero direction vector')
  return [
    smoothstepTexelWarp(x / m, faceSize),
    smoothstepTexelWarp(y / m, faceSize),
    smoothstepTexelWarp(z / m, faceSize),
  ]
}

export interface FaceUV {
  face: number
  u: number
  v: number
}

/**
 * Inverse of `bakeCube.ts`'s `faceDirection`, for the plain (gnomonic)
 * projection this module loads: given any-scale direction `(x,y,z)`,
 * returns which of the 6 faces it lands on and its `(u,v)` there, both in
 * `[-1,1]`. Ported from `pipeline/build/eac.py::direction_to_face_uv`'s
 * "plain" branch (whose unwarp is the identity) — so a CPU probe built on
 * this samples exactly the texel the pipeline itself wrote for that
 * direction, the same round-trip `bakeCube.test.ts` checks for the
 * synthetic bake.
 */
export function directionToFaceUV(x: number, y: number, z: number): FaceUV {
  if (x === 0 && y === 0 && z === 0) {
    throw new Error('directionToFaceUV: zero direction vector')
  }
  const ax = Math.abs(x)
  const ay = Math.abs(y)
  const az = Math.abs(z)

  if (ax >= ay && ax >= az) {
    return x > 0 ? { face: 0, u: -z / x, v: -y / x } : { face: 1, u: -z / x, v: y / x }
  }
  if (ay >= ax && ay >= az) {
    return y > 0 ? { face: 2, u: x / y, v: z / y } : { face: 3, u: -x / y, v: z / y }
  }
  return z > 0 ? { face: 4, u: x / z, v: -y / z } : { face: 5, u: x / z, v: y / z }
}

/**
 * Decodes a half-float-encoded natural-log density (R channel) back to
 * individuals per m^2 — the inverse of `bake_cube.py`'s
 * `log(max(density_per_sqm, DENSITY_FLOOR))` encode. Mirrors
 * `bakeCube.ts`'s `decodeLogTau`, but named for what this channel actually
 * holds in the real atlas.
 */
export function decodeLogDensity(halfBits: number): number {
  return Math.exp(THREE.DataUtils.fromHalfFloat(halfBits))
}

export interface CubeProbe {
  face: number
  px: number
  py: number
  /** Decoded density, individuals per m^2 (R channel). */
  density: number
  /** Density-model support 0..1 (A channel); legacy API name. */
  land: number
}

/**
 * Nearest-texel CPU-side sample of a loaded cube's decoded R (log-density)
 * and A (land-fraction) channels along `dir`. This is what lets the HUD
 * print "what does the loaded texture say is under Amsterdam" without a
 * GPU readback — see `geoProbe.ts` for the lat/lon entry point.
 */
export function sampleCubeDensity(
  faces: readonly Uint16Array[],
  meta: CubeMeta,
  dir: readonly [number, number, number],
): CubeProbe {
  const { face, u, v } = directionToFaceUV(dir[0], dir[1], dir[2])
  const faceData = faces[face]
  if (!faceData) throw new Error(`sampleCubeDensity: missing face ${face}`)

  const size = meta.face_size
  const px = Math.min(size - 1, Math.max(0, Math.floor(((u + 1) / 2) * size)))
  const py = Math.min(size - 1, Math.max(0, Math.floor(((v + 1) / 2) * size)))
  const idx = (px + py * size) * CUBE_CHANNELS
  const rBits = faceData[idx]
  const aBits = faceData[idx + 3]
  if (rBits === undefined || aBits === undefined) {
    throw new Error(`sampleCubeDensity: texel (${px},${py}) on face ${face} out of range`)
  }

  const land = THREE.DataUtils.fromHalfFloat(aBits)
  return { face, px, py, density: land > 0 ? decodeLogDensity(rBits) : 0, land }
}

/**
 * Nearest-texel CPU composition probe, matching sampleCubeDensity's lookup.
 * V2 G is a weighted numerator: divide by its OWN support, independently of A.
 * Legacy G has no declared support and is unavailable. This is not a filtered
 * GPU equivalent; filtered consumers must sample numerator/support with the
 * same directions, reconstruction and LOD before dividing.
 */
export function sampleCubeComposition(
  loaded: LoadedCube,
  dir: readonly [number, number, number],
): { fraction: number | null; support: number } {
  if (loaded.meta.schema !== SOIL_V2) return { fraction: null, support: 0 }
  const { face, u, v } = directionToFaceUV(dir[0], dir[1], dir[2])
  const size = loaded.meta.face_size
  const px = Math.min(size - 1, Math.max(0, Math.floor(((u + 1) / 2) * size)))
  const py = Math.min(size - 1, Math.max(0, Math.floor(((v + 1) / 2) * size)))
  const pixel = px + py * size
  const gBits = loaded.faces[face]?.[pixel * CUBE_CHANNELS + 1]
  const supportBits = loaded.groupSupportFaces?.[face]?.[pixel]
  if (gBits === undefined || supportBits === undefined) {
    throw new Error(`sampleCubeComposition: missing texel (${px},${py}) on face ${face}`)
  }
  const support = THREE.DataUtils.fromHalfFloat(supportBits)
  return { fraction: support > 0 ? THREE.DataUtils.fromHalfFloat(gBits) / support : null, support }
}
