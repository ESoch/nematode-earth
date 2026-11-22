import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { DataUtils } from 'three/webgpu'
import {
  parseCubeMeta, validateFaceBytes, directionToFaceUV, decodeLogDensity, sampleCubeDensity,
  densityFaceToTauFace, needsTauEncoding, meanLandMaskFraction, assertLandMaskPopulated,
  loadCubeTexture, sampleCubeComposition, CUBE_FACE_COUNT, type CubeMeta, type LoadedCube,
} from '../src/field/loadCube'
import { faceDirection } from '../src/field/bakeCube'
import { E_A } from '../src/math/optical'

// Real node built-ins through Vitest, with only the signatures this test uses.
// The browser app intentionally has no dependency on node type declarations.
const { createHash, webcrypto } = await vi.importActual<{
  createHash(algorithm: string): { update(bytes: Uint8Array): { digest(encoding: 'hex'): string } }
  webcrypto: Crypto
}>('node:crypto')
const { readFileSync } = await vi.importActual<{
  readFileSync(path: URL): Uint8Array
}>('node:fs')

const validMetaJson = (overrides: Partial<Record<string, unknown>> = {}): unknown => ({
  projection: 'plain',
  face_size: 128,
  bytes_per_face: 128 * 128 * 4 * 2,
  total_bytes: 128 * 128 * 4 * 2 * 6,
  face_order: ['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
  ...overrides,
})

describe('parseCubeMeta', () => {
  it('accepts a well-formed sidecar', () => {
    const meta = parseCubeMeta(validMetaJson())
    expect(meta.projection).toBe('plain')
    expect(meta.face_size).toBe(128)
    expect(meta.bytes_per_face).toBe(131072)
    expect(meta.total_bytes).toBe(786432)
    expect(meta.face_order).toEqual(['+X', '-X', '+Y', '-Y', '+Z', '-Z'])
  })

  it('rejects a non-object', () => {
    expect(() => parseCubeMeta(null)).toThrow(/JSON object/)
    expect(() => parseCubeMeta('nope')).toThrow(/JSON object/)
    expect(() => parseCubeMeta(42)).toThrow(/JSON object/)
  })

  it('rejects a missing or wrong-typed projection', () => {
    expect(() => parseCubeMeta(validMetaJson({ projection: undefined }))).toThrow(/projection/)
    expect(() => parseCubeMeta(validMetaJson({ projection: 7 }))).toThrow(/projection/)
    expect(() => parseCubeMeta(validMetaJson({ projection: '' }))).toThrow(/projection/)
  })

  it('rejects a non-positive-integer face_size', () => {
    expect(() => parseCubeMeta(validMetaJson({ face_size: 0 }))).toThrow(/face_size/)
    expect(() => parseCubeMeta(validMetaJson({ face_size: -128 }))).toThrow(/face_size/)
    expect(() => parseCubeMeta(validMetaJson({ face_size: 128.5 }))).toThrow(/face_size/)
    expect(() => parseCubeMeta(validMetaJson({ face_size: '128' }))).toThrow(/face_size/)
  })

  it('rejects a face_order of the wrong length or element type', () => {
    expect(() => parseCubeMeta(validMetaJson({ face_order: ['+X'] }))).toThrow(/face_order/)
    expect(() => parseCubeMeta(validMetaJson({ face_order: [1, 2, 3, 4, 5, 6] }))).toThrow(/face_order/)
  })

  it('rejects bytes_per_face inconsistent with face_size (RGBA16Float)', () => {
    // Would silently pass for an R16Float (1-channel) sidecar mistakenly pointed at this loader.
    expect(() => parseCubeMeta(validMetaJson({ bytes_per_face: 128 * 128 * 2 }))).toThrow(/bytes_per_face/)
  })

  it('rejects total_bytes inconsistent with bytes_per_face * 6', () => {
    expect(() => parseCubeMeta(validMetaJson({ total_bytes: 1 }))).toThrow(/total_bytes/)
  })
})

describe('validateFaceBytes', () => {
  const meta: CubeMeta = parseCubeMeta(validMetaJson())

  it('accepts an exact match', () => {
    expect(() => validateFaceBytes(0, meta.bytes_per_face, meta)).not.toThrow()
  })

  it('throws a clear error on a short read', () => {
    expect(() => validateFaceBytes(3, meta.bytes_per_face - 4, meta)).toThrow(/face_3\.bin/)
  })

  it('throws a clear error on an oversized read', () => {
    expect(() => validateFaceBytes(0, meta.bytes_per_face + 4, meta)).toThrow(/expected/)
  })
})

describe('directionToFaceUV', () => {
  it('is the algebraic inverse of bakeCube.ts faceDirection at face centres', () => {
    for (let face = 0; face < CUBE_FACE_COUNT; face++) {
      const [x, y, z] = faceDirection(face, 0, 0)
      const result = directionToFaceUV(x, y, z)
      expect(result.face).toBe(face)
      expect(result.u).toBeCloseTo(0, 12)
      expect(result.v).toBeCloseTo(0, 12)
    }
  })

  it('round-trips arbitrary (u,v) through faceDirection and back', () => {
    for (let face = 0; face < CUBE_FACE_COUNT; face++) {
      for (const [u, v] of [[0.3, -0.6], [-0.9, 0.9], [0.01, 0.42]] as const) {
        const [x, y, z] = faceDirection(face, u, v)
        const result = directionToFaceUV(x, y, z)
        expect(result.face).toBe(face)
        expect(result.u).toBeCloseTo(u, 10)
        expect(result.v).toBeCloseTo(v, 10)
      }
    }
  })

  it('throws on the zero vector', () => {
    expect(() => directionToFaceUV(0, 0, 0)).toThrow()
  })
})

describe('decodeLogDensity', () => {
  it('inverts bake_cube.py\'s log(max(density, floor)) encode within half-float precision', () => {
    for (const density of [1.0, 6e5, 1.34e7, 2e7]) {
      const bits = DataUtils.toHalfFloat(Math.log(density))
      const decoded = decodeLogDensity(bits)
      // Half-float has ~3 significant decimal digits; log-space quantization
      // bounds the RELATIVE error in the decoded density, same reasoning as
      // bakeCube.test.ts's round-trip check for the synthetic bake.
      const relErr = Math.abs(decoded - density) / density
      expect(relErr).toBeLessThan(0.02)
    }
  })
})

describe('sampleCubeDensity', () => {
  it.each([0, Math.log(100)])('suppresses unsupported density with log sentinel %s', (logDensity) => {
    const meta = parseCubeMeta(validMetaJson({ face_size: 1, bytes_per_face: 8, total_bytes: 48 }))
    const face = new Uint16Array([DataUtils.toHalfFloat(logDensity), 0, 0, 0])
    expect(sampleCubeDensity(Array(6).fill(face), meta, [1, 0, 0]))
      .toMatchObject({ density: 0, land: 0 })
  })

  it('does not multiply a supported all-cell density by fractional A', () => {
    const meta = parseCubeMeta(validMetaJson({ face_size: 1, bytes_per_face: 8, total_bytes: 48 }))
    const face = new Uint16Array([DataUtils.toHalfFloat(Math.log(64)), 0, 0, DataUtils.toHalfFloat(0.25)])
    const result = sampleCubeDensity(Array(6).fill(face), meta, [1, 0, 0])
    expect(result.density).toBeCloseTo(64, 0)
    expect(result.land).toBe(0.25)
  })

  it('reads the R (log-density) and A (land-fraction) channels of the nearest texel', () => {
    const size = 2
    const meta: CubeMeta = parseCubeMeta(validMetaJson({
      face_size: size,
      bytes_per_face: size * size * 4 * 2,
      total_bytes: size * size * 4 * 2 * 6,
    }))

    // One face (face 4, +Z), 2x2 texels, 4 channels each. Texel (1,1) —
    // the one +Z direction (0,0,1) should land on, since faceDirection's
    // face-centre (u=0,v=0) maps to the boundary between all 4 texels and
    // floor() picks the texel whose lower-left corner is at NDC (0,0), i.e.
    // pixel index (1,1) at size=2.
    const face4 = new Uint16Array(size * size * 4)
    const targetIdx = (1 + 1 * size) * 4
    face4[targetIdx + 0] = DataUtils.toHalfFloat(Math.log(1.34e7)) // R
    face4[targetIdx + 3] = DataUtils.toHalfFloat(1.0) // A

    const faces = [
      new Uint16Array(size * size * 4),
      new Uint16Array(size * size * 4),
      new Uint16Array(size * size * 4),
      new Uint16Array(size * size * 4),
      face4,
      new Uint16Array(size * size * 4),
    ]

    const result = sampleCubeDensity(faces, meta, [0, 0, 1])
    expect(result.face).toBe(4)
    expect(result.px).toBe(1)
    expect(result.py).toBe(1)
    expect(Math.abs(result.density - 1.34e7) / 1.34e7).toBeLessThan(0.02)
    expect(result.land).toBeCloseTo(1.0, 2)
  })

  it('throws when the direction resolves to a face missing from the faces array', () => {
    const meta: CubeMeta = parseCubeMeta(validMetaJson({ face_size: 2, bytes_per_face: 32, total_bytes: 192 }))
    expect(() => sampleCubeDensity([new Uint16Array(16)], meta, [0, 0, 1])).toThrow(/face 4/)
  })
})

describe('densityFaceToTauFace', () => {
  it('converts log(density) to log(tau = E_A * density) in R, leaving G/B/A untouched', () => {
    const size = 2
    const face = new Uint16Array(size * size * 4)
    const densities = [6e5, 1.29e6, 5.47e6, 1.06e7] // Sahara..Siberia order of magnitude
    for (let i = 0; i < densities.length; i++) {
      const density = densities[i]
      if (density === undefined) throw new Error('unreachable')
      face[i * 4 + 0] = DataUtils.toHalfFloat(Math.log(density))
      face[i * 4 + 1] = DataUtils.toHalfFloat(0)
      face[i * 4 + 2] = DataUtils.toHalfFloat(0)
      face[i * 4 + 3] = DataUtils.toHalfFloat(1.0)
    }

    const tauFace = densityFaceToTauFace(face)
    expect(tauFace.length).toBe(face.length)
    expect(tauFace).not.toBe(face) // new array, raw fetched bytes untouched

    for (let i = 0; i < densities.length; i++) {
      const density = densities[i]
      if (density === undefined) throw new Error('unreachable')
      const rBits = tauFace[i * 4 + 0]
      if (rBits === undefined) throw new Error('missing texel')
      const tau = Math.exp(DataUtils.fromHalfFloat(rBits))
      const expectedTau = E_A * density
      expect(Math.abs(tau - expectedTau) / expectedTau).toBeLessThan(0.02)
      // G/B/A pass through unchanged.
      expect(tauFace[i * 4 + 1]).toBe(face[i * 4 + 1])
      expect(tauFace[i * 4 + 2]).toBe(face[i * 4 + 2])
      expect(tauFace[i * 4 + 3]).toBe(face[i * 4 + 3])
    }
  })

  it('brings real-world density magnitudes into the synthetic tau range (sanity bound)', () => {
    // This is the actual bug this function fixes: without it, tau would be
    // ~1e6-1e7 (raw density) instead of comparable to synthetic.ts's
    // TAU_DESERT..TAU_RICH range (~0.0024..0.19) — many orders of magnitude
    // past any plausible tauCap, saturating the march uniformly everywhere.
    const face = new Uint16Array(4)
    face[0] = DataUtils.toHalfFloat(Math.log(1.06e7)) // Siberia
    const tau = Math.exp(DataUtils.fromHalfFloat(densityFaceToTauFace(face)[0] ?? 0))
    expect(tau).toBeGreaterThan(0.001)
    expect(tau).toBeLessThan(1)
  })
})

// The R-channel re-encode `densityFaceToTauFace` performs (adding ln(E_A) ==
// -18.02) is correct for the DENSITY atlas and catastrophic for every other
// one, and `loadCubeTexture` applied it to all of them unconditionally.
//
// The urban atlas carries a built-up FRACTION in R, bounded in [0,1]. Adding
// -18.02 sent every texel on Earth to about -18, and `cityLightIntensityNode`
// clamps to [0,1] before its gamma — so city lights evaluated to exactly zero
// on every fragment of the planet, in every frame, from the commit that added
// them. Nothing errored; the feature was simply multiplied out of existence.
// The elevation atlas carried the same offset as a silent -18.02 m bias.
describe('needsTauEncoding', () => {
  it('re-encodes the density atlas, which declares no field at all', () => {
    expect(needsTauEncoding(parseCubeMeta(validMetaJson()) )).toBe(true)
  })

  it('re-encodes an atlas that declares itself density explicitly', () => {
    expect(needsTauEncoding(parseCubeMeta(validMetaJson({ field: 'density' })))).toBe(true)
  })

  it('leaves elevation alone — metres are not a log density', () => {
    expect(needsTauEncoding(parseCubeMeta(validMetaJson({ field: 'elevation' })))).toBe(false)
  })

  it('leaves built-up fraction alone — this is the bug that zeroed city lights', () => {
    expect(needsTauEncoding(parseCubeMeta(validMetaJson({ field: 'built_up_fraction' })))).toBe(false)
  })

  it('throws on an unrecognised field rather than guessing an encoding', () => {
    // Silence is what let this bug live. A new atlas must fail loudly here,
    // not inherit whichever branch happens to be the default.
    expect(() => needsTauEncoding(parseCubeMeta(validMetaJson({ field: 'soil_carbon' }))))
      .toThrow(/soil_carbon/)
  })
})

describe('the encoding rule preserves each atlas as its consumers expect', () => {
  const faceWithR = (values: readonly number[]): Uint16Array => {
    const face = new Uint16Array(values.length * 4)
    values.forEach((v, i) => { face[i * 4] = DataUtils.toHalfFloat(v) })
    return face
  }

  it('keeps a built-up fraction inside [0,1], so the clamp cannot annihilate it', () => {
    const cities = [0.39844, 0.21631, 0.12915, 0] // Tokyo, Cairo, Lagos, ocean
    const face = faceWithR(cities)
    const meta = parseCubeMeta(validMetaJson({ field: 'built_up_fraction' }))
    const out = needsTauEncoding(meta) ? densityFaceToTauFace(face) : face
    cities.forEach((expected, i) => {
      const r = DataUtils.fromHalfFloat(out[i * 4]!)
      expect(r).toBeGreaterThanOrEqual(0)
      expect(r).toBeLessThanOrEqual(1)
      expect(r).toBeCloseTo(expected, 3)
    })
  })

  it('keeps elevation in metres, with no -18 m bias', () => {
    const metres = [8157, 0, -69, -10752] // Everest, datum, Death Valley, Challenger Deep
    const face = faceWithR(metres)
    const meta = parseCubeMeta(validMetaJson({ field: 'elevation' }))
    const out = needsTauEncoding(meta) ? densityFaceToTauFace(face) : face
    // Byte-identical to the source, which is a stronger and simpler claim
    // than any metre tolerance: half-float ULP at 8157 m is 4 m, so a
    // toBeCloseTo would have to be looser than the -18 m bug it guards
    // against. "Untouched" admits no tolerance argument at all.
    expect(out).toEqual(face)
  })

  it('still re-encodes density to tau, which is the one atlas that needs it', () => {
    const density = 5.47e6
    const face = faceWithR([Math.log(density)])
    const meta = parseCubeMeta(validMetaJson())
    const out = needsTauEncoding(meta) ? densityFaceToTauFace(face) : face
    // Relative, not absolute: the value is stored as a half-float LOG, so a
    // ~0.001 absolute error in the log becomes a ~0.1% relative error here.
    const tau = Math.exp(DataUtils.fromHalfFloat(out[0]!))
    expect(Math.abs(tau / (E_A * density) - 1)).toBeLessThan(0.01)
  })
})

// Matter shades land-vs-ocean from the elevation atlas's G channel, and
// seven of the nine shipped elevation tiers were baked before that channel
// held anything — they declare `G = "reserved (0)"` and are uniformly zero.
// Pointing the viewer at one is a one-word edit that renders the entire
// planet as ocean with no error anywhere. Three separate bugs this session
// were silent zeroing (the tau re-encode, the sun latitude, this); the
// sidecar's prose is not a guard, so check the bytes.
describe('meanLandMaskFraction', () => {
  const faceOfG = (values: readonly number[]): Uint16Array => {
    const face = new Uint16Array(values.length * 4)
    values.forEach((v, i) => { face[i * 4 + 1] = DataUtils.toHalfFloat(v) })
    return face
  }

  it('is 0 for a stale atlas whose G channel was never populated', () => {
    const faces = Array.from({ length: CUBE_FACE_COUNT }, () => new Uint16Array(4 * 4))
    expect(meanLandMaskFraction(faces)).toBe(0)
  })

  it('recovers the mean of a populated G channel', () => {
    // The reader strides 16 texels, so the fixture has to be at least two
    // strides long for the mean to mean anything: 32 texels, land for the
    // first 16 and ocean for the second, which puts one sampled texel in
    // each half.
    const values = Array.from({ length: 32 }, (_, i) => (i < 16 ? 1 : 0))
    const faces = Array.from({ length: CUBE_FACE_COUNT }, () => faceOfG(values))
    expect(meanLandMaskFraction(faces)).toBeCloseTo(0.5, 6)
  })

  it('samples sparsely rather than reading all 6.3M texels of a 1024 atlas', () => {
    // Documenting the stride, because a fixture shorter than it silently
    // collapses to "whatever texel 0 holds" — which is how the test above
    // was wrong the first time.
    const values = Array.from({ length: 32 }, (_, i) => (i === 0 ? 1 : 0))
    const faces = Array.from({ length: CUBE_FACE_COUNT }, () => faceOfG(values))
    expect(meanLandMaskFraction(faces)).toBeCloseTo(0.5, 6) // texels 0 and 16 only
  })

  it('reads G specifically, not R — the two carry completely different quantities', () => {
    // R holds signed metres. A reader that grabbed the wrong channel would
    // come back with something like 8157, not a fraction, and would sail
    // straight past any "is it non-zero" check.
    const face = new Uint16Array(4 * 4)
    for (let i = 0; i < 4; i++) face[i * 4] = DataUtils.toHalfFloat(8157)
    const faces = Array.from({ length: CUBE_FACE_COUNT }, () => face)
    expect(meanLandMaskFraction(faces)).toBe(0)
  })
})

describe('assertLandMaskPopulated', () => {
  const populated = (fraction: number): readonly Uint16Array[] => {
    const face = new Uint16Array(4 * 4)
    for (let i = 0; i < 4; i++) face[i * 4 + 1] = DataUtils.toHalfFloat(fraction)
    return Array.from({ length: CUBE_FACE_COUNT }, () => face)
  }

  it('accepts an atlas near Earth\'s real land fraction', () => {
    // The pipeline gate lands at 0.2925 on real data; cube faces are not
    // area-weighted, so the load-time band has to be looser than the bake's.
    expect(() => assertLandMaskPopulated(populated(0.2925), '/data/elevation_plain_1024')).not.toThrow()
  })

  it('rejects a stale all-zero G channel, naming the atlas', () => {
    const faces = Array.from({ length: CUBE_FACE_COUNT }, () => new Uint16Array(4 * 4))
    expect(() => assertLandMaskPopulated(faces, '/data/elevation_plain_512'))
      .toThrow(/elevation_plain_512/)
  })

  it('rejects an all-land G channel too — a saturated mask is as wrong as an empty one', () => {
    expect(() => assertLandMaskPopulated(populated(1), '/data/elevation_plain_512')).toThrow()
  })
})

describe('loadCubeTexture', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  /** Stubs global fetch with a minimal (1x1-texel) cube atlas so loadCubeTexture can run end-to-end without a network. */
  const stubFetch = () => {
    const faceSize = 1
    const meta = validMetaJson({
      face_size: faceSize,
      bytes_per_face: faceSize * faceSize * 4 * 2,
      total_bytes: faceSize * faceSize * 4 * 2 * CUBE_FACE_COUNT,
    })
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url.endsWith('cube_meta.json')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) } as Response)
      }
      const bytes = new Uint16Array(4) // one texel, 4 channels, all zero
      return Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(bytes.buffer) } as Response)
    }))
  }

  it('does not build or return an arrayTexture - nothing in the app reads it, and building it is a real allocation cost on the cold-load path', async () => {
    stubFetch()
    const loaded = await loadCubeTexture('/data/test_cube')
    expect(loaded).not.toHaveProperty('arrayTexture')
    // The rest of LoadedCube is untouched by removing it.
    expect(loaded.texture).toBeDefined()
    expect(loaded.faces).toHaveLength(CUBE_FACE_COUNT)
    expect(loaded.meta.face_size).toBe(1)
  })
})

// Only fetch is replaced: decoding, SHA-256, validation, texture construction,
// and CPU probes all run through the real loader. Hash fixtures independently
// with node:crypto rather than the production WebCrypto implementation.
const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex')
const halfBytes = (values: number[]): Uint8Array => {
  const bytes = new Uint8Array(values.length * 2)
  const view = new DataView(bytes.buffer)
  values.forEach((value, i) => view.setUint16(i * 2, DataUtils.toHalfFloat(value), true))
  return bytes
}
const v2Fixture = () => {
  const files: Record<string, Uint8Array> = {}
  const primaryHashes: Record<string, string> = {}
  const supportHashes: Record<string, string> = {}
  for (let face = 0; face < 6; face++) {
    // Four distinct cases, with density support deliberately unlike G support.
    const primary = halfBytes([
      Math.log(64), 0, 0.25, 0.5, // known zero composition
      Math.log(64), 0, 0.5, 1, // missing G, still valid density/CV
      Math.log(64), 0.125, 1.5, 0.125, // composition .125/.5 = .25; CV may exceed 1
      0, 0.1875, 0, 0, // composition .1875/.25 = .75 even without density
    ])
    const support = halfBytes([0.25, 0, 0.5, 0.25])
    files[`face_${face}.bin`] = primary
    files[`groups_support_${face}.bin`] = support
    primaryHashes[`face_${face}.bin`] = sha256(primary)
    supportHashes[`groups_support_${face}.bin`] = sha256(support)
  }
  const meta = {
    schema: 'soil-cube-candidate-v2', field: 'density', projection: 'plain',
    face_size: 2, bytes_per_face: 32, total_bytes: 192,
    face_order: ['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
    groups_policy: 'separate-coverage', face_sha256: primaryHashes,
    group_support: {
      format: 'R16Float', bytes_per_face: 8, total_bytes: 48,
      face_order: ['+X', '-X', '+Y', '-Y', '+Z', '-Z'], sha256: supportHashes,
    },
  }
  return { meta, files }
}
type V2Fixture = ReturnType<typeof v2Fixture>

function serveFixture({ meta, files }: V2Fixture): void {
  vi.stubGlobal('fetch', async (url: string) => {
    const name = url.slice(url.lastIndexOf('/') + 1)
    if (name === 'cube_meta.json') return Response.json(meta)
    const bytes = files[name]
    return bytes ? new Response(new Uint8Array(bytes).buffer) : new Response(null, { status: 404 })
  })
}

function alterHalf(fixture: V2Fixture, name: string, offset: number, bits: number, rehash = true): void {
  const bytes = fixture.files[name]!
  new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).setUint16(offset * 2, bits, true)
  if (rehash) {
    const hashes = name.startsWith('groups_') ? fixture.meta.group_support.sha256 : fixture.meta.face_sha256
    hashes[name] = sha256(bytes)
  }
}

function disposeLoaded(loaded: LoadedCube): void {
  loaded.texture.images.forEach((face) => face.dispose())
  loaded.texture.dispose()
}

describe('v2 soil bundle metadata', () => {
  it('retains independent support and primary hashes', () => {
    const { meta } = v2Fixture()
    expect(parseCubeMeta(meta)).toMatchObject({
      schema: 'soil-cube-candidate-v2', groups_policy: 'separate-coverage',
      group_support: meta.group_support, face_sha256: meta.face_sha256,
    })
  })

  it.each([
    ['unknown schema', { schema: 'soil-cube-candidate-v3' }, /schema/],
    ['missing schema with a v2 descriptor', { schema: undefined }, /schema/],
    ['wrong schema with a v2 descriptor', { schema: 'soil-cube-candidate-v1' }, /schema/],
    ['wrong policy', { groups_policy: 'require-complete' }, /groups_policy/],
    ['missing policy', { groups_policy: undefined }, /groups_policy/],
    ['wrong projection', { projection: 'eac' }, /projection/],
    ['wrong field', { field: 'elevation' }, /field/],
    ['missing field', { field: undefined }, /field/],
    ['wrong face order', { face_order: ['-X', '+X', '+Y', '-Y', '+Z', '-Z'] }, /face_order/],
    ['missing support', { group_support: undefined }, /group_support/],
    ['null support', { group_support: null }, /group_support/],
    ['array support', { group_support: [] }, /group_support/],
    ['missing primary checksums', { face_sha256: undefined }, /face_sha256/],
  ] as const)('rejects %s', (_name, override, error) => {
    expect(() => parseCubeMeta({ ...v2Fixture().meta, ...override })).toThrow(error)
  })

  it.each([
    ['format', { format: 'RGBA16Float' }, /format/],
    ['face bytes', { bytes_per_face: 32 }, /bytes_per_face/],
    ['total bytes', { total_bytes: 8 }, /total_bytes/],
    ['order', { face_order: ['-X', '+X', '+Y', '-Y', '+Z', '-Z'] }, /face_order/],
    ['missing checksums', { sha256: undefined }, /sha256/],
  ] as const)('rejects malformed support %s', (_name, override, error) => {
    const { meta } = v2Fixture()
    expect(() => parseCubeMeta({ ...meta, group_support: { ...meta.group_support, ...override } })).toThrow(error)
  })

  it.each(['primary', 'support'] as const)('requires exactly six canonical %s checksum names and valid digests', (kind) => {
    const { meta } = v2Fixture()
    const hashes = kind === 'primary' ? meta.face_sha256 : meta.group_support.sha256
    const name = kind === 'primary' ? 'face_5.bin' : 'groups_support_5.bin'
    const validHash = hashes[name]!
    delete hashes[name]
    expect(() => parseCubeMeta(meta)).toThrow(/sha256/)
    hashes[`../${name}`] = validHash
    expect(() => parseCubeMeta(meta)).toThrow(/sha256/)
    delete hashes[`../${name}`]
    hashes[name] = 'not-a-sha256'
    expect(() => parseCubeMeta(meta)).toThrow(/sha256/)
    hashes[name] = validHash
    hashes['unexpected.bin'] = validHash
    expect(() => parseCubeMeta(meta)).toThrow(/sha256/)
  })

  it('still accepts legacy metadata without independent support', () => {
    expect(parseCubeMeta(validMetaJson())).not.toHaveProperty('group_support')
    expect(parseCubeMeta(validMetaJson({ schema: 'soil-cube-candidate-v1' }))).not.toHaveProperty('group_support')
  })
})

describe('v2 soil loading and composition', () => {
  beforeEach(() => {
    // Real WebCrypto, for node environments that do not expose it globally.
    if (!globalThis.crypto?.subtle) vi.stubGlobal('crypto', webcrypto)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('loads missing G without losing density/CV and keeps G as a numerator on the GPU', async () => {
    serveFixture(v2Fixture())
    const loaded = await loadCubeTexture('/candidate')
    try {
      expect(loaded.groupSupportFaces).toHaveLength(6)
      const probe = sampleCubeDensity(loaded.faces, loaded.meta, [1, 0.5, -0.5])
      expect(probe.density).toBeCloseTo(64, 0)
      expect(probe.land).toBe(1)
      expect(DataUtils.fromHalfFloat(loaded.faces[0]![6]!)).toBe(0.5)
      const gpu = loaded.texture.images[0]!.image.data as Uint16Array
      expect(Math.exp(DataUtils.fromHalfFloat(gpu[4]!)) / E_A).toBeCloseTo(64, 0)
      expect(DataUtils.fromHalfFloat(gpu[9]!)).toBe(0.125)
      expect(DataUtils.fromHalfFloat(gpu[10]!)).toBe(1.5)
    } finally { disposeLoaded(loaded) }
  })

  it.each([
    ['known zero', [1, 0.5, 0.5], 0, 0.25],
    ['unavailable', [1, 0.5, -0.5], null, 0],
    ['weighted numerator', [1, -0.5, 0.5], 0.25, 0.5],
    ['composition without density', [1, -0.5, -0.5], 0.75, 0.25],
    // -Z uses u=x/z, v=y/z: this direction gives u=-.5, v=.5 (texel 0,1).
    ['last face same registration', [0.5, -0.5, -1], 0.25, 0.5],
  ] as const)('samples %s using independent support', async (_name, dir, fraction, support) => {
    serveFixture(v2Fixture())
    const loaded = await loadCubeTexture('/candidate')
    try {
      expect(sampleCubeComposition).toBeTypeOf('function')
      expect(sampleCubeComposition(loaded, dir)).toEqual({ fraction, support })
    } finally { disposeLoaded(loaded) }
  })

  it('reports legacy composition unavailable even with positive G and A', async () => {
    const fixture = v2Fixture()
    const legacy = validMetaJson({ face_size: 2, bytes_per_face: 32, total_bytes: 192 })
    vi.stubGlobal('fetch', async (url: string) => url.endsWith('cube_meta.json')
      ? Response.json(legacy) : new Response(new Uint8Array(fixture.files['face_0.bin']!).buffer))
    const loaded = await loadCubeTexture('/legacy')
    try {
      expect(loaded.groupSupportFaces).toBeUndefined()
      expect(sampleCubeComposition).toBeTypeOf('function')
      expect(sampleCubeComposition(loaded, [1, -0.5, 0.5])).toEqual({ fraction: null, support: 0 })
    } finally { disposeLoaded(loaded) }
  })

  it.each(['face_5.bin', 'groups_support_5.bin'])('rejects checksum corruption in %s', async (name) => {
    const fixture = v2Fixture()
    alterHalf(fixture, name, 0, DataUtils.toHalfFloat(0.125), false)
    serveFixture(fixture)
    await expect(loadCubeTexture('/candidate')).rejects.toThrow(new RegExp(`${name}.*SHA-256`))
  })

  it.each(['face_5.bin', 'groups_support_5.bin'])('rejects truncated %s before decoding', async (name) => {
    const fixture = v2Fixture()
    fixture.files[name] = fixture.files[name]!.slice(1)
    serveFixture(fixture)
    await expect(loadCubeTexture('/candidate')).rejects.toThrow(new RegExp(`${name}.*expected`))
  })

  it('rejects missing support files rather than treating the bundle as legacy', async () => {
    const fixture = v2Fixture()
    delete fixture.files['groups_support_3.bin']
    serveFixture(fixture)
    await expect(loadCubeTexture('/candidate')).rejects.toThrow(/groups_support_3.bin.*404/)
  })

  it.each([
    ['R NaN', 'face_4.bin', 0, 0x7e00],
    ['G infinity', 'face_4.bin', 1, 0x7c00],
    ['B NaN', 'face_4.bin', 2, 0x7e00],
    ['A infinity', 'face_4.bin', 3, 0x7c00],
    ['negative G', 'face_4.bin', 1, 0xb800],
    ['negative B', 'face_4.bin', 2, 0xb800],
    ['negative A', 'face_4.bin', 3, 0xb800],
    ['A greater than one', 'face_4.bin', 3, 0x4000],
    ['G greater than own support', 'face_4.bin', 1, 0x3800],
    ['G positive without support', 'face_4.bin', 5, 0x3400],
    ['support NaN', 'groups_support_4.bin', 0, 0x7e00],
    ['support negative', 'groups_support_4.bin', 0, 0xb800],
    ['support greater than one', 'groups_support_4.bin', 0, 0x4000],
  ] as const)('rejects %s even with matching hashes', async (_name, name, offset, bits) => {
    const fixture = v2Fixture()
    alterHalf(fixture, name, offset, bits)
    serveFixture(fixture)
    await expect(loadCubeTexture('/candidate')).rejects.toThrow(/face_4.bin|groups_support_4.bin/)
  })

  const candidateDir = new URL('../public/data/soil_plain_512_v2/', import.meta.url)
  it('loads the shipped 512 bundle through real hash/byte/channel validation', async () => {
    vi.stubGlobal('fetch', async (url: string) => {
      const name = url.slice(url.lastIndexOf('/') + 1)
      if (!/^(cube_meta\.json|face_[0-5]\.bin|groups_support_[0-5]\.bin)$/.test(name)) {
        return new Response(null, { status: 404 })
      }
      return new Response(new Uint8Array(readFileSync(new URL(name, candidateDir))).buffer)
    })
    const loaded = await loadCubeTexture('/candidate')
    try {
      expect(loaded.meta.face_size).toBe(512)
      expect(loaded.groupSupportFaces).toHaveLength(6)
      let missingGroupsWithDensity = 0
      loaded.faces.forEach((face, f) => {
        for (let pixel = 0; pixel < 512 * 512; pixel++) {
          if (DataUtils.fromHalfFloat(face[pixel * 4 + 3]!) > 0
            && DataUtils.fromHalfFloat(loaded.groupSupportFaces![f]![pixel]!) === 0) missingGroupsWithDensity++
        }
      })
      expect(missingGroupsWithDensity).toBe(3631)
    } finally { disposeLoaded(loaded) }
  })
})
