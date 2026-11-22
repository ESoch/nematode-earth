import { describe, it, expect, vi, afterEach } from 'vitest'
import * as THREE from 'three/webgpu'
import { WGS84 } from '../src/math/wgs84'
import {
  parseSamplesMeta, decodeSamples, samplePosition, surfaceNormal, isPointVisible,
  isAntarcticBiome, buildSamplePoints, loadSamplePoints,
  SAMPLE_RECORD_BYTES, ANTARCTIC_BIOME_INDEX,
  type SamplesMeta, type DecodedSample,
} from '../src/regimes/SamplePoints'

const validMetaJson = (overrides: Partial<Record<string, unknown>> = {}): unknown => ({
  n_samples: 2,
  record_bytes: 13,
  total_bytes: 26,
  field_order: ['latitude_deg', 'longitude_deg', 'total_per_100g_dry_soil', 'biome_index'],
  ...overrides,
})

describe('parseSamplesMeta', () => {
  it('accepts a well-formed sidecar', () => {
    const meta = parseSamplesMeta(validMetaJson())
    expect(meta.n_samples).toBe(2)
    expect(meta.record_bytes).toBe(13)
    expect(meta.total_bytes).toBe(26)
  })

  it('rejects a non-object', () => {
    expect(() => parseSamplesMeta(null)).toThrow(/JSON object/)
    expect(() => parseSamplesMeta('nope')).toThrow(/JSON object/)
  })

  it('rejects a missing or wrong-typed n_samples', () => {
    expect(() => parseSamplesMeta(validMetaJson({ n_samples: undefined }))).toThrow(/n_samples/)
    expect(() => parseSamplesMeta(validMetaJson({ n_samples: -1 }))).toThrow(/n_samples/)
    expect(() => parseSamplesMeta(validMetaJson({ n_samples: 2.5 }))).toThrow(/n_samples/)
  })

  it('rejects a record_bytes other than the 13-byte layout this decoder implements', () => {
    // A future re-bake that changes the record layout must fail loudly here
    // rather than have decodeSamples silently mis-stride every field.
    expect(() => parseSamplesMeta(validMetaJson({ record_bytes: 17, total_bytes: 34 })))
      .toThrow(/record_bytes/)
  })

  it('rejects total_bytes inconsistent with record_bytes * n_samples', () => {
    expect(() => parseSamplesMeta(validMetaJson({ total_bytes: 999 }))).toThrow(/total_bytes/)
  })

  it('rejects a field_order that does not match the 4-field layout this decoder assumes', () => {
    expect(() => parseSamplesMeta(validMetaJson({ field_order: ['latitude_deg'] })))
      .toThrow(/field_order/)
  })
})

describe('decodeSamples', () => {
  const meta: SamplesMeta = parseSamplesMeta(validMetaJson())

  const encode = (records: readonly [number, number, number, number][]): ArrayBuffer => {
    const buf = new ArrayBuffer(records.length * SAMPLE_RECORD_BYTES)
    const view = new DataView(buf)
    records.forEach(([lat, lon, total, biome], i) => {
      const base = i * SAMPLE_RECORD_BYTES
      view.setFloat32(base + 0, lat, true)
      view.setFloat32(base + 4, lon, true)
      view.setFloat32(base + 8, total, true)
      view.setUint8(base + 12, biome)
    })
    return buf
  }

  it('decodes lat/lon/total/biome for every record, little-endian, at the documented offsets', () => {
    const buf = encode([
      [53.09239939, 6.626211158, 3124.0, 4],
      [-70.5, 100.0, 500.0, 25],
    ])
    const samples = decodeSamples(buf, meta)
    expect(samples).toHaveLength(2)
    expect(samples[0]?.latDeg).toBeCloseTo(53.09239939, 4)
    expect(samples[0]?.lonDeg).toBeCloseTo(6.626211158, 4)
    expect(samples[0]?.totalPer100g).toBeCloseTo(3124.0, 1)
    expect(samples[0]?.biomeIndex).toBe(4)
    expect(samples[1]?.biomeIndex).toBe(25)
  })

  it('throws on a short buffer rather than silently mis-striding', () => {
    const buf = encode([[0, 0, 0, 0], [0, 0, 0, 0]]).slice(0, 20)
    expect(() => decodeSamples(buf, meta)).toThrow(/samples\.bin/)
  })

  it('throws on an oversized buffer too', () => {
    const short = parseSamplesMeta(validMetaJson({ n_samples: 1, total_bytes: 13 }))
    const buf = encode([[0, 0, 0, 0], [0, 0, 0, 0]])
    expect(() => decodeSamples(buf, short)).toThrow(/expected/)
  })
})

describe('samplePosition', () => {
  it.each([[45, 30], [-52.1, -123.5]])('preserves geodetic coordinates at %s°, %s°', (lat, lon) => {
    const [x, y, z] = samplePosition(lat, lon)
    // Geodetic latitude is the elevation angle of the WGS 84 surface normal.
    const normalZ = z / (WGS84.b ** 2)
    const normalHorizontal = Math.hypot(x, y) / (WGS84.a ** 2)
    expect(Math.atan2(normalZ, normalHorizontal) * 180 / Math.PI).toBeCloseTo(lat, 9)
    expect(Math.atan2(y, x) * 180 / Math.PI).toBeCloseTo(lon, 9)
  })

  it('lands on the true WGS84 ellipsoid (x^2/a^2 + y^2/a^2 + z^2/b^2 = 1), matching Shell.ts\'s own surface', () => {
    for (const [lat, lon] of [[0, 0], [52.1, 5.3], [-66.5, 100], [89, -179], [-3, -60]] as const) {
      const [x, y, z] = samplePosition(lat, lon)
      const residual = (x * x) / (WGS84.a * WGS84.a) + (y * y) / (WGS84.a * WGS84.a) + (z * z) / (WGS84.b * WGS84.b)
      expect(residual).toBeCloseTo(1, 9)
    }
  })

  it('places the equator/prime-meridian point at (a, 0, 0)', () => {
    const [x, y, z] = samplePosition(0, 0)
    expect(x).toBeCloseTo(WGS84.a, 3)
    expect(y).toBeCloseTo(0, 3)
    expect(z).toBeCloseTo(0, 3)
  })

  it('places the north pole at (0, 0, b) -- the true polar radius, not a', () => {
    const [x, y, z] = samplePosition(90, 37)
    expect(x).toBeCloseTo(0, 3)
    expect(y).toBeCloseTo(0, 3)
    expect(z).toBeCloseTo(WGS84.b, 3)
  })
})

describe('surfaceNormal', () => {
  it('is a unit vector everywhere', () => {
    for (const [lat, lon] of [[0, 0], [45, 30], [-80, 170]] as const) {
      const n = surfaceNormal(samplePosition(lat, lon))
      expect(Math.hypot(n[0], n[1], n[2])).toBeCloseTo(1, 9)
    }
  })

  it('points straight along +X at the equator/prime-meridian point', () => {
    const n = surfaceNormal(samplePosition(0, 0))
    expect(n[0]).toBeCloseTo(1, 9)
    expect(n[1]).toBeCloseTo(0, 9)
    expect(n[2]).toBeCloseTo(0, 9)
  })

  it('points straight along +Z at the north pole', () => {
    const n = surfaceNormal(samplePosition(90, 0))
    expect(n[0]).toBeCloseTo(0, 6)
    expect(n[1]).toBeCloseTo(0, 6)
    expect(n[2]).toBeCloseTo(1, 9)
  })
})

describe('isPointVisible', () => {
  // A camera 1000 km up, directly above (0,0) -- the sub-camera / nadir point.
  const ALT = 1_000_000
  const camWorld: [number, number, number] = [WGS84.a + ALT, 0, 0]
  // Horizon half-angle from nadir for this altitude: arccos(a/(a+alt)).
  const horizonDeg = (Math.acos(WGS84.a / (WGS84.a + ALT)) * 180) / Math.PI

  it('sees the point directly beneath the camera', () => {
    expect(isPointVisible(samplePosition(0, 0), camWorld)).toBe(true)
  })

  it('does not see the antipodal point on the far side of the planet', () => {
    expect(isPointVisible(samplePosition(0, 180), camWorld)).toBe(false)
  })

  it('sees a point well within the horizon cap', () => {
    const insideDeg = horizonDeg * 0.5
    expect(isPointVisible(samplePosition(0, insideDeg), camWorld)).toBe(true)
  })

  it('does not see a point well beyond the horizon cap', () => {
    const outsideDeg = Math.min(90, horizonDeg * 3)
    expect(isPointVisible(samplePosition(0, outsideDeg), camWorld)).toBe(false)
  })

  it('flips sign exactly at the horizon, matching the analytic tangent angle', () => {
    expect(isPointVisible(samplePosition(0, horizonDeg - 0.5), camWorld)).toBe(true)
    expect(isPointVisible(samplePosition(0, horizonDeg + 0.5), camWorld)).toBe(false)
  })
})

describe('isAntarcticBiome', () => {
  it('is true only for the Antarctica-override biome code', () => {
    expect(isAntarcticBiome(ANTARCTIC_BIOME_INDEX)).toBe(true)
    expect(ANTARCTIC_BIOME_INDEX).toBe(25)
    expect(isAntarcticBiome(4)).toBe(false)
    expect(isAntarcticBiome(0)).toBe(false)
    expect(isAntarcticBiome(98)).toBe(false)
  })
})

describe('buildSamplePoints', () => {
  const ALT = 1_000_000
  const camWorld: [number, number, number] = [WGS84.a + ALT, 0, 0]

  const samples: DecodedSample[] = [
    { latDeg: 0, lonDeg: 0, totalPer100g: 100, biomeIndex: 4 }, // nadir: visible, temperate biome
    { latDeg: 0, lonDeg: 180, totalPer100g: 200, biomeIndex: 11 }, // antipodal: not visible
    { latDeg: 0, lonDeg: 5, totalPer100g: 300, biomeIndex: 25 }, // near nadir: visible, Antarctic biome
  ]

  it('counts Antarctic and other biome codes', () => {
    const layer = buildSamplePoints(samples)
    expect(layer.nSamples).toBe(3)
    expect(layer.nAntarctic).toBe(1)
    expect(layer.nOther).toBe(2)
    layer.dispose()
  })

  it('exposes a THREE.Points ready to add to a scene', () => {
    const layer = buildSamplePoints(samples)
    expect(layer.points).toBeInstanceOf(THREE.Points)
    expect(layer.points.geometry.getAttribute('position')).toBeDefined()
    expect(layer.points.geometry.getAttribute('color')).toBeDefined()
    layer.dispose()
  })

  it('update() draws only the visible points, compacted to the front of the buffer', () => {
    const layer = buildSamplePoints(samples)
    layer.update(camWorld)

    // Sample 1 (antipodal) is culled; samples 0 and 2 survive, in original order.
    expect(layer.points.geometry.drawRange.count).toBe(2)

    const pos = layer.points.geometry.getAttribute('position').array as Float32Array
    const world0 = samplePosition(0, 0)
    const world2 = samplePosition(0, 5)
    expect(pos[0]).toBeCloseTo(world0[0] - camWorld[0], 1)
    expect(pos[1]).toBeCloseTo(world0[1] - camWorld[1], 1)
    expect(pos[2]).toBeCloseTo(world0[2] - camWorld[2], 1)
    expect(pos[3]).toBeCloseTo(world2[0] - camWorld[0], 1)
    expect(pos[4]).toBeCloseTo(world2[1] - camWorld[1], 1)
    expect(pos[5]).toBeCloseTo(world2[2] - camWorld[2], 1)
    layer.dispose()
  })

  it('update() re-culls on every call as the camera moves, not just once', () => {
    const layer = buildSamplePoints(samples)
    layer.update(camWorld)
    expect(layer.points.geometry.drawRange.count).toBe(2)

    // Move the camera to directly above the antipodal point instead.
    const camWorld2: [number, number, number] = [-(WGS84.a + ALT), 0, 0]
    layer.update(camWorld2)
    expect(layer.points.geometry.drawRange.count).toBe(1) // only the antipodal sample now
    layer.dispose()
  })

  it('colours points by Antarctic or other biome code', () => {
    const layer = buildSamplePoints(samples, { otherColor: 0xff0000, antarcticColor: 0x00ff00 })
    layer.update(camWorld)
    const col = layer.points.geometry.getAttribute('color').array as Float32Array
    // slot 0 = sample 0 (biome 4, red), slot 1 = sample 2 (biome 25, green)
    expect(col[0]).toBeCloseTo(1, 5)
    expect(col[1]).toBeCloseTo(0, 5)
    expect(col[3]).toBeCloseTo(0, 5)
    expect(col[4]).toBeCloseTo(1, 5)
    layer.dispose()
  })

  it('does not scale point size or count by total_per_100g -- sampling density, not abundance', () => {
    // Deliberate design choice (see module docstring): the material has a
    // single scalar size for every point regardless of totalPer100g, and
    // buildSamplePoints never reads totalPer100g at all except to pass it
    // through DecodedSample -- confirm no size attribute exists.
    const layer = buildSamplePoints(samples)
    expect(layer.points.geometry.getAttribute('size')).toBeUndefined()
    const material = layer.points.material as THREE.PointsMaterial
    expect(typeof material.size).toBe('number')
    layer.dispose()
  })
})

describe('loadSamplePoints', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('fetches the sidecar + binary and builds a layer end to end', async () => {
    const meta = { n_samples: 1, record_bytes: 13, total_bytes: 13, field_order: ['latitude_deg', 'longitude_deg', 'total_per_100g_dry_soil', 'biome_index'] }
    const buf = new ArrayBuffer(13)
    const view = new DataView(buf)
    view.setFloat32(0, 52.1, true)
    view.setFloat32(4, 5.3, true)
    view.setFloat32(8, 4200, true)
    view.setUint8(12, 4)

    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url.endsWith('samples_meta.json')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(meta) } as Response)
      }
      return Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(buf) } as Response)
    }))

    const layer = await loadSamplePoints('/data/nematode_samples_v2')
    expect(layer.nSamples).toBe(1)
    expect(layer.nOther).toBe(1)
    layer.dispose()
  })
})
