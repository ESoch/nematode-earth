import { afterEach, expect, it, vi } from 'vitest'
import { loadCubeTexture } from '../src/field/loadCube'

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })

const meta = {
  projection: 'plain', face_size: 1, bytes_per_face: 8, total_bytes: 48,
  face_order: ['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
}

it('settles outstanding face downloads before exposing a failed atlas for retry', async () => {
  let releaseFaces!: () => void
  const slowFaces = new Promise<void>((resolve) => { releaseFaces = resolve })
  let pendingFaces = 0
  let pendingAtFailure: number | undefined
  vi.stubGlobal('fetch', async (url: string) => {
    if (url.endsWith('cube_meta.json')) return Response.json({
      projection: 'plain', face_size: 1, bytes_per_face: 8, total_bytes: 48,
      face_order: ['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
    })
    if (url.endsWith('face_0.bin')) return new Response(null, { status: 503 })
    pendingFaces++
    await slowFaces
    pendingFaces--
    return new Response(new Uint16Array(4).buffer)
  })
  const attempt = loadCubeTexture('/recovery-fixture').catch(() => {
    pendingAtFailure = pendingFaces
  })
  // Let the failed response and all its promise continuations settle while
  // five other real loader requests are deliberately still pending.
  await new Promise((resolve) => setTimeout(resolve, 0))
  releaseFaces()
  await attempt
  expect(pendingAtFailure).toBe(0)
})

it('aborts and settles sibling response bodies after face validation fails', async () => {
  const signals: AbortSignal[] = []
  const bodies: ReadableStreamDefaultController<Uint8Array>[] = []
  let cancelled = 0
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    if (init?.signal) signals.push(init.signal)
    if (url.endsWith('cube_meta.json')) return Response.json(meta)
    if (url.endsWith('face_0.bin')) return new Response(new Uint8Array(2))
    return new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        bodies.push(controller)
        init?.signal?.addEventListener('abort', () => {
          // Settlement must include the body reader, not just fetch headers.
          queueMicrotask(() => { cancelled++; controller.error(init.signal!.reason) })
        }, { once: true })
      },
    }))
  })
  try {
    await expect(loadCubeTexture('/bad-face')).rejects.toThrow(/face_0.*expected/)
    expect(cancelled).toBe(5)
    expect(signals).toHaveLength(7)
    expect(signals.every((signal) => signal.aborted)).toBe(true)
  } finally {
    for (const body of bodies) body.error(new Error('test cleanup'))
  }
})

it.each(['headers', 'metadata body', 'face body'] as const)(
  'bounds a stalled %s and prevents late work after timeout', async (stage) => {
    vi.useFakeTimers()
    let release!: () => void
    const stall = new Promise<void>((resolve) => { release = resolve })
    let signal: AbortSignal | undefined
    let faceRequests = 0
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      signal = init?.signal ?? undefined
      if (url.endsWith('cube_meta.json')) {
        if (stage === 'headers') await stall
        if (stage === 'metadata body') return { ok: true, json: async () => { await stall; return meta } }
        return Response.json(meta)
      }
      faceRequests++
      return { ok: true, arrayBuffer: async () => {
        if (stage === 'face body') await stall
        return new Uint16Array(4).buffer
      } }
    })
    let failure: unknown
    const attempt = loadCubeTexture('/stalled').then((loaded) => {
      loaded.texture.images.forEach((face) => face.dispose())
      loaded.texture.dispose()
    }, (error: unknown) => { failure = error })
    try {
      await vi.advanceTimersByTimeAsync(120_000)
      expect(failure).toBeInstanceOf(Error)
      expect((failure as Error).message).toMatch(/timed out/i)
      expect(signal?.aborted).toBe(true)
    } finally {
      release()
      await attempt
      await vi.advanceTimersByTimeAsync(0)
    }
    expect(faceRequests).toBe(stage === 'face body' ? 6 : 0)
    expect(vi.getTimerCount()).toBe(0)
  },
)

it.each(['primary', 'support'] as const)('settles all v2 hashes after a %s checksum fails', async (kind) => {
  const fixture = holdV2Hashes()
  let outcome = 'pending'
  const attempt = loadCubeTexture('/v2').then(() => { outcome = 'ready' }, (error: Error) => {
    outcome = error.message
  })
  await fixture.started
  const badIndex = kind === 'primary' ? 0 : 6
  fixture.digests[badIndex]!(new Uint8Array(32).fill(1).buffer)
  await new Promise((resolve) => setTimeout(resolve, 0))
  try {
    expect(fixture.signals.every((signal) => signal.aborted)).toBe(true)
    expect(outcome).toBe('pending') // Hashing cannot be aborted: drain it.
  } finally {
    fixture.digests.forEach((resolve) => resolve(new Uint8Array(32).buffer))
    await attempt
  }
  expect(outcome).toMatch(/checksum mismatch/)
})

it('bounds uninterruptible v2 hashing and clears the attempt deadline', async () => {
  vi.useFakeTimers()
  const fixture = holdV2Hashes()
  let outcome = 'pending'
  const attempt = loadCubeTexture('/v2').then(() => { outcome = 'ready' }, (error: Error) => {
    outcome = error.message
  })
  await fixture.started
  try {
    await vi.advanceTimersByTimeAsync(120_000)
    expect(outcome).toMatch(/timed out/i)
    expect(fixture.signals.every((signal) => signal.aborted)).toBe(true)
  } finally {
    fixture.digests.forEach((resolve) => resolve(new Uint8Array(32).buffer))
    await attempt
    await vi.advanceTimersByTimeAsync(0)
  }
  expect(outcome).toMatch(/timed out/i)
  expect(vi.getTimerCount()).toBe(0)
})

function holdV2Hashes() {
  const digests: ((value: ArrayBuffer) => void)[] = []
  const signals: AbortSignal[] = []
  let allStarted!: () => void
  const started = new Promise<void>((resolve) => { allStarted = resolve })
  const hashes = (prefix: string) => Object.fromEntries(
    Array.from({ length: 6 }, (_, i) => [`${prefix}_${i}.bin`, '0'.repeat(64)]),
  )
  vi.stubGlobal('crypto', { subtle: { digest: () => new Promise<ArrayBuffer>((resolve) => {
    digests.push(resolve)
    if (digests.length === 12) allStarted()
  }) } })
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    if (init?.signal) signals.push(init.signal)
    if (url.endsWith('cube_meta.json')) return Response.json({
      ...meta, schema: 'soil-cube-candidate-v2', field: 'density', groups_policy: 'separate-coverage',
      face_sha256: hashes('face'),
      group_support: {
        format: 'R16Float', bytes_per_face: 2, total_bytes: 12,
        face_order: meta.face_order, sha256: hashes('groups_support'),
      },
    })
    return new Response(new Uint16Array(url.includes('groups_support') ? 1 : 4).buffer)
  })
  return { digests, signals, started }
}
