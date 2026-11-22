import { afterEach, describe, expect, it, vi } from 'vitest'
import { createViewerPolicy } from '../src/core/viewerPolicy'

afterEach(() => vi.unstubAllEnvs())

describe.each([false, true])('viewer development mode = %s', (development) => {
  it('gates diagnostic shortcuts, including shifted letters, using the build environment', () => {
    vi.stubEnv('DEV', development)
    const policy = createViewerPolicy()
    for (const key of ['d', 'r', 'v', 'p', 't', 'i']) {
      expect(policy.allowsKey(key), key).toBe(development)
      expect(policy.allowsKey(key.toUpperCase()), key.toUpperCase()).toBe(development)
    }
    for (const key of ['s', 'S', 'u', 'U', '[', ']', '?', '/', ' ']) {
      expect(policy.allowsKey(key), key).toBe(true)
    }
  })

  it('offers only usable shortcuts and source-switching hints', () => {
    vi.stubEnv('DEV', development)
    const policy = createViewerPolicy()
    expect(policy.keyBindings.map(({ key }) => key).sort()).toEqual(
      development
        ? [' ', '[', ']', '?', 'd', 'i', 'p', 'r', 's', 't', 'u', 'v'].sort()
        : [' ', '[', ']', '?', 's', 'u'].sort(),
    )
    for (const source of ['real', 'synthetic'] as const) {
      expect(policy.sourceHint(source).includes('press d')).toBe(development)
    }
    expect(policy.syntheticCaption.includes('Press d')).toBe(development)
  })

  it('publishes the visual-check handle only in development, without invoking it', () => {
    vi.stubEnv('DEV', development)
    const policy = createViewerPolicy()
    const target = { unrelated: 42 } as { unrelated: number; __nematode?: object }
    let poseReads = 0
    const handle = { pose: () => { poseReads++; return 'measured pose' } }
    policy.exposeDiagnostics(target, handle)
    expect(Object.hasOwn(target, '__nematode')).toBe(development)
    expect(target.unrelated).toBe(42)
    expect(poseReads).toBe(0)
    if (development) {
      expect(target.__nematode).toBe(handle)
      expect((target.__nematode as typeof handle).pose()).toBe('measured pose')
    }
  })
})
