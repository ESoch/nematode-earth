import { describe, it, expect } from 'vitest'
import { isWebGPUAvailable } from '../src/platform/webgpu'

describe('isWebGPUAvailable', () => {
  it('returns false when navigator has no gpu', () => {
    expect(isWebGPUAvailable({})).toBe(false)
  })

  it('returns false when gpu is undefined', () => {
    expect(isWebGPUAvailable({ gpu: undefined })).toBe(false)
  })

  it('returns true when gpu is present', () => {
    expect(isWebGPUAvailable({ gpu: {} })).toBe(true)
  })

  it('defaults to globalThis.navigator when no argument given', () => {
    // The vitest environment's navigator (or absence of one) has no `gpu`,
    // so this must resolve to false, not merely avoid throwing.
    expect(() => isWebGPUAvailable()).not.toThrow()
    expect(isWebGPUAvailable()).toBe(false)
  })
})
