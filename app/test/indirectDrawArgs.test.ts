import { describe, it, expect } from 'vitest'
import {
  INDIRECT_INDEXED_ARGS_WORDS, encodeIndexedIndirectArgs, decodeIndexedIndirectArgs,
  finalizeIndexedIndirectArgs,
} from '../src/regime2/indirectDrawArgs'

describe('encode/decode round-trip', () => {
  it('is exact for ordinary values', () => {
    const args = { indexCount: 6, instanceCount: 12345, firstIndex: 0, baseVertex: 0, firstInstance: 0 }
    const buf = encodeIndexedIndirectArgs(args)
    expect(buf.length).toBe(INDIRECT_INDEXED_ARGS_WORDS)
    expect(decodeIndexedIndirectArgs(buf)).toEqual(args)
  })
  it('round-trips a negative baseVertex through two-s-complement', () => {
    const args = { indexCount: 6, instanceCount: 0, firstIndex: 0, baseVertex: -1, firstInstance: 0 }
    const buf = encodeIndexedIndirectArgs(args)
    expect(buf[3]).toBe(0xffffffff) // -1 as an unsigned 32-bit word
    expect(decodeIndexedIndirectArgs(buf).baseVertex).toBe(-1)
  })
  it('rejects a buffer of the wrong length - the 4-word non-indexed layout, for instance', () => {
    expect(() => decodeIndexedIndirectArgs(new Uint32Array(4))).toThrow()
  })
})

describe('the 5-word indexed layout is not the 4-word non-indexed layout', () => {
  it('writing 4-word drawIndirect-style values [count, instances, first, firstInstance] into 5 words would land firstInstance in baseVertex', () => {
    // Simulates exactly the bug the doc warns about: taking the CANONICAL
    // non-indexed 4-word args (vertexCount, instanceCount, firstVertex,
    // firstInstance) and writing them into a 5-word slot without adding the
    // missing baseVertex word. word[3], which SHOULD be baseVertex, ends up
    // holding whatever firstInstance was meant to be instead - and the real
    // firstInstance (word[4]) is left as whatever the buffer was
    // initialized to, not the caller's intended value.
    const wrongFourWord = new Uint32Array([6, 100, 0, 7]) // "firstInstance=7" meant for word[3]
    const misread = decodeIndexedIndirectArgs(
      Uint32Array.from([...wrongFourWord, 0]), // padded to 5 words, as a real buffer allocation would be
    )
    expect(misread.baseVertex).toBe(7) // the intended firstInstance, landed in the wrong field
    expect(misread.firstInstance).toBe(0) // NOT 7 - silently lost
  })
})

describe('firstInstance must be 0', () => {
  it('throws rather than silently accepting a nonzero firstInstance', () => {
    expect(() => encodeIndexedIndirectArgs({
      indexCount: 6, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 1,
    })).toThrow(/firstInstance/)
  })
})

describe('word-type validation', () => {
  it('rejects a non-integer or out-of-range unsigned word', () => {
    expect(() => encodeIndexedIndirectArgs({
      indexCount: -1, instanceCount: 0, firstIndex: 0, baseVertex: 0, firstInstance: 0,
    })).toThrow()
    expect(() => encodeIndexedIndirectArgs({
      indexCount: 1.5, instanceCount: 0, firstIndex: 0, baseVertex: 0, firstInstance: 0,
    })).toThrow()
  })
  it('rejects an out-of-range signed baseVertex', () => {
    expect(() => encodeIndexedIndirectArgs({
      indexCount: 6, instanceCount: 0, firstIndex: 0, baseVertex: 2 ** 31, firstInstance: 0,
    })).toThrow()
  })
})

describe('finalizeIndexedIndirectArgs', () => {
  it('always writes firstInstance and baseVertex as 0', () => {
    const a = finalizeIndexedIndirectArgs(6, 500, 4e6)
    expect(a.firstInstance).toBe(0)
    expect(a.baseVertex).toBe(0)
    expect(a.firstIndex).toBe(0)
  })
  it('clamps instanceCount to the budget - never reads past the records buffer', () => {
    const a = finalizeIndexedIndirectArgs(6, 5e6, 4e6)
    expect(a.instanceCount).toBe(4e6)
  })
  it('never goes negative even if the emitted count somehow underflows', () => {
    const a = finalizeIndexedIndirectArgs(6, -3, 4e6)
    expect(a.instanceCount).toBe(0)
  })
  it('passes through an in-budget count unchanged', () => {
    const a = finalizeIndexedIndirectArgs(6, 12345, 4e6)
    expect(a.instanceCount).toBe(12345)
  })
  it('produces args that encode without throwing (firstInstance=0 by construction)', () => {
    expect(() => encodeIndexedIndirectArgs(finalizeIndexedIndirectArgs(6, 999, 4e6))).not.toThrow()
  })
})
