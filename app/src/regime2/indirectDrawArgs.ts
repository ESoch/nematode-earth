/**
 * WebGPU indexed-indirect-draw argument buffer layout.
 *
 * `drawIndexedIndirect` reads exactly 5 x 32-bit words from the indirect
 * buffer:
 *
 *   [indexCount, instanceCount, firstIndex, baseVertex(SIGNED), firstInstance]
 *
 * This is NOT three.js's own canonical indirect-draw example, which is
 * built around the NON-indexed `drawIndirect` layout (4 words:
 * [vertexCount, instanceCount, firstVertex, firstInstance]). Copying that
 * example's values onto INDEXED geometry silently mis-draws: word index 3
 * (`firstInstance` in the 4-word scheme) lands in `baseVertex`'s slot in the
 * 5-word scheme instead, and there is no word 4 to hold the real
 * `firstInstance` at all - the draw does not error, it just draws the wrong
 * thing, or nothing. `encodeIndexedIndirectArgs`/`decodeIndexedIndirectArgs`
 * exist specifically so this layout is built and read in exactly one place,
 * tested, rather than open-coded as a raw `Uint32Array` at each call site.
 *
 * Regime 2's actual GPU-side wiring (the compute kernel that fills this
 * buffer, the storage-buffer usage-flag ordering, the one-frame compute-
 * after-render latency) is not attempted in this module or anywhere else
 * this session - see the session-4 report for why. What is here is the
 * part of "indirect draw" that is pure arithmetic and can be verified
 * without a GPU: getting this specific byte layout right, and refusing to
 * construct the one input (`firstInstance != 0`) that fails silently rather
 * than loudly.
 */

export const INDIRECT_INDEXED_ARGS_WORDS = 5

export interface IndexedIndirectDrawArgs {
  indexCount: number
  instanceCount: number
  firstIndex: number
  /** The ONLY signed word in the layout - added to every vertex index before indexing into the vertex buffers. */
  baseVertex: number
  /**
   * MUST be 0 unless the adapter has the optional `indirect-first-instance`
   * feature: nonzero without it makes the ENTIRE draw a silent no-op, not
   * an error - invisible on a machine that happens to have the feature,
   * blank on one that doesn't. `finalizeIndexedIndirectArgs` below always
   * writes 0; survivors must be compacted so the live range starts there.
   */
  firstInstance: number
}

function assertU32(name: string, v: number): void {
  if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) {
    throw new Error(`indexed indirect draw: ${name} must be a uint32, got ${v}`)
  }
}

function assertI32(name: string, v: number): void {
  if (!Number.isInteger(v) || v < -0x80000000 || v > 0x7fffffff) {
    throw new Error(`indexed indirect draw: ${name} must be an int32, got ${v}`)
  }
}

/** Builds the 5-word buffer. Throws rather than silently accepting a nonzero `firstInstance`. */
export function encodeIndexedIndirectArgs(args: IndexedIndirectDrawArgs): Uint32Array {
  assertU32('indexCount', args.indexCount)
  assertU32('instanceCount', args.instanceCount)
  assertU32('firstIndex', args.firstIndex)
  assertI32('baseVertex', args.baseVertex)
  assertU32('firstInstance', args.firstInstance)
  if (args.firstInstance !== 0) {
    throw new Error(
      `indexed indirect draw: firstInstance must be 0 (compact survivors so the live range starts at 0), got ${args.firstInstance} - `
      + 'nonzero silently no-ops the ENTIRE draw on any adapter without the indirect-first-instance feature',
    )
  }
  const buf = new Uint32Array(INDIRECT_INDEXED_ARGS_WORDS)
  buf[0] = args.indexCount >>> 0
  buf[1] = args.instanceCount >>> 0
  buf[2] = args.firstIndex >>> 0
  buf[3] = args.baseVertex >>> 0 // two's-complement reinterpretation of the signed value into the u32 slot
  buf[4] = args.firstInstance >>> 0
  return buf
}

/** Reads the 5-word buffer back, reinterpreting word 3 as signed. */
export function decodeIndexedIndirectArgs(buf: Uint32Array): IndexedIndirectDrawArgs {
  if (buf.length !== INDIRECT_INDEXED_ARGS_WORDS) {
    throw new Error(`indexed indirect draw: expected ${INDIRECT_INDEXED_ARGS_WORDS} words, got ${buf.length}`)
  }
  const indexCount = buf[0]
  const instanceCount = buf[1]
  const firstIndex = buf[2]
  const baseVertexWord = buf[3]
  const firstInstance = buf[4]
  if (indexCount === undefined || instanceCount === undefined || firstIndex === undefined
    || baseVertexWord === undefined || firstInstance === undefined) {
    throw new Error('indexed indirect draw: missing word')
  }
  return { indexCount, instanceCount, firstIndex, baseVertex: baseVertexWord | 0, firstInstance }
}

/**
 * Finalizes one frame's args from the emit kernel's atomic emitted-count,
 * clamped to the record buffer's capacity (`budget`) so an over-budget
 * frame draws exactly `budget` instances - never more, which would read
 * past the end of the records buffer - and never a stale prior-frame count.
 * `firstInstance` and `baseVertex` are always 0: geometry indices and
 * instance indices both start at the base of their respective buffers,
 * with compaction (not offsetting) handling which instances survived.
 */
export function finalizeIndexedIndirectArgs(
  indexCount: number,
  emittedCount: number,
  budget: number,
): IndexedIndirectDrawArgs {
  return {
    indexCount,
    instanceCount: Math.min(Math.max(Math.trunc(emittedCount), 0), budget),
    firstIndex: 0,
    baseVertex: 0,
    firstInstance: 0,
  }
}
