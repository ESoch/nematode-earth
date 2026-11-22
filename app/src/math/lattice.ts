import { WGS84 } from './wgs84'

/**
 * The world-anchored cell lattice's deterministic hash.
 *
 * Instance positions are hashed from the INTEGER cell index in an ECEF-
 * aligned lattice, never from anything camera-relative, so an instance is a
 * pure function of where it is in the world: it cannot swim between frames,
 * and it cannot swim when the camera moves. The cell size is octave-
 * quantized (math/lod.ts's `octaveSplit`) so the index itself is stable
 * while the level holds - a continuously varying cell size would move every
 * instance every frame, which is exactly the swimming a field-driven
 * particle system has to avoid to read as solid rather than as noise.
 *
 * Only the hash itself is built here (`pcgHashU32`/`cellHash`/`hashUnit`)
 * plus the i32-index bias/range constants it depends on. The candidate-
 * enumeration (`cellBoxFor`) and incremental-altitude reconstruction that
 * the full compute-shader emission kernel needs are GPU-execution-dependent
 * infrastructure this session does not attempt to build blind - see
 * sweep-report.md's session-4 report for why.
 */

/**
 * Indices are carried biased by 2^30 so every reachable cell index (positive
 * or negative offset from the ECEF origin) lands in the non-negative range,
 * which is what lets the hash below run entirely in unsigned, wrap-defined
 * 32-bit arithmetic - exactly what a WGSL `u32` mirror needs to match
 * bit-for-bit. The reachable extent is the outer shell radius a + H_max =
 * 6.570e6 m, so the finest safe cell at this bias is 6.570e6 / 2^30 =
 * 6.12 mm - finer than the ~4 cm cell at which cross-section aggregation
 * approaches single-animal scale (math/crossSection.ts's omega -> 1), i.e.
 * the lattice outlives the whole of regime 2 and hands over near the 5 m
 * boundary of spec 7.8, where a different anchoring is needed anyway.
 */
export const CELL_BIAS = 1 << 30
export const MIN_SAFE_CELL_SIZE = (WGS84.a + 192000) / CELL_BIAS

/** PCG output-mixing hash, the same construction three.js's TSL `hash` uses. */
export function pcgHashU32(x: number): number {
  const state = (Math.imul(x >>> 0, 747796405) + 2891336453) >>> 0
  const shifted = ((state >>> 28) + 4) >>> 0
  const word = Math.imul((state >>> shifted) ^ state, 277803737) >>> 0
  return ((word >>> 22) ^ word) >>> 0
}

/**
 * Three-axis mix, then avalanche. `cx`/`cy`/`cz` are the RAW (possibly
 * negative) integer cell indices along each axis - biased into the
 * non-negative range internally - and `salt` distinguishes independent
 * draws at the same cell (jitter x/y/z, cross-section u1/u2, ...) so they
 * don't correlate.
 */
export function cellHash(cx: number, cy: number, cz: number, salt: number): number {
  const x = (cx + CELL_BIAS) >>> 0
  const y = (cy + CELL_BIAS) >>> 0
  const z = (cz + CELL_BIAS) >>> 0
  let h = Math.imul(x, 0x27220a95) >>> 0
  h = pcgHashU32((h ^ Math.imul(y, 0x165667b1)) >>> 0)
  h = pcgHashU32((h ^ Math.imul(z, 0x9e3779b1)) >>> 0)
  return pcgHashU32((h + Math.imul(salt >>> 0, 0x85ebca6b)) >>> 0)
}

/** Maps a hash's full uint32 range onto [0,1). */
export function hashUnit(h: number): number {
  return (h >>> 0) / 4294967296
}
