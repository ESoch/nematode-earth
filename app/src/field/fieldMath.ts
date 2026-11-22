import { TAU_DESERT, TAU_RICH } from './synthetic'

/**
 * Pure-JS mirror of the hash/fbm/envelope math in `synthetic.ts`, used as an
 * intermediate, pointwise-testable stepping stone for the TSL port in
 * `syntheticTSL.ts`.
 *
 * `synthetic.ts` is the tested CPU reference and per Session 2's brief is not
 * modified — its `hash3`/`fbm` are private, unexported functions, so true
 * code-sharing ("both implementations call the same function") is not
 * possible without editing it. This module is therefore a deliberate
 * duplicate, not a shared dependency: `test/fieldMath.test.ts` asserts it
 * reproduces `syntheticTau`'s output exactly (same formula, same operation
 * order), and `syntheticTauNode` in `syntheticTSL.ts` mirrors this file
 * term-for-term so the duplication stays auditable in one place rather than
 * silently drifting between three copies.
 */
export function hash3(x: number, y: number, z: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
  return s - Math.floor(s)
}

export function fbm(x: number, y: number, z: number, octaves = 4): number {
  let sum = 0
  let amp = 0.5
  let freq = 1.0
  for (let i = 0; i < octaves; i++) {
    sum += amp * hash3(x * freq, y * freq, z * freq)
    freq *= 2.07
    amp *= 0.5
  }
  return sum
}

/** Pure-JS mirror of `syntheticTau` — see module doc for why this duplicates rather than imports. */
export function syntheticTauPure(dirX: number, dirY: number, dirZ: number): number {
  const n = Math.hypot(dirX, dirY, dirZ) || 1
  const x = dirX / n
  const y = dirY / n
  const z = dirZ / n

  const absLat = Math.abs(Math.asin(Math.max(-1, Math.min(1, z)))) / (Math.PI / 2)

  const subArctic = Math.exp(-Math.pow((absLat - 0.78) / 0.28, 2))
  const tropicalFloor = 0.05 + 0.18 * Math.exp(-Math.pow(absLat / 0.20, 2))
  const envelope = Math.max(tropicalFloor, subArctic)

  const continents = fbm(x * 2.3, y * 2.3, z * 2.3, 3)
  const detail = fbm(x * 11.0, y * 11.0, z * 11.0, 4)
  const modulation = 0.55 * continents + 0.45 * detail

  const t = envelope * (0.12 + 0.88 * modulation)
  return TAU_DESERT + (TAU_RICH - TAU_DESERT) * Math.max(0, Math.min(1, t))
}
