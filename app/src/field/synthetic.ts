import { arealCoverage } from '../math/optical'

export const TAU_DESERT = 0.0024
export const TAU_MEAN = 0.047
export const TAU_TUNDRA = 0.12
export const TAU_RICH = 0.19

/** Cheap deterministic value noise so the field has sub-continental texture. */
function hash3(x: number, y: number, z: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
  return s - Math.floor(s)
}

function fbm(x: number, y: number, z: number, octaves = 4): number {
  let sum = 0
  let amp = 0.5
  let freq = 1.0
  for (let i = 0; i < octaves; i++) {
    sum += amp * hash3(x * freq, y * freq, z * freq)
    freq *= 2.07
    amp *= 0.5
  }
  return sum // roughly [0, 1)
}

/**
 * Analytic stand-in for the baked atlas.
 *
 * Reproduces the structure that matters for Session 1: a strong latitudinal
 * gradient with the POLES BRIGHTEST (sub-Arctic holds 38% of global abundance
 * against 21% tropical), plus enough spatial noise to exercise mip selection.
 * The absolute values are anchored to the spec's biome table so the rendered
 * contrast is representative rather than arbitrary.
 */
export function syntheticTau(dirX: number, dirY: number, dirZ: number): number {
  const n = Math.hypot(dirX, dirY, dirZ) || 1
  const x = dirX / n
  const y = dirY / n
  const z = dirZ / n

  const absLat = Math.abs(Math.asin(Math.max(-1, Math.min(1, z)))) / (Math.PI / 2) // 0..1

  // Latitudinal envelope: dim tropics, bright sub-Arctic, slight polar rolloff.
  const subArctic = Math.exp(-Math.pow((absLat - 0.78) / 0.28, 2))
  const tropicalFloor = 0.05 + 0.18 * Math.exp(-Math.pow(absLat / 0.20, 2))
  const envelope = Math.max(tropicalFloor, subArctic)

  // Continental-scale variation, then finer texture.
  const continents = fbm(x * 2.3, y * 2.3, z * 2.3, 3)
  const detail = fbm(x * 11.0, y * 11.0, z * 11.0, 4)
  const modulation = 0.55 * continents + 0.45 * detail

  const t = envelope * (0.12 + 0.88 * modulation)
  return TAU_DESERT + (TAU_RICH - TAU_DESERT) * Math.max(0, Math.min(1, t))
}

export function syntheticStats(samples = 20000): {
  min: number; max: number; mean: number; coverageRatio: number
} {
  let min = Infinity
  let max = -Infinity
  let sum = 0
  // Deterministic Fibonacci-sphere sampling: no RNG, reproducible in CI.
  const ga = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < samples; i++) {
    const z = 1 - (2 * i) / (samples - 1)
    const r = Math.sqrt(Math.max(0, 1 - z * z))
    const th = ga * i
    const t = syntheticTau(Math.cos(th) * r, Math.sin(th) * r, z)
    min = Math.min(min, t)
    max = Math.max(max, t)
    sum += t
  }
  return {
    min, max, mean: sum / samples,
    coverageRatio: arealCoverage(max) / arealCoverage(min),
  }
}
