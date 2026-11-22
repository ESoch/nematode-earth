import type { SweepState } from './Timeline'
import type { Vec3 } from '../math/raySpheroid'

/**
 * Spatially propagating erosion, not a crossfade (spec 6.1).
 *
 * Each layer erodes where a scalar field phi(dir) in [0,1] falls behind a
 * travelling front. phi is a geometric sweep from a seed direction blended
 * with layer-specific noise, so the boundary is a ragged line crossing the
 * globe rather than a global opacity ramp - the difference is the entire
 * point of the beat: the eye reads sequence as causality, and a uniform fade
 * reads as a slideshow transition instead.
 *
 * This CPU implementation is the source of truth for the front algebra and
 * is exactly what `test/dissolve.test.ts` exercises. The TSL mirror
 * (`dissolveNodes.ts`) reproduces `layerAlpha` exactly - it is three
 * operations - but uses a different, GPU-defined noise for the phi texture
 * (a sin-hash is not reproducible across GPU vendors). Nothing compares the
 * two noises pointwise; with noise disabled, phi is analytic on both sides
 * and agrees exactly there.
 */
export type LayerName = 'atmosphere' | 'water' | 'rock'

export interface LayerParams {
  /** Half-width of the front in phi units. Wide = boils off, narrow = shatters. */
  width: number
  noiseFreq: number
  noiseWeight: number
  octaves: number
  /** Ridged noise gives hard-edged crumbling patches instead of a soft fade. */
  ridged: boolean
}

export const LAYERS: Record<LayerName, LayerParams> = {
  atmosphere: { width: 0.35, noiseFreq: 0.5, noiseWeight: 0.25, octaves: 3, ridged: false },
  water: { width: 0.12, noiseFreq: 4.0, noiseWeight: 0.45, octaves: 4, ridged: false },
  rock: { width: 0.05, noiseFreq: 9.0, noiseWeight: 0.6, octaves: 5, ridged: true },
}

/** Front seed: the point the erosion starts from. Unit vector. */
export const DEFAULT_SEED: Vec3 = [0.6, 0.48, 0.64]

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0))
  return t * t * (3 - 2 * t)
}

/**
 * Matter coverage at one point. 1 = matter still there, 0 = eroded.
 *
 * The front centre travels from -2w to 1+2w as frontNorm goes 0 -> 1, and the
 * transition band is [f-w, f+w]. At frontNorm = 0 the band is [-3w, -w],
 * entirely below phi = 0, so alpha is exactly 1 everywhere including phi = 0.
 * At frontNorm = 1 it is [1+w, 1+3w], entirely above phi = 1, so alpha is
 * exactly 0 everywhere including phi = 1. Both endpoints are exact, not
 * asymptotic: a residual 1/255 of undissolved atmosphere at t=13 would sit on
 * screen for the whole of the what-remains beat.
 */
export function layerAlpha(phi: number, frontNorm: number, width: number): number {
  const f = -2 * width + clamp01(frontNorm) * (1 + 4 * width)
  return smoothstep(f - width, f + width, phi)
}

function hash3(x: number, y: number, z: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
  return s - Math.floor(s)
}

function smootherstep01(x: number): number {
  const t = clamp01(x)
  return t * t * t * (t * (t * 6 - 15) + 10)
}

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t

/**
 * Trilinearly-interpolated value noise on an integer lattice, corners hashed
 * via `hash3`. A raw `hash3(x,y,z)` sample (as used by `field/synthetic.ts`'s
 * fbm) is essentially WHITE noise - its huge internal coefficients (127.1,
 * 311.7, 74.7) make it change wildly for any input delta, so its output is
 * uncorrelated between neighbouring points regardless of a caller's `freq`
 * scale. That is fine for synthetic.ts's spatial texture, but it makes
 * `noiseFreq` inert here: measured, atmosphere (freq 1.5) and rock (freq 9)
 * came out statistically indistinguishable in roughness. Interpolating
 * between lattice corners instead gives noise that is actually smooth WITHIN
 * a cell and only varies between cells, so `freq` (which sets the cell size)
 * meaningfully controls how rough the field reads.
 */
function valueNoise3(x: number, y: number, z: number): number {
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const z0 = Math.floor(z)
  const u = smootherstep01(x - x0)
  const v = smootherstep01(y - y0)
  const w = smootherstep01(z - z0)
  const c = (dx: number, dy: number, dz: number): number => hash3(x0 + dx, y0 + dy, z0 + dz)
  const x00 = lerp(c(0, 0, 0), c(1, 0, 0), u)
  const x10 = lerp(c(0, 1, 0), c(1, 1, 0), u)
  const x01 = lerp(c(0, 0, 1), c(1, 0, 1), u)
  const x11 = lerp(c(0, 1, 1), c(1, 1, 1), u)
  const y0v = lerp(x00, x10, v)
  const y1v = lerp(x01, x11, v)
  return lerp(y0v, y1v, w)
}

function fbm(d: Vec3, freq: number, octaves: number, ridged: boolean): number {
  let sum = 0
  let amp = 0.5
  let f = freq
  let norm = 0
  for (let i = 0; i < octaves; i++) {
    const n = valueNoise3(d[0] * f, d[1] * f, d[2] * f)
    sum += amp * (ridged ? 1 - Math.abs(2 * n - 1) : n)
    norm += amp
    f *= 2.07
    amp *= 0.5
  }
  return norm > 0 ? sum / norm : 0
}

/**
 * phi in [0,1]. The geometric term alone has an exact and useful property:
 * the set { phi < f } is a spherical cap of solid-angle fraction exactly f,
 * because (1 - cos)/2 is the cap area fraction - this file's own test suite
 * uses that closed form as ground truth for the noise-disabled case.
 */
export function erosionPhi(dir: Vec3, p: LayerParams, seed: Vec3, noiseEnabled = true): number {
  const n = Math.hypot(dir[0], dir[1], dir[2]) || 1
  const u: Vec3 = [dir[0] / n, dir[1] / n, dir[2] / n]
  const geo = (1 - (u[0] * seed[0] + u[1] * seed[1] + u[2] * seed[2])) / 2
  if (!noiseEnabled || p.noiseWeight === 0) return clamp01(geo)
  const noise = fbm(u, p.noiseFreq, p.octaves, p.ridged)
  return clamp01((1 - p.noiseWeight) * geo + p.noiseWeight * noise)
}

export function frontFor(layer: LayerName, state: SweepState): number {
  return layer === 'atmosphere' ? state.frontAtmosphere
    : layer === 'water' ? state.frontWater
      : state.frontRock
}

export function matterAlphaAt(
  dir: Vec3,
  layer: LayerName,
  state: SweepState,
  seed: Vec3 = DEFAULT_SEED,
  noiseEnabled = true,
): number {
  const p = LAYERS[layer]
  return layerAlpha(erosionPhi(dir, p, seed, noiseEnabled), frontFor(layer, state), p.width)
}
