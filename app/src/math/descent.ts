import { limbRatio, saturationCapTau } from './optical'

export const H_S_MAX = 32000
/** f32 conditioning of Rout^2 - Rin^2 gives 19% chord error at H=1 m. */
export const H_S_MIN = 200
export const K_SHIP = 3.0
export const K_MAX = 3.6
export const EVEREST_M = 8849

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

/** Hs(alt) = clamp(alt/6, 200 m, 32 km). Reaches full shell at 192 km. */
export function scaleHeightAt(altM: number): number {
  return clamp(altM / 6, H_S_MIN, H_S_MAX)
}

/**
 * Exaggerated Everest must not exceed one e-folding of the shell, so
 * k <= Hs / 8849 m. At Hs = 32 km that is 3.616, hence K_MAX = 3.6.
 *
 * 10x is rejected: it puts Everest 2.8 e-folds OUTSIDE the shell, so the
 * mountain reads as taller than the film it is supposed to sit in.
 * Spec section 7.5.
 */
export function exaggerationAt(altM: number): number {
  return clamp(scaleHeightAt(altM) / EVEREST_M, 1.0, K_MAX)
}

/** Cap rendered nadir depth so the rim never blows. Spec section 7.3. */
export function renderedTau(tauData: number, Hs: number): number {
  return Math.min(tauData, saturationCapTau(Hs))
}

export interface DescentState {
  alt: number
  Hs: number
  Hmax: number
  k: number
  tauRendered: number
  tauLimb: number
  Q: number
}

export function descentAt(altM: number, tauData: number): DescentState {
  const Hs = scaleHeightAt(altM)
  const tauRendered = renderedTau(tauData, Hs)
  const Q = limbRatio(Hs)
  return { alt: altM, Hs, Hmax: 6 * Hs, k: exaggerationAt(altM), tauRendered, tauLimb: tauRendered * Q, Q }
}
