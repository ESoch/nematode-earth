import { scaleHeightAt, renderedTau } from '../math/descent'
import { WGS84 } from '../math/wgs84'
import { sigmaPixAt } from '../math/lod'

/**
 * The LOD partition (spec 7.7), derived from Nelson Max, IEEE TVCG 1995 eq. (4).
 * Cite it as "derived from", never as "Max states": Max's derivation assumes
 * identical spheres.
 *
 *   j_inst = w*g,  j_vol = (1-w)*g
 *
 * Energy-conserving IN EXPECTATION at any optical depth, because the RTE at
 * fixed tau is linear in the source term - NEVER "exactly conserving": the
 * per-pixel Poisson error is 1/sqrt(N_eff), 13.5% at the shipped budget
 * (math/lod.ts's effectiveSamples/noiseFraction). Two load-bearing
 * conditions, both required of whatever consumes this partition (not
 * enforced by this file, which is pure arithmetic with no rendering in it):
 *   1. every instance is multiplied by transmittance from the SAME sigma_t
 *      field the shell marches;
 *   2. the medium is emission-only (no scattering term exists anywhere).
 *
 * THE CAP COUPLING. Spec 7.3 caps the shell's rendered depth at
 * tau_r = min(tau_data, 4/Q) so the rim never blows, and says "the disc fading
 * as you descend is correct - the LOD partition is handing that energy to the
 * regime-2 instances". That only balances if the instance share absorbs the
 * capped-away fraction, so the shares here are
 *
 *   shell = (1-w) * (tau_r/tau_data)
 *   inst  = 1 - shell
 *
 * which reduces to the plain w-partition whenever the cap is inactive and
 * conserves exactly BY CONSTRUCTION otherwise (shell+inst=1 is an identity
 * of this formula, not a physical claim - see the "conserves energy by
 * construction" test, which is really testing arithmetic, not physics).
 *
 * THE BAND IS DERIVED, NOT CHOSEN. The cap first bites the global-mean tau at
 * Hs = 2*pi*R*(tau/4)^2 = 5526.7 m, i.e. alt = 6*Hs = 33.16 km. Regime 2
 * therefore opens there rather than at a hand-picked altitude, and completes a
 * decade lower. ABOVE the band the capped energy is dropped rather than
 * re-homed - deliberate, and already licensed by spec 7.4 ("the limb is an
 * authoring decision, not physics"). The largest loss is the richest pixels at
 * orbit: 1 - 0.1131/0.19 = 40.5% of their energy. Raising REGIME2_TOP_ALT is
 * the lever if art direction rejects that; the instance system is
 * scale-invariant in cost, so raising it is affordable.
 *
 * SPEC TENSION, surfaced rather than silently resolved: spec 7.3 says the
 * partition hands the capped-away energy to the instances, but the cap
 * starts biting the global-mean tau at 33.16 km while spec 7.7 places the
 * handoff at 10 km. Between 33.16 km and the 10 km figure, the capped
 * fraction is dropped, not re-homed - 45.1% of the global-mean energy at
 * 10 km. This file derives REGIME2_TOP_ALT from the cap so the two agree,
 * at the cost of running regime 2 three times higher than spec 7.7 states.
 * If the 10 km figure is authoritative instead, that deficit is real and
 * needs an explicit decision, not a silent fix here.
 */
const TAU_MEAN = 0.047

export const REGIME2_TOP_ALT = 6 * (2 * Math.PI * WGS84.R_MEAN * Math.pow(TAU_MEAN / 4, 2))
export const REGIME2_FULL_ALT = REGIME2_TOP_ALT / 10

const clamp01 = (v: number): number => Math.min(Math.max(v, 0), 1)
const smooth = (u: number): number => u * u * (3 - 2 * u)

/**
 * Every potentially visible particle must lie in both the camera's emission
 * window and the shell. The equatorial radius encloses the oblate shell at
 * every latitude, so this test can safely skip disjoint volumes.
 */
export function particleWindowIntersectsShell(cameraRadius: number, windowRadius: number, shellHeight: number): boolean {
  return cameraRadius - windowRadius <= WGS84.a + shellHeight
}

/** Handoff ramp, log-linear in altitude across the derived band. */
export function handoffWeight(altM: number): number {
  if (altM >= REGIME2_TOP_ALT) return 0
  if (altM <= REGIME2_FULL_ALT) return 1
  const u = (Math.log10(altM) - Math.log10(REGIME2_FULL_ALT)) /
    (Math.log10(REGIME2_TOP_ALT) - Math.log10(REGIME2_FULL_ALT))
  return 1 - smooth(clamp01(u))
}

export function shellShare(w: number, tauRendered: number, tauData: number): number {
  const r = tauData > 0 ? clamp01(tauRendered / tauData) : 1
  return (1 - w) * r
}

export function instanceShare(w: number, tauRendered: number, tauData: number): number {
  return 1 - shellShare(w, tauRendered, tauData)
}

/**
 * Radial taper on the instance emission window. Instances only exist within a
 * bounded window around the camera; without a taper its boundary would be a
 * visible sphere. Whatever draws the shell march must multiply its own sigma
 * by 1 - taper*w at the same position for the partition to stay exact across
 * the boundary - not done by this file, which only computes the taper value.
 */
export function windowTaper(dist: number, rIn: number, rOut: number): number {
  if (dist <= rIn) return 1
  if (dist >= rOut) return 0
  return 1 - smooth((dist - rIn) / (rOut - rIn))
}

export interface PartitionState {
  alt: number
  Hs: number
  w: number
  r: number
  shell: number
  inst: number
  sigmaPix: number
  capBites: boolean
}

export function partitionAt(altM: number, tauData: number): PartitionState {
  const Hs = scaleHeightAt(altM)
  const tauR = renderedTau(tauData, Hs)
  const w = handoffWeight(altM)
  const r = tauData > 0 ? clamp01(tauR / tauData) : 1
  const shell = shellShare(w, tauR, tauData)
  return {
    alt: altM, Hs, w, r, shell, inst: 1 - shell,
    sigmaPix: sigmaPixAt(altM), capBites: tauR < tauData * (1 - 1e-12),
  }
}
