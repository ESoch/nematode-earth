import { exp, float, log, max } from 'three/tsl'
import type { CubeTexture, Node } from 'three/webgpu'
import { soilFilterEncoding } from './loadCube'

type FloatNode = Node<'float'>

/** R is already density per TOTAL cell area, with missing source area counted
 * as zero contribution. A records valid-model area, not a second multiplier
 * and not an ocean mask. It only suppresses the log(0) sentinel when empty.
 */
export function supportedAreaMean(areaMean: number, validFraction: number): number {
  return validFraction > 0 ? areaMean : 0
}

/** GPU mirror; covered end-to-end by experiments/soilCoverageProbe.ts. */
export function supportedAreaMeanNode(areaMean: FloatNode, validFraction: FloatNode): FloatNode {
  return validFraction.greaterThan(0).select(areaMean, float(0))
}

/** These encoding choices run at graph construction. Legacy and synthetic
 * cubes store log(tau) in R, raw CV in B and the coverage fraction in A.
 */
export function filteredTauNode(texture: CubeTexture, red: FloatNode): FloatNode {
  const encoding = soilFilterEncoding(texture)
  return encoding ? red.div(encoding.scale) : exp(red)
}

export function filteredLogTauNode(texture: CubeTexture, red: FloatNode): FloatNode {
  return soilFilterEncoding(texture) ? log(max(filteredTauNode(texture, red), float(1e-30))) : red
}

export function filteredSupportNode(texture: CubeTexture, alpha: FloatNode): FloatNode {
  const encoding = soilFilterEncoding(texture)
  return encoding ? alpha.div(encoding.scale) : alpha
}

export function filteredCvNode(texture: CubeTexture, blue: FloatNode, alpha: FloatNode): FloatNode {
  if (!soilFilterEncoding(texture)?.cvUsesDensitySupport) return blue
  // Both channels use the same scale, which cancels in the ratio.
  return alpha.greaterThan(0).select(blue.div(alpha), float(0))
}
