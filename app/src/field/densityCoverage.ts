import { float } from 'three/tsl'
import type { Node } from 'three/webgpu'

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
