import { DataUtils } from 'three/webgpu'
import { directionToFaceUV, smoothstepCubeDirection } from '../field/loadCube'

/** Binary16 spacing, used for measured sampler/output rounding bounds. */
export function halfFloatUlp(value: number): number {
  return 2 ** Math.max(-24, Math.floor(Math.log2(Math.abs(value))) - 10)
}

/** CPU reference for interior-face, filtered RGBA samples. Explicitly refuses
 * seam-crossing queries: hardware cube seams need a cross-face reference.
 * Input R is already GPU log(tau), not raw on-disk log(density).
 */
export function filteredSoilReference(
  faces: readonly Uint16Array[], size: number, direction: readonly [number, number, number],
  measuredWeights?: readonly [number, number],
) {
  const d = smoothstepCubeDirection(...direction, size)
  const { face, u, v } = directionToFaceUV(...d)
  const x = (u + 1) * size / 2 - .5, y = (v + 1) * size / 2 - .5
  const x0 = Math.floor(x), y0 = Math.floor(y)
  const fx = measuredWeights?.[0] ?? x - x0, fy = measuredWeights?.[1] ?? y - y0
  if (x0 < 0 || y0 < 0 || x0 + 1 >= size || y0 + 1 >= size) {
    throw new Error('Reference requires an interior-face query')
  }
  const values = [0, 0, 0, 0]
  for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
    const weight = (i ? fx : 1 - fx) * (j ? fy : 1 - fy)
    for (let c = 0; c < 4; c++) {
      values[c]! += weight * DataUtils.fromHalfFloat(faces[face]![(x0 + i + (y0 + j) * size) * 4 + c]!)
    }
  }
  return { tau: values[3]! > 0 ? Math.exp(values[0]!) : 0,
    cv: values[2]!, support: values[3]!, groupNumerator: values[1]!, logTau: values[0]!,
    x0, y0, weights: [fx, fy] }
}
