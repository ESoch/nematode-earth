import { DataUtils } from 'three/webgpu'
import { faceDirection } from '../field/bakeCube'
import { directionToFaceUV, smoothstepCubeDirection } from '../field/loadCube'
import { E_A } from '../math/optical'

/** Binary16 spacing, including subnormals. */
export function halfFloatUlp(value: number): number {
  return 2 ** Math.max(-24, Math.floor(Math.log2(Math.abs(value))) - 10)
}

/** Return the tap on the adjacent cube face when a filter straddles an edge.
 * Corner taps touch three faces and have platform-dependent tap assignment.
 */
function tap(size: number, face: number, x: number, y: number): [number, number] {
  const outsideX = x < 0 || x >= size, outsideY = y < 0 || y >= size
  if (outsideX && outsideY) throw new Error('Three-face corner is outside the probe domain')
  if (outsideX || outsideY) {
    const next = directionToFaceUV(...faceDirection(face, (x + .5) * 2 / size - 1, (y + .5) * 2 / size - 1))
    const index = (uv: number) => Math.max(0, Math.min(size - 1, Math.floor((uv + 1) * size / 2)))
    return [next.face, index(next.u) + size * index(next.v)]
  }
  return [face, x + size * y]
}

/** Independent physical reference from ORIGINAL disk texels. The shipping
 * product's validation reports identical density and CV support. Its R is
 * already the mean over all area; its B is conditional on supported area.
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
  let tau = 0, cvNumerator = 0, support = 0
  for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
    const weight = (i ? fx : 1 - fx) * (j ? fy : 1 - fy)
    if (weight === 0) continue
    const [f, p] = tap(size, face, x0 + i, y0 + j)
    const read = (c: number) => DataUtils.fromHalfFloat(faces[f]![4 * p + c]!)
    const a = read(3)
    if (a === 0) continue
    tau += weight * Math.exp(read(0)) * E_A
    cvNumerator += weight * read(2) * a
    support += weight * a
  }
  return { tau, cv: support > 0 ? cvNumerator / support : 0, support, x0, y0, weights: [fx, fy] }
}
