import type { CubeTexture, Node } from 'three/webgpu'
import { cubeTexture, vec3 } from 'three/tsl'

/** Sample a data cube in the pipeline's right-handed geographic frame.
 * three.js 0.185.1 CubeTextureNode.setupUV negates X for WebGPU environment
 * maps, even with an explicit direction. Our baked faces already follow the
 * native sampler convention (+X,-X,+Y,-Y,+Z,-Z); that extra flip mirrors
 * geography and disagrees with CPU probes and compute-array sampling.
 * Cancel it at this single boundary, not by rotating the camera or data.
 * These data materials use no environment rotation.
 * GPU regression: experiments/cubeOrientationProbe.ts checks all six faces
 * and asymmetric off-axis texels through the real terrain sampling path.
 */
export function geographicCubeTexture(texture: CubeTexture, direction: Node<'vec3'>) {
  return cubeTexture(texture, vec3(direction.x.negate(), direction.y, direction.z))
}
