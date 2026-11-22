import type { PerspectiveCamera } from 'three'
import { WGS84 } from '../math/wgs84'

/** Visual calibration knob: CSS pixels, deliberately independent of DPR. */
export const MAX_PROJECTED_TEXEL_CSS_PX = 8
export const ACCEPTED_SOIL_FACE_SIZE = 512

export interface ZoomQuality {
  faceSize: number
  worstTexelM: number
  focalCssPx: number
  maxTexelCssPx: number
  floorM: number
}

/**
 * A single public limit for the entire sweep, based on the coarsest accepted
 * soil detail, not whichever elevation/urban tier happens to be loading.
 *
 * Plain cube texels have nonuniform area: 512 has ~19.55 km mean spacing but
 * ~24.88 km worst spacing (128: ~99.53 km). 2R/N bounds the largest edge and
 * is slightly conservative: 24.887 / 99.547 km. The mean is not a safe limit.
 *
 * Projected size = texel metres * focal CSS pixels / altitude metres. Read
 * the actual projection matrix AFTER framing so FOV, zoom and cropped view
 * offsets are included. An off-axis translation changes position, not size.
 * No sweep, ground visibility, load progress, or device pixels enter here.
 * Invalid layout returns null so callers retain their last usable floor.
 */
export function zoomQualityForView(
  camera: Pick<PerspectiveCamera, 'projectionMatrix'>,
  widthCssPx: number,
  heightCssPx: number,
  faceSize = ACCEPTED_SOIL_FACE_SIZE,
  maxTexelCssPx = MAX_PROJECTED_TEXEL_CSS_PX,
): ZoomQuality | null {
  if (![widthCssPx, heightCssPx, maxTexelCssPx].every((v) => Number.isFinite(v) && v > 0)) return null
  const matrix = camera.projectionMatrix.elements
  const fx = Math.abs(matrix[0]!) * widthCssPx / 2
  const fy = Math.abs(matrix[5]!) * heightCssPx / 2
  if (![fx, fy].every((v) => Number.isFinite(v) && v > 0)) return null
  const focalCssPx = Math.max(fx, fy)
  const acceptedSize = Number.isSafeInteger(faceSize) && faceSize > 0
    ? Math.min(faceSize, ACCEPTED_SOIL_FACE_SIZE)
    : 128
  const worstTexelM = 2 * WGS84.R_MEAN / acceptedSize
  const floorM = worstTexelM * focalCssPx / maxTexelCssPx
  if (!Number.isFinite(floorM)) return null
  return { faceSize: acceptedSize, worstTexelM, focalCssPx, maxTexelCssPx, floorM }
}
