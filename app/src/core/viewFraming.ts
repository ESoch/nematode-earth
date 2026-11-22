import type { PerspectiveCamera } from 'three'

/** Keep the opening globe inside portrait screens without changing altitude. */
export function verticalFovForViewport(width: number, height: number): number {
  if (!(width > 0 && height > 0) || !Number.isFinite(width / height)) return 50
  // A phone also needs room above and below the globe for the introduction
  // and touch controls; use a little more angular breathing room there.
  const shortSideFov = (Math.min(width, height) < 700 ? 60 : 50) * Math.PI / 180
  return 2 * Math.atan(Math.tan(shortSideFov / 2) / Math.min(1, width / height)) * 180 / Math.PI
}

export interface ViewRegion {
  left: number
  top: number
  width: number
  height: number
}

/**
 * Keep a full-window render buffer, but compose the opening globe in the space
 * left by the interface. This changes the projection, never world position or
 * altitude. Matter, shell and particle passes all use this same camera.
 *
 * The region's short side gets a 50° field. The opening globe subtends about
 * 42°, leaving room for its atmosphere. A view offset moves that field into
 * the region without stretching a sphere or cropping the canvas on descent.
 * Returns false when layout is unmeasurable; callers must keep the previous
 * projection's viewport dimensions for derived screen-space calculations.
 */
export function frameCameraToRegion(
  camera: PerspectiveCamera,
  width: number,
  height: number,
  region: ViewRegion,
): boolean {
  if (![width, height, region.left, region.top, region.width, region.height].every(Number.isFinite)
    || width <= 0 || height <= 0 || region.width <= 0 || region.height <= 0) return false
  const shortSide = Math.min(region.width, region.height)
  camera.fov = 2 * Math.atan(Math.tan(25 * Math.PI / 180) * height / shortSide) * 180 / Math.PI
  camera.setViewOffset(
    width, height,
    width / 2 - (region.left + region.width / 2),
    height / 2 - (region.top + region.height / 2),
    width, height,
  )
  return true
}
