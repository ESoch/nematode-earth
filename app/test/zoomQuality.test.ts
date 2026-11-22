import { describe, expect, it } from 'vitest'
import { PerspectiveCamera, Vector3 } from 'three'
import { zoomQualityForView } from '../src/core/zoomQuality'
import { frameCameraToRegion } from '../src/core/viewFraming'
import { ScaleCamera } from '../src/core/ScaleCamera'

describe('screen-aware dataset detail', () => {
  // Hand-derived fixtures: plain cube edges are bounded by 2R/N; at 90°
  // and 800 CSS px high, focal length is 400 px. The budget here is 8 px.
  // Catch using mean texel area, planet radius as distance, or a finer atlas.
  it.each([
    [512, 24_886.71875, 1_244_335.9375],
    [128, 99_546.875, 4_977_343.75],
    [2048, 24_886.71875, 1_244_335.9375],
  ])('bounds a %i-face dataset by its worst texel, capped at the accepted detail', (faceSize, worstTexelM, floorM) => {
    const camera = new PerspectiveCamera(90, 2, 1, 1e9)
    const quality = zoomQualityForView(camera, 1600, 800, faceSize)
    expect(quality?.worstTexelM).toBeCloseTo(worstTexelM, 6)
    expect(quality?.floorM).toBeCloseTo(floorM, 6)
  })

  it('reserves the accepted soil detail before loading, without a coarse temporary limit', () => {
    const camera = new PerspectiveCamera(90, 2, 1, 1e9)
    const c = new ScaleCamera()
    const before = zoomQualityForView(camera, 1600, 800)!
    expect(before?.floorM).toBeCloseTo(1_244_335.9375, 6)
    c.qualityFloorM = before.floorM
    c.setAltitude(0.01)
    c.qualityFloorM = zoomQualityForView(camera, 1600, 800, 512)!.floorM
    expect(c.altitude).toBeCloseTo(1_244_335.9375, 6)
  })

  it('uses the actual effective projection including optical zoom', () => {
    const camera = new PerspectiveCamera(90, 2, 1, 1e9)
    camera.zoom = 2
    camera.updateProjectionMatrix()
    const quality = zoomQualityForView(camera, 1600, 800)
    expect(quality?.focalCssPx).toBeCloseTo(800, 8)
    expect(quality?.floorM).toBeCloseTo(2_488_671.875, 6)
  })

  it('includes view-offset cropping in the effective field of view', () => {
    const camera = new PerspectiveCamera(90, 2, 1, 1e9)
    camera.setViewOffset(1600, 800, 200, 100, 800, 400)
    const quality = zoomQualityForView(camera, 800, 400)
    expect(quality?.focalCssPx).toBeCloseTo(400, 8)
    expect(quality?.floorM).toBeCloseTo(1_244_335.9375, 6)
  })

  it.each([
    [1440, 1000, { left: 410, top: 90, width: 980, height: 760 }],
    [390, 844, { left: 20, top: 252, width: 350, height: 288 }],
    [844, 390, { left: 300, top: 70, width: 520, height: 200 }],
    [320, 568, { left: 20, top: 114, width: 280, height: 180 }],
  ])('keeps the worst texel within 8 CSS px in the real %i×%i framing', (width, height, region) => {
    const camera = new PerspectiveCamera(50, width / height, 1, 1e9)
    frameCameraToRegion(camera, width, height, region)
    const quality = zoomQualityForView(camera, width, height)!
    expect(quality).not.toBeNull()
    // Exercise the real camera matrix by projecting a tangent patch.
    const origin = new Vector3(0, 0, -quality.floorM).project(camera)
    const right = new Vector3(24_886.71875, 0, -quality.floorM).project(camera)
    const top = new Vector3(0, 24_886.71875, -quality.floorM).project(camera)
    expect((right.x - origin.x) * width / 2).toBeCloseTo(8, 8)
    expect((top.y - origin.y) * height / 2).toBeCloseTo(8, 8)
  })

  it('recomputes on actual resize without zooming in when the viewport shrinks again', () => {
    const camera = new PerspectiveCamera(90, 2, 1, 1e9)
    const c = new ScaleCamera()
    c.qualityFloorM = zoomQualityForView(camera, 800, 400)?.floorM ?? 0
    c.setAltitude(0.01)
    expect(c.altitude).toBeCloseTo(622_167.96875, 6)
    c.qualityFloorM = zoomQualityForView(camera, 1600, 800)!.floorM
    expect(c.altitude).toBeCloseTo(1_244_335.9375, 6)
    c.qualityFloorM = zoomQualityForView(camera, 800, 400)!.floorM
    expect(c.altitude).toBeCloseTo(1_244_335.9375, 6)
  })

  it('uses the stricter screen axis if the projection and canvas aspect differ', () => {
    const camera = new PerspectiveCamera(90, 1, 1, 1e9)
    const quality = zoomQualityForView(camera, 1600, 800)
    expect(quality?.floorM).toBeCloseTo(2_488_671.875, 6)
  })

  it('lets visual calibration change the pixel budget without changing the dataset', () => {
    const camera = new PerspectiveCamera(90, 2, 1, 1e9)
    expect(zoomQualityForView(camera, 1600, 800, 512, 4)?.floorM).toBeCloseTo(2_488_671.875, 6)
  })

  it.each([0, -1, NaN, Infinity, 127.5])('falls back conservatively for invalid face size %s', (faceSize) => {
    const camera = new PerspectiveCamera(90, 2, 1, 1e9)
    expect(zoomQualityForView(camera, 1600, 800, faceSize)?.floorM).toBeCloseTo(4_977_343.75, 6)
  })

  it.each([[0, 800], [1600, 0], [-1, 800], [NaN, 800], [1600, Infinity]])(
    'does not replace the last valid floor during an unmeasurable %s×%s layout', (width, height) => {
      const camera = new PerspectiveCamera(90, 2, 1, 1e9)
      expect(zoomQualityForView(camera, width, height)).toBeNull()
    },
  )
})
