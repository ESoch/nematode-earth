import { describe, expect, it } from 'vitest'
import { verticalFovForViewport } from '../src/core/viewFraming'
import { PerspectiveCamera, Vector3 } from 'three'
import * as framing from '../src/core/viewFraming'

describe('the opening globe fits the short side of the viewport', () => {
  it.each([[1440, 900, 50], [390, 844, 60], [844, 390, 60]])('fits %i by %i with a %i degree short-side field', (width, height, expected) => {
    const vertical = verticalFovForViewport(width, height) * Math.PI / 180
    const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * width / height)
    expect(Math.min(vertical, horizontal) * 180 / Math.PI).toBeCloseTo(expected, 10)
  })

  it('preserves the desktop vertical field of view', () => {
    expect(verticalFovForViewport(1800, 986)).toBe(50)
  })

  it('returns a usable camera angle while the viewport cannot be measured', () => {
    expect(verticalFovForViewport(0, 0)).toBe(50)
  })
})

describe('full-canvas projection fitted around the interface', () => {
  it.each([
    [1440, 1000, { left: 410, top: 90, width: 980, height: 760 }],
    [390, 844, { left: 20, top: 252, width: 350, height: 288 }],
    [320, 568, { left: 20, top: 114, width: 280, height: 180 }],
    [2560, 1080, { left: 416, top: 90, width: 2080, height: 840 }],
  ])('centers a circular globe inside the free area at %i×%i', (width, height, region) => {
    const camera = new PerspectiveCamera(50, width / height, 1, 1000)
    framing.frameCameraToRegion(camera, width, height, region)
    // A radius-one sphere at z=-3 has tangent coordinates (sqrt(8)/3, 0, -8/3).
    // Project real geometry through the actual camera, not a mirrored formula.
    const project = (p: Vector3) => {
      p.project(camera)
      return { x: (p.x + 1) * width / 2, y: (1 - p.y) * height / 2 }
    }
    const center = project(new Vector3(0, 0, -3))
    const right = project(new Vector3(Math.sqrt(8) / 3, 0, -8 / 3))
    const top = project(new Vector3(0, Math.sqrt(8) / 3, -8 / 3))
    expect(center.x).toBeCloseTo(region.left + region.width / 2, 8)
    expect(center.y).toBeCloseTo(region.top + region.height / 2, 8)
    expect(right.x - center.x).toBeCloseTo(center.y - top.y, 8)
    expect(right.x).toBeLessThan(region.left + region.width)
    expect(top.y).toBeGreaterThan(region.top)
    expect(2 * (right.x - center.x)).toBeGreaterThan(Math.min(region.width, region.height) * 0.7)
  })

  it('does not poison the projection during a transient zero-sized layout', () => {
    const camera = new PerspectiveCamera(50, 1, 1, 1000)
    framing.frameCameraToRegion(camera, 0, 0, { left: 0, top: 0, width: 0, height: 0 })
    expect(camera.projectionMatrix.elements.every(Number.isFinite)).toBe(true)
  })

  it('reports whether the projection matches the requested viewport for downstream quality limits', () => {
    const camera = new PerspectiveCamera(50, 1, 1, 1000)
    expect(framing.frameCameraToRegion(camera, 800, 600, {
      left: 100, top: 50, width: 600, height: 400,
    })).toBe(true)
    const last = [...camera.projectionMatrix.elements]
    expect(framing.frameCameraToRegion(camera, 1600, 900, {
      left: 100, top: 50, width: 0, height: 400,
    })).toBe(false)
    expect(camera.projectionMatrix.elements).toEqual(last)
  })
})
