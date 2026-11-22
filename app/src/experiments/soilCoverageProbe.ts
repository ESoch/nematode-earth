import * as THREE from 'three/webgpu'
import { uniform } from 'three/tsl'
import { createRenderer } from '../core/Renderer'
import { buildShellMaterial } from '../regimes/Shell'
import { soilFaceForFiltering } from '../field/loadCube'
import { E_A, sigma0 } from '../math/optical'
import { WGS84 } from '../math/wgs84'

/** Browser-only regression: actual production shell, constant RGBA fixtures.
 * Run from an otherwise idle same-origin page (not beside a live viewer):
 * (await import('/src/experiments/soilCoverageProbe.ts')).runSoilCoverageProbe()
 * R is an ALL-CELL area mean. Changing A from 1 to .5 must not price that
 * mean a second time. A=0 must suppress even an arbitrary positive floor.
 * Offscreen readback forces GPU completion; this is not a submission timer.
 */
export async function runSoilCoverageProbe() {
  const renderer = await createRenderer(document.createElement('canvas'))
  renderer.setPixelRatio(1)
  renderer.setSize(32, 32, false)
  renderer.setClearColor(0, 0)
  const target = new THREE.RenderTarget(32, 32, { type: THREE.HalfFloatType })
  const geometry = new THREE.SphereGeometry(WGS84.a * 1.031, 64, 32)
  const camera = new THREE.PerspectiveCamera(50, 1, 100, 1e9)
  camera.position.set(WGS84.a + 1.15e7, 0, 0)
  camera.lookAt(0, 0, 0)
  const legacyAlpha: Record<string, number> = {}, v2Alpha: Record<string, number> = {}
  const radiance: Record<string, number[]> = {}
  const backdrop = new THREE.Color(.125, .25, .375)
  let composite: number[] = []
  const encoding = { scale: 8192, cvUsesDensitySupport: true }
  try {
    for (const mode of ['legacy', 'v2'] as const) {
      for (const [name, coverage] of [['full', 1], ['coast', 0.5], ['missing', 0], ['dense', 1]] as const) {
        const tau = name === 'dense' ? .5 : .04
        const faces = Array.from({ length: 6 }, () => {
          const bytes = new Uint16Array(2 * 2 * 4)
          for (let i = 0; i < bytes.length; i += 4) {
            bytes[i] = THREE.DataUtils.toHalfFloat(Math.log(mode === 'v2' ? tau / E_A : tau))
            bytes[i + 3] = THREE.DataUtils.toHalfFloat(coverage)
          }
          return new THREE.DataTexture(mode === 'v2' ? soilFaceForFiltering(bytes, encoding) : bytes, 2, 2, THREE.RGBAFormat, THREE.HalfFloatType)
        })
        const texture = new THREE.CubeTexture(faces, THREE.CubeReflectionMapping,
          THREE.ClampToEdgeWrapping, THREE.ClampToEdgeWrapping,
          THREE.LinearFilter, THREE.LinearFilter, THREE.RGBAFormat, THREE.HalfFloatType)
        if (mode === 'v2') texture.userData.soilFilter = encoding
        texture.generateMipmaps = false
        texture.needsUpdate = true
        const material = buildShellMaterial({ Hs: uniform(32000), Hmax: uniform(192000),
          sigma0: uniform(0), nMax: 128,
          cube: { texture, sigma0PerTau: uniform(sigma0(1, 32000, 192000)), tauCap: uniform(1) } })
        const scene = new THREE.Scene()
        scene.add(new THREE.Mesh(geometry, material))
        try {
          renderer.setClearColor(0, 0)
          renderer.setRenderTarget(target)
          await renderer.renderAsync(scene, camera)
          const pixel = await renderer.readRenderTargetPixelsAsync(target, 16, 16, 1, 1)
          const values = mode === 'v2' ? v2Alpha : legacyAlpha
          const rgba = Array.from(pixel.slice(0, 4), THREE.DataUtils.fromHalfFloat)
          values[name] = Math.round(rgba[3]! * 255)
          if (mode === 'v2') radiance[name] = rgba
          if (mode === 'v2' && name === 'full') {
            renderer.setClearColor(backdrop, 1)
            await renderer.renderAsync(scene, camera)
            const pixel = await renderer.readRenderTargetPixelsAsync(target, 16, 16, 1, 1)
            composite = Array.from(pixel.slice(0, 4), THREE.DataUtils.fromHalfFloat)
          }
        } finally { material.dispose(); texture.dispose(); faces.forEach(f => f.dispose()) }
      }
    }
    const valid = (a: Record<string, number>) => a.full! > 3 && a.coast === a.full && a.missing === 0
    const tint = [.72, .86, .95]
    // RGB is the emitted column, tint × alpha over black. Retain the direct
    // measurements so a blend error cannot hide behind unchanged opacity.
    const radiancePass = Object.values(radiance).every(rgba => tint.every((t, c) =>
      Math.abs(rgba[c]! - t * rgba[3]!) < Math.max(1e-6, rgba[3]! * .002)))
    const full = radiance.full!
    const compositePass = backdrop.toArray().every((bg, c) =>
      Math.abs(composite[c]! - (full[c]! + bg * (1 - full[3]!))) < .0006) && composite[3] === 1
    return { alpha: legacyAlpha, v2Alpha, radiance, radiancePass, compositePass,
      pass: valid(legacyAlpha) && valid(v2Alpha) && radiancePass && compositePass }

  } finally { renderer.setRenderTarget(null); target.dispose(); geometry.dispose(); renderer.dispose() }
}
