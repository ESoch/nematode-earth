import * as THREE from 'three/webgpu'
import { uniform } from 'three/tsl'
import { createRenderer } from '../core/Renderer'
import { buildShellMaterial } from '../regimes/Shell'
import { sigma0 } from '../math/optical'
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
  const target = new THREE.RenderTarget(32, 32)
  const geometry = new THREE.SphereGeometry(WGS84.a * 1.031, 64, 32)
  const camera = new THREE.PerspectiveCamera(50, 1, 100, 1e9)
  camera.position.set(WGS84.a + 1.15e7, 0, 0)
  camera.lookAt(0, 0, 0)
  const alpha: Record<string, number> = {}
  try {
    for (const [name, coverage] of [['full', 1], ['coast', 0.5], ['missing', 0]] as const) {
      const faces = Array.from({ length: 6 }, () => {
        const bytes = new Uint16Array(2 * 2 * 4)
        for (let i = 0; i < bytes.length; i += 4) {
          bytes[i] = THREE.DataUtils.toHalfFloat(Math.log(0.04))
          bytes[i + 3] = THREE.DataUtils.toHalfFloat(coverage)
        }
        return new THREE.DataTexture(bytes, 2, 2, THREE.RGBAFormat, THREE.HalfFloatType)
      })
      const texture = new THREE.CubeTexture(faces, THREE.CubeReflectionMapping,
        THREE.ClampToEdgeWrapping, THREE.ClampToEdgeWrapping,
        THREE.LinearFilter, THREE.LinearFilter, THREE.RGBAFormat, THREE.HalfFloatType)
      texture.generateMipmaps = false
      texture.needsUpdate = true
      const material = buildShellMaterial({ Hs: uniform(32000), Hmax: uniform(192000),
        sigma0: uniform(0), nMax: 128,
        cube: { texture, sigma0PerTau: uniform(sigma0(1, 32000, 192000)), tauCap: uniform(1) } })
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(geometry, material))
      try {
        renderer.setRenderTarget(target)
        await renderer.renderAsync(scene, camera)
        const pixel = await renderer.readRenderTargetPixelsAsync(target, 16, 16, 1, 1)
        alpha[name] = pixel[3]!
      } finally { material.dispose(); texture.dispose(); faces.forEach(f => f.dispose()) }
    }
    return { alpha, pass: alpha.full! > 3 && alpha.coast === alpha.full && alpha.missing === 0 }
  } finally { renderer.setRenderTarget(null); target.dispose(); geometry.dispose(); renderer.dispose() }
}
