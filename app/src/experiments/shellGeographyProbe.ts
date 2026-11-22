import * as THREE from 'three/webgpu'
import { uniform, vec4, float, cameraPosition, positionWorld, normalize } from 'three/tsl'
import { createRenderer } from '../core/Renderer'
import { buildShellMaterial } from '../regimes/Shell'
import { sampleLogTauNode, smoothstepCubeDirectionNode } from '../field/terrainNodes'
import { faceDirection } from '../field/bakeCube'
import { sigma0 } from '../math/optical'
import { WGS84 } from '../math/wgs84'

/** GPU regression: a nadir ray keeps its world/geocentric direction fixed.
 * Compare the ACTUAL shell against a uniform shell with tau read through
 * the terrain sampler at that direction, with identical reconstruction.
 * A latitude gradient distinguishes world direction from ellipsoid-scaled
 * sphere-space direction. Constant fixtures alone cannot catch this bug.
 * Run from an idle same-origin page, not alongside a live viewer.
 */
export async function runShellGeographyProbe() {
  const renderer = await createRenderer(document.createElement('canvas'))
  renderer.setPixelRatio(1)
  renderer.setSize(33, 33, false) // odd size: central pixel is exactly nadir
  renderer.setClearColor(0, 0)
  const target = new THREE.RenderTarget(33, 33, { type: THREE.FloatType })
  const globe = new THREE.SphereGeometry(WGS84.a * 1.04, 64, 32)
  const results: { latitude: number; expected: number; measured: number; relativeError: number }[] = []
  try {
    for (const latitude of [-35, 35]) {
      const phi = latitude * Math.PI / 180
      const direction = new THREE.Vector3(Math.cos(phi), 0, Math.sin(phi))
      const faces = Array.from({ length: 6 }, (_, face) => {
        const data = new Uint16Array(128 * 128 * 4)
        for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) {
          const d = faceDirection(face, (x + .5) / 64 - 1, (y + .5) / 64 - 1)
          const lat = Math.atan2(d[2], Math.hypot(d[0], d[1]))
          data[(y * 128 + x) * 4] = THREE.DataUtils.toHalfFloat(
            Math.max(-8, Math.min(-1, Math.log(.04) + 100 * (lat - phi))))
          data[(y * 128 + x) * 4 + 3] = THREE.DataUtils.toHalfFloat(1)
        }
        return new THREE.DataTexture(data, 128, 128, THREE.RGBAFormat, THREE.HalfFloatType)
      })
      const texture = new THREE.CubeTexture(faces, THREE.CubeReflectionMapping,
        THREE.ClampToEdgeWrapping, THREE.ClampToEdgeWrapping, THREE.LinearFilter,
        THREE.LinearFilter, THREE.RGBAFormat, THREE.HalfFloatType)
      texture.generateMipmaps = false
      texture.needsUpdate = true
      const sampler = new THREE.NodeMaterial()
      // Use the same rasterized central ray as the production shell rather
      // than assuming that a hardware MSAA sample falls at pixel center.
      const point = cameraPosition.add(normalize(positionWorld.sub(cameraPosition)).mul(1.15e7))
      sampler.side = THREE.BackSide
      sampler.fragmentNode = vec4(sampleLogTauNode(texture,
        smoothstepCubeDirectionNode(point, 128)), float(0), float(0), float(1))
      const sampleScene = new THREE.Scene()
      sampleScene.add(new THREE.Mesh(globe, sampler))
      const camera = new THREE.PerspectiveCamera(50, 1, 100, 1e9)
      camera.position.copy(direction).multiplyScalar(WGS84.a + 1.15e7)
      camera.lookAt(0, 0, 0)
      try {
        renderer.setRenderTarget(target)
        await renderer.renderAsync(sampleScene, camera)
        const sample = await renderer.readRenderTargetPixelsAsync(target, 16, 16, 1, 1)
        const tau = Math.exp(sample[0]!)
        const alphas: number[] = []
        for (const useCube of [false, true]) {
          const material = buildShellMaterial({ Hs: uniform(32000), Hmax: uniform(192000),
            sigma0: uniform(sigma0(tau, 32000, 192000)), nMax: 128,
            ...(useCube ? { cube: { texture, sigma0PerTau: uniform(sigma0(1, 32000, 192000)), tauCap: uniform(1) } } : {}) })
          const scene = new THREE.Scene()
          scene.add(new THREE.Mesh(globe, material))
          try {
            await renderer.renderAsync(scene, camera)
            const pixel = await renderer.readRenderTargetPixelsAsync(target, 16, 16, 1, 1)
            alphas.push(pixel[3]!)
          } finally { material.dispose() }
        }
        const [expected, measured] = alphas as [number, number]
        results.push({ latitude, expected, measured, relativeError: Math.abs(measured / expected - 1) })
      } finally { sampler.dispose(); texture.dispose(); faces.forEach(f => f.dispose()) }
    }
    return { pass: results.every(r => r.expected > .01 && r.relativeError < .005), results }
  } finally { renderer.setRenderTarget(null); target.dispose(); globe.dispose(); renderer.dispose() }
}
