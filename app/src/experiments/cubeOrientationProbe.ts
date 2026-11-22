import * as THREE from 'three/webgpu'
import { vec3, vec4, float } from 'three/tsl'
import { createRenderer } from '../core/Renderer'
import { sampleElevationNode } from '../field/terrainNodes'
import { faceDirection } from '../field/bakeCube'

/** Real GPU regression for geographic cube face identity AND local axes.
 * Six distinct face values with an asymmetric u/v ramp avoid a symmetric
 * fixture passing a mirrored sampler. Uses the actual terrain sampler.
 */
export async function runCubeOrientationProbe() {
  const renderer = await createRenderer(document.createElement('canvas'))
  renderer.setPixelRatio(1)
  renderer.setSize(8, 8, false)
  const target = new THREE.RenderTarget(8, 8, { type: THREE.HalfFloatType })
  const geometry = new THREE.PlaneGeometry(2, 2)
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2)
  camera.position.z = 1
  const faces = Array.from({ length: 6 }, (_, face) => {
    const data = new Uint16Array(4 * 4 * 4)
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
      const u = (x + 0.5) / 2 - 1, v = (y + 0.5) / 2 - 1
      data[(y * 4 + x) * 4] = THREE.DataUtils.toHalfFloat(10 * (face + 1) + 2 * u + v)
    }
    return new THREE.DataTexture(data, 4, 4, THREE.RGBAFormat, THREE.HalfFloatType)
  })
  const texture = new THREE.CubeTexture(faces, THREE.CubeReflectionMapping,
    THREE.ClampToEdgeWrapping, THREE.ClampToEdgeWrapping, THREE.LinearFilter,
    THREE.LinearFilter, THREE.RGBAFormat, THREE.HalfFloatType)
  texture.generateMipmaps = false
  texture.needsUpdate = true
  const results: { face: number; u: number; v: number; expected: number; measured: number }[] = []
  try {
    for (let face = 0; face < 6; face++) for (const [u, v] of [[0, 0], [0.25, -0.25]]) {
      const d = faceDirection(face, u!, v!)
      const material = new THREE.NodeMaterial()
      material.fragmentNode = vec4(sampleElevationNode(texture, vec3(d[0], d[1], d[2])), float(0), float(0), float(1))
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(geometry, material))
      try {
        renderer.setRenderTarget(target)
        await renderer.renderAsync(scene, camera)
        const data = await renderer.readRenderTargetPixelsAsync(target, 4, 4, 1, 1)
        results.push({ face, u: u!, v: v!, expected: 10 * (face + 1) + 2 * u! + v!,
          measured: THREE.DataUtils.fromHalfFloat(data[0]!) })
      } finally { material.dispose() }
    }
    return { pass: results.every(r => Math.abs(r.measured - r.expected) < 0.05), results }
  } finally { renderer.setRenderTarget(null); geometry.dispose(); target.dispose(); texture.dispose(); faces.forEach(f => f.dispose()); renderer.dispose() }
}
