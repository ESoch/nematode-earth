import * as THREE from 'three/webgpu'
import { float, uniform, vec4 } from 'three/tsl'
import { createRenderer } from '../core/Renderer'
import { loadCubeTexture, soilFilterEncoding } from '../field/loadCube'
import { faceDirection } from '../field/bakeCube'
import { smoothstepCubeDirectionNode } from '../field/terrainNodes'
import { geographicCubeTexture } from '../field/geographicCube'
import { filteredTauNode, filteredCvNode, filteredSupportNode } from '../field/densityCoverage'
import { filteredSoilReference, halfFloatUlp } from './soilCandidateReference'

/** Actual candidate bytes through the production loader, geographic sampler,
 * smoothstep reconstruction and coverage gate, with forced GPU readback.
 * Run on an idle page: no active viewer/device contention. This validates
 * sampling/encoding, not ecological accuracy or full-frame energy conservation.
 */
export async function runSoilCandidateProbe(baseUrl: string) {
  const loaded = await loadCubeTexture(baseUrl)
  const { meta, faces, groupSupportFaces: masks } = loaded
  const encoding = soilFilterEncoding(loaded.texture)
  if (!masks || !encoding?.cvUsesDensitySupport) throw new Error('Probe requires a shared-CV-support soil v2 atlas')
  const size = meta.face_size
  const probes: { face: number; kind: string; direction: [number, number, number] }[] = []
  const kinds = ['supported', 'unsupported', 'partial-coverage', 'composition-unavailable', 'coastal-midpoint', 'texel-center']
  for (let f = 0; f < 6; f++) {
    for (const kind of kinds) {
      search: for (let y = 2; y < size - 3; y++) for (let x = 2; x < size - 3; x++) {
        const i = x + y * size
        const a = THREE.DataUtils.fromHalfFloat(loaded.faces[f]![i*4+3]!)
        const gs = THREE.DataUtils.fromHalfFloat(masks[f]![i]!)
        const matches = kind === 'supported' || kind === 'texel-center' ? a > .99
          : kind === 'unsupported' ? a === 0
          : kind === 'partial-coverage' ? a > .05 && a < .95
          : kind === 'coastal-midpoint' ? a > .25 && loaded.faces[f]![(i + 1) * 4 + 3] === 0 : a > 0 && gs === 0
        if (!matches) continue
        const offset = kind === 'texel-center' ? [0, 0] : kind === 'coastal-midpoint' ? [.5, 0] : [.23, .61]
        const d = faceDirection(f, (x + .5 + offset[0]!) * 2 / size - 1, (y + .5 + offset[1]!) * 2 / size - 1)
        probes.push({ face: f, kind, direction: d })
        break search
      }
    }
    // ±X own four edges each. ±Y own the remaining two edges with ±Z.
    // Choose the strongest density change on each edge, testing registration
    // on both faces, or exact zero for an entirely unsupported seam.
    const axes = f < 2 ? ['u', 'v'] : f < 4 ? ['v'] : []
    for (const axis of axes) for (const sign of [-1, 1]) {
      let best = -1
      let seamDirection: [number, number, number] | undefined
      for (let along = 1; along < size - 1; along++) {
        const coordinate = (along + .5) * 2 / size - 1
        const d = faceDirection(f, axis === 'u' ? sign : coordinate, axis === 'v' ? sign : coordinate)
        const first = filteredSoilReference(faces, size, d, [0, 0])
        const acrossX = first.x0 < 0 || first.x0 === size - 1
        const other = filteredSoilReference(faces, size, d, acrossX ? [1, 0] : [0, 1])
        const difference = Math.abs(first.tau - other.tau)
        if (difference > best) { best = difference; seamDirection = d }
      }
      if (seamDirection) probes.push({ face: f, kind: 'face-seam', direction: seamDirection })
    }
  }
  const renderer = await createRenderer(document.createElement('canvas'))
  renderer.setPixelRatio(1)
  renderer.setSize(8, 8, false)
  const target = new THREE.RenderTarget(8, 8, { type: THREE.FloatType })
  const calibrationTarget = new THREE.RenderTarget(8, 8, { type: THREE.FloatType })
  // Alternating exact 0/1 ramps measure the hardware's interpolation weights
  // independently from the physical candidate values.
  const pattern = new Uint16Array(size * size * 4)
  const one = THREE.DataUtils.toHalfFloat(1)
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    pattern[(x + y * size) * 4] = x % 2 ? one : 0
    pattern[(x + y * size) * 4 + 1] = y % 2 ? one : 0
  }
  const calibrationFaces = Array.from({ length: 6 }, () =>
    new THREE.DataTexture(pattern, size, size, THREE.RGBAFormat, THREE.HalfFloatType))
  const calibrationTexture = new THREE.CubeTexture(calibrationFaces, THREE.CubeReflectionMapping,
    THREE.ClampToEdgeWrapping, THREE.ClampToEdgeWrapping, THREE.LinearFilter,
    THREE.LinearFilter, THREE.RGBAFormat, THREE.HalfFloatType)
  calibrationTexture.generateMipmaps = false
  calibrationTexture.needsUpdate = true
  const geometry = new THREE.PlaneGeometry(2, 2)
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2)
  camera.position.z = 1
  const direction = uniform(new THREE.Vector3(1, 0, 0))
  const sample = geographicCubeTexture(loaded.texture,
    smoothstepCubeDirectionNode(direction, size)).level(float(0))
  const material = new THREE.NodeMaterial()
  material.toneMapped = false
  material.fragmentNode = vec4(filteredTauNode(loaded.texture, sample.r),
    filteredCvNode(loaded.texture, sample.b, sample.a), filteredSupportNode(loaded.texture, sample.a), 1)
  const scene = new THREE.Scene()
  const mesh = new THREE.Mesh(geometry, material)
  scene.add(mesh)
  const calibration = geographicCubeTexture(calibrationTexture,
    smoothstepCubeDirectionNode(direction, size)).level(float(0))
  const calibrationMaterial = new THREE.NodeMaterial()
  calibrationMaterial.toneMapped = false
  calibrationMaterial.fragmentNode = vec4(calibration.r, calibration.g, 0, 1)
  const results = []
  try {
    for (const probe of probes) {
      direction.value.set(...probe.direction)
      renderer.setRenderTarget(target)
      await renderer.renderAsync(scene, camera)
      const bytes = await renderer.readRenderTargetPixelsAsync(target, 4, 4, 1, 1)
      const measured = Array.from(bytes.slice(0, 3))
      const expected = filteredSoilReference(faces, size, probe.direction)
      mesh.material = calibrationMaterial
      renderer.setRenderTarget(calibrationTarget)
      await renderer.renderAsync(scene, camera)
      const rawCalibration = await renderer.readRenderTargetPixelsAsync(calibrationTarget, 4, 4, 1, 1)
      const calibrated = Array.from(rawCalibration.slice(0, 4))
      const weights: [number, number] = [expected.x0 % 2 ? 1-calibrated[0]! : calibrated[0]!,
        expected.y0 % 2 ? 1-calibrated[1]! : calibrated[1]!]
      // The calibration ramp rotates with adjacent faces. Exact edge-center
      // probes use their known 50/50 weights; interiors use measured weights.
      const seam = probe.kind === 'face-seam'
      const hardwareExpected = seam ? expected : filteredSoilReference(faces, size, probe.direction, weights)
      mesh.material = material
      const weightPass = seam || weights.every((w, i) => Math.abs(w - expected.weights[i]!) <= 1 / 512 + 2e-5)
      // R and the CV numerator each round at upload and sampling. A scales
      // exactly at upload. CV division also inherits a denominator rounding.
      const { tau, cv, support } = hardwareExpected
      const smallest = 2 ** -24 / encoding.scale
      const tauPass = Math.abs(measured[0]! - tau) <= tau * 2 ** -10 + smallest
      const cvPass = support > 0
        ? Math.abs(measured[1]! - cv) <= cv * 3 * 2 ** -11 + smallest / support
        : measured[1] === 0
      const supportPass = Math.abs(measured[2]! - support) <= halfFloatUlp(support * encoding.scale) / encoding.scale / 2 + 1e-7
      const pass = weightPass && tauPass && cvPass && supportPass
      results.push({ face: probe.face, kind: probe.kind, expected, hardwareExpected, measured, pass,
        weightPass, tauPass, cvPass, supportPass })
    }
    // Negative control: G must not influence density or CV.
    const control = probes.findIndex(probe => probe.kind === 'texel-center')
    if (control < 0) throw new Error('Missing supported texel-center control')
    direction.value.set(...probes[control]!.direction)
    const baseline = results[control]!.measured
    for (const face of loaded.texture.images as THREE.DataTexture[]) {
      const data = face.image.data as Uint16Array
      for (let i = 1; i < data.length; i += 4) data[i] = THREE.DataUtils.toHalfFloat(100)
      face.needsUpdate = true
    }
    loaded.texture.needsUpdate = true
    renderer.setRenderTarget(target)
    await renderer.renderAsync(scene, camera)
    const mutated = Array.from((await renderer.readRenderTargetPixelsAsync(target, 4, 4, 1, 1)).slice(0, 3))
    const gIndependent = mutated.every((value, i) => value === baseline[i])
    return { pass: kinds.every(kind => probes.some(probe => probe.kind === kind))
        && probes.filter(probe => probe.kind === 'face-seam').length === 12
        && results.every(r => r.pass) && gIndependent,
      source: baseUrl, schema: meta.schema, faceSize: size,
      primaryHashes: meta.face_sha256, groupSupportHashes: meta.group_support!.sha256,
      probeCount: probes.length, gIndependent, results }
  } finally {
    renderer.setRenderTarget(null); material.dispose(); geometry.dispose(); target.dispose()
    calibrationMaterial.dispose(); calibrationTarget.dispose(); calibrationTexture.dispose()
    calibrationFaces.forEach(f => f.dispose())
    loaded.texture.dispose()
    for (const face of loaded.texture.images as THREE.DataTexture[]) face.dispose()
    renderer.dispose()
  }
}
