import * as THREE from 'three/webgpu'
import { exp, float, uniform, vec4 } from 'three/tsl'
import { createRenderer } from '../core/Renderer'
import { loadCubeTexture, densityFaceToTauFace } from '../field/loadCube'
import { faceDirection } from '../field/bakeCube'
import { smoothstepCubeDirectionNode } from '../field/terrainNodes'
import { geographicCubeTexture } from '../field/geographicCube'
import { supportedAreaMeanNode } from '../field/densityCoverage'
import { filteredSoilReference, halfFloatUlp } from './soilCandidateReference'

/** Actual candidate bytes through the production loader, geographic sampler,
 * smoothstep reconstruction and coverage gate, with forced GPU readback.
 * Run on an idle page: no active viewer/device contention. This validates
 * sampling/encoding, not ecological accuracy or full-frame energy conservation.
 */
export async function runSoilCandidateProbe(baseUrl: string) {
  const loaded = await loadCubeTexture(baseUrl)
  const rawMeta = await (await fetch(`${baseUrl}/cube_meta.json`)).json()
  const masks = await Promise.all(Array.from({ length: 6 }, async (_, f) =>
    new Uint16Array(await (await fetch(`${baseUrl}/groups_support_${f}.bin`)).arrayBuffer())))
  const size = loaded.meta.face_size
  const cpuFaces = loaded.faces.map(densityFaceToTauFace)
  const probes: { face: number; kind: string; direction: [number, number, number] }[] = []
  for (let f = 0; f < 6; f++) {
    for (const kind of ['supported', 'unsupported', 'partial-coverage', 'composition-unavailable']) {
      search: for (let y = 2; y < size - 3; y++) for (let x = 2; x < size - 3; x++) {
        const i = x + y * size
        const a = THREE.DataUtils.fromHalfFloat(loaded.faces[f]![i*4+3]!)
        const gs = THREE.DataUtils.fromHalfFloat(masks[f]![i]!)
        const matches = kind === 'supported' ? a > .99
          : kind === 'unsupported' ? a === 0
          : kind === 'partial-coverage' ? a > .05 && a < .95 : a > 0 && gs === 0
        if (!matches) continue
        // Deliberately sub-texel, so a nearest-only probe cannot pass.
        const d = faceDirection(f, (x+.5+.23)*2/size-1, (y+.5+.61)*2/size-1)
        probes.push({ face: f, kind, direction: [d[0], d[1], d[2]] })
        break search
      }
    }
  }
  const renderer = await createRenderer(document.createElement('canvas'))
  renderer.setPixelRatio(1)
  renderer.setSize(8, 8, false)
  const target = new THREE.RenderTarget(8, 8, { type: THREE.HalfFloatType })
  const calibrationTarget = new THREE.RenderTarget(8, 8, { type: THREE.FloatType })
  // Alternating exact 0/1 ramps measure the hardware's interpolation weights
  // independently of the candidate's large coastal log-density gradients.
  const calibrationFaces = Array.from({ length: 6 }, () => {
    const data = new Uint16Array(size * size * 4)
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      data[(x+y*size)*4] = THREE.DataUtils.toHalfFloat(x%2)
      data[(x+y*size)*4+1] = THREE.DataUtils.toHalfFloat(y%2)
    }
    return new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.HalfFloatType)
  })
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
  material.fragmentNode = vec4(supportedAreaMeanNode(exp(sample.r), sample.a), sample.b, sample.a, 1)
  const scene = new THREE.Scene()
  const mesh = new THREE.Mesh(geometry, material)
  scene.add(mesh)
  const calibration = geographicCubeTexture(calibrationTexture,
    smoothstepCubeDirectionNode(direction, size)).level(float(0))
  const calibrationMaterial = new THREE.NodeMaterial()
  calibrationMaterial.toneMapped = false
  calibrationMaterial.fragmentNode = vec4(calibration.r, calibration.g, sample.r, exp(sample.r))
  const results = []
  try {
    for (const probe of probes) {
      direction.value.set(...probe.direction)
      renderer.setRenderTarget(target)
      await renderer.renderAsync(scene, camera)
      const bytes = await renderer.readRenderTargetPixelsAsync(target, 4, 4, 1, 1)
      const measured = Array.from(bytes.slice(0, 3), THREE.DataUtils.fromHalfFloat)
      const expected = filteredSoilReference(cpuFaces, size, probe.direction)
      mesh.material = calibrationMaterial
      renderer.setRenderTarget(calibrationTarget)
      await renderer.renderAsync(scene, camera)
      const rawCalibration = await renderer.readRenderTargetPixelsAsync(calibrationTarget, 4, 4, 1, 1)
      const calibrated = Array.from(rawCalibration.slice(0, 4))
      const weights: [number, number] = [expected.x0 % 2 ? 1-calibrated[0]! : calibrated[0]!,
        expected.y0 % 2 ? 1-calibrated[1]! : calibrated[1]!]
      const hardwareExpected = filteredSoilReference(cpuFaces, size, probe.direction, weights)
      mesh.material = material
      const idealReferencePass = Math.abs(measured[0]! - expected.tau) <= Math.max(5e-7, expected.tau*.005)
        && Math.abs(measured[1]! - expected.cv) < .001
        && Math.abs(measured[2]! - expected.support) < .001
      // Independently measured 0/1 texture weights on this GPU are quantized
      // in 1/256 increments; the filtered log value is also binary16 rounded.
      // Keep the original ideal comparison visible; do not replace it with a
      // blanket percentage tolerance. Test each source of rounding separately.
      const weightPass = weights.every((w, i) => Math.abs(w-expected.weights[i]!) <= 1/512 + 2e-5)
      const logPass = Math.abs(calibrated[2]! - hardwareExpected.logTau)
        <= halfFloatUlp(hardwareExpected.logTau)/2 + 1e-5
      const decodedTau = measured[2]! > 0 ? Math.exp(calibrated[2]!) : 0
      const expPass = Math.abs(calibrated[3]! - Math.exp(calibrated[2]!)) <= Math.exp(calibrated[2]!)*2e-6
      // Float32 control isolates exp from half-float readback: the latter was
      // observed rounding positive tau toward zero (up to one ULP, not 1/2).
      const outputPass = Math.abs(measured[0]! - decodedTau) <= halfFloatUlp(decodedTau) + decodedTau*2e-6
        && Math.abs(measured[1]! - hardwareExpected.cv) <= halfFloatUlp(hardwareExpected.cv)/2 + 1e-6
        && Math.abs(measured[2]! - hardwareExpected.support) <= halfFloatUlp(hardwareExpected.support)/2 + 1e-6
      const pass = weightPass && logPass && expPass && outputPass
      results.push({ face: probe.face, kind: probe.kind, expected, measured, pass,
        idealReferencePass, weightPass, logPass, expPass, outputPass, calibrated, hardwareExpected })
    }
    // Negative control: G must not influence density or CV.
    const baseline = results[results.length-1]!.measured
    for (const face of loaded.texture.images as THREE.DataTexture[]) {
      const data = face.image.data as Uint16Array
      for (let i = 1; i < data.length; i += 4) data[i] = THREE.DataUtils.toHalfFloat(100)
      face.needsUpdate = true
    }
    loaded.texture.needsUpdate = true
    renderer.setRenderTarget(target)
    await renderer.renderAsync(scene, camera)
    const mutated = Array.from((await renderer.readRenderTargetPixelsAsync(target, 4, 4, 1, 1)).slice(0, 3), THREE.DataUtils.fromHalfFloat)
    const gIndependent = mutated.every((value, i) => value === baseline[i])
    return { pass: probes.length >= 12 && results.every(r => r.pass) && gIndependent,
      source: baseUrl, schema: rawMeta.schema, faceSize: size,
      primaryHashes: rawMeta.face_sha256, groupSupportHashes: rawMeta.group_support.sha256,
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
