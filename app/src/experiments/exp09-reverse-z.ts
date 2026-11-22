import * as THREE from 'three/webgpu'
import { createRenderer } from '../core/Renderer'
import type { ExpResult } from './types'

/**
 * Exp-9: reverse-Z, single frustum, near = 1 m, far = 1e9 m.
 *
 * REPRODUCED HANG (found via a page that had sat on "Running four
 * experiments…" for 110+ seconds with no console output at all): this is
 * the FIRST of the four Session-1 runners (see report.ts's `runners` order),
 * so main.ts's per-experiment progress logging showed nothing past
 * "exp-09-reverse-z: starting…" - not because this file is instrumented
 * differently, but because it genuinely never returned. Confirmed live: with
 * a second tab open on `viewer.html` (the interactive viewer, whose
 * `setAnimationLoop` never stops rendering) at the same time, this
 * function's very first `renderer.renderAsync`/`readRenderTargetPixelsAsync`
 * round-trip stayed permanently unresolved - even a brand-new, unrelated
 * `navigator.gpu.requestAdapter()` call from DevTools on the SAME page hung
 * too, meaning the GPU process itself was contended, not just this
 * function's own promise chain. Closing the viewer tab let this function
 * complete in ~0.2s. Do not run this harness (`index.html`) at the same
 * time as an open `viewer.html` tab. `report.ts`'s `RUNNER_TIMEOUT_MS` is
 * the backstop for whenever that happens anyway - it turns the hang into a
 * loud, specific failure instead of the whole page going silent forever,
 * but it cannot make the contention itself not happen.
 *
 * Two coplanar-ish quads are placed at each altitude on the descent path,
 * separated by a small gap along the view axis. Under a conventional depth
 * buffer at these ranges the nearer quad loses to precision loss and the
 * pair flickers; under reverse-Z the nearer quad must win at every altitude.
 *
 * The quads are scaled to `alt * 0.4` so their angular width stays constant
 * (~0.4 rad) regardless of distance, which works out to roughly 116 px on
 * screen at this camera's 50-degree FOV and 256 px canvas. This is load-
 * bearing, not cosmetic: an earlier version scaled to `max(1, alt / 1000)`,
 * which held the quad's *angular* size constant at only 0.001 rad — about a
 * third of a pixel — for every altitude at or above 1000 m. Sub-pixel
 * geometry samples as background, so seven of the ten altitudes probed
 * nothing and were correctly (but uselessly) reported inconclusive. At ~116
 * px the quads are always resolvable, so a pass/fail actually reflects the
 * depth buffer rather than a rasterization accident.
 *
 * Two separations are probed per altitude:
 * - Primary (gates `pass`): separation proportional to altitude
 *   (`alt * 1e-4`), i.e. a constant *relative* depth gap — the case that
 *   matters for a descent where the near/far ratio, not the absolute gap,
 *   is what precision has to resolve.
 * - Secondary (`data.fixedSeparationResults`, informational only): a
 *   constant 1 cm separation regardless of altitude. This is strictly
 *   harder at long range — resolving 1 cm at 1e7 m is ~1e-9 relative,
 *   beyond f32 even with reverse-Z — and is recorded as a known limit of
 *   the descent, not folded into the pass criterion.
 *
 * Readback: renders into an explicit off-screen `THREE.RenderTarget` rather
 * than the canvas's default framebuffer. `readRenderTargetPixelsAsync` requires
 * a real `RenderTarget` — its WebGPU backend implementation indexes straight
 * into `renderTarget.textures[textureIndex]`
 * (node_modules/three/src/renderers/common/Renderer.js:3063-3067), so a `null`
 * target throws. A canvas-based fallback isn't viable either: once a canvas has
 * been handed to `WebGPURenderer` as its context, `canvas.getContext('2d')`
 * cannot also succeed on that same canvas in a browser, so 2D pixel readback
 * of the WebGPU default framebuffer is not an option here.
 */

// Below this level on both channels, nothing was drawn under the sampled
// pixel at all (background). That's a probe failure, not evidence that red
// beat green — reported separately so it can't masquerade as a depth-
// precision failure.
const BACKGROUND_LEVEL = 8

// Secondary probe's fixed depth gap, independent of altitude (see doc comment above).
const FIXED_SEPARATION_METERS = 0.01

// Off-screen probe target dimensions. Sample coordinate is derived from
// these (below), not hardcoded, to keep the two coupled — see the readback
// note in the doc comment above `createRenderer` for why that coupling
// matters.
const PROBE_W = 256
const PROBE_H = 256

type Probe = 'near' | 'far' | 'inconclusive'

function classify(red: number, green: number): Probe {
  if (red < BACKGROUND_LEVEL && green < BACKGROUND_LEVEL) return 'inconclusive'
  return green > red ? 'near' : 'far'
}

export async function runExp09(): Promise<ExpResult> {
  const canvas = document.createElement('canvas')
  canvas.width = PROBE_W
  canvas.height = PROBE_H

  let renderer: THREE.WebGPURenderer | undefined
  let target: THREE.RenderTarget | undefined
  let near: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial> | undefined
  let far: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial> | undefined

  // Lowest altitude is 2 m, not 1 m: at near=1 the near quad's centre would
  // sit exactly on the clip plane, so it could be clipped and read back as
  // background — a probe failure that has nothing to do with depth
  // precision. 2 m keeps the near quad comfortably inside the frustum while
  // still exercising the extreme near end of the 1 m..1e9 m range.
  const altitudes = [1e7, 1e6, 1e5, 192000, 32000, 10000, 1000, 100, 10, 2]
  const failures: number[] = []
  const inconclusive: number[] = []
  const fixedFailures: number[] = []
  const fixedInconclusive: number[] = []

  try {
    renderer = await createRenderer(canvas)

    // createRenderer sizes the renderer to the window and mutates the canvas
    // (e.g. 256x256 -> 3024x1610 at dpr 2). Put it back to the probe's
    // intended dimensions before deriving the render target from it, or the
    // target and the sampled pixel coordinate silently disagree about where
    // the centre is and every probe reads background.
    renderer.setPixelRatio(1)
    renderer.setSize(PROBE_W, PROBE_H, false)

    target = new THREE.RenderTarget(PROBE_W, PROBE_H)

    const camera = new THREE.PerspectiveCamera(50, 1, 1, 1e9)
    const scene = new THREE.Scene()

    near = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    )
    far = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ color: 0xff0000 }),
    )
    scene.add(near, far)

    renderer.setRenderTarget(target)

    for (const alt of altitudes) {
      const s = alt * 0.4
      near.scale.setScalar(s)
      far.scale.setScalar(s)
      near.position.set(0, 0, -alt)
      camera.position.set(0, 0, 0)
      camera.lookAt(0, 0, -1)
      camera.updateMatrixWorld()

      // Primary probe: separation proportional to distance.
      const sep = alt * 1e-4
      far.position.set(0, 0, -alt - sep)
      await renderer.renderAsync(scene, camera)
      const px = await renderer.readRenderTargetPixelsAsync(target, PROBE_W >> 1, PROBE_H >> 1, 1, 1)
      const result = classify(px[0]!, px[1]!)
      if (result === 'inconclusive') inconclusive.push(alt)
      else if (result === 'far') failures.push(alt)

      // Secondary probe: fixed 1 cm separation, informational only.
      far.position.set(0, 0, -alt - FIXED_SEPARATION_METERS)
      await renderer.renderAsync(scene, camera)
      const pxFixed = await renderer.readRenderTargetPixelsAsync(target, PROBE_W >> 1, PROBE_H >> 1, 1, 1)
      const fixedResult = classify(pxFixed[0]!, pxFixed[1]!)
      if (fixedResult === 'inconclusive') fixedInconclusive.push(alt)
      else if (fixedResult === 'far') fixedFailures.push(alt)
    }
  } finally {
    renderer?.setRenderTarget(null)
    target?.dispose()
    near?.geometry.dispose()
    near?.material.dispose()
    far?.geometry.dispose()
    far?.material.dispose()
    renderer?.dispose()
  }

  const pass = failures.length === 0 && inconclusive.length === 0
  const detailParts: string[] = []
  if (failures.length > 0) {
    detailParts.push(`Depth precision failed at altitudes: ${failures.join(', ')} m.`)
  }
  if (inconclusive.length > 0) {
    detailParts.push(
      `Probe was inconclusive (no geometry visible at the sampled pixel, not a red-wins result) ` +
      `at altitudes: ${inconclusive.join(', ')} m.`,
    )
  }
  const detail = pass
    ? `No z-fighting across ${altitudes.length} altitudes from 1e7 m to 2 m with near=1 far=1e9 ` +
      `(relative separation). Fixed 1 cm separation failed at: ${fixedFailures.join(', ') || 'none'} m.`
    : detailParts.join(' ')

  return {
    id: 'exp-09-reverse-z',
    pass,
    detail,
    data: {
      altitudes,
      failures,
      inconclusive,
      near: 1,
      far: 1e9,
      reversedDepthBuffer: true,
      fixedSeparationResults: {
        separationMeters: FIXED_SEPARATION_METERS,
        failures: fixedFailures,
        inconclusive: fixedInconclusive,
        passCount: altitudes.length - fixedFailures.length - fixedInconclusive.length,
        total: altitudes.length,
        note: 'Stricter secondary probe with a constant 1 cm gap regardless of altitude. ' +
          'Informational only — does not gate exp-09 pass/fail.',
      },
    },
  }
}
