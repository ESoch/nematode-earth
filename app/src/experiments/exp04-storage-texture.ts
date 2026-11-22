import * as THREE from 'three/webgpu'
import { Fn, texture, uvec2, vec4, instanceIndex, textureStore } from 'three/tsl'
import { createRenderer } from '../core/Renderer'
import type { ExpResult } from './types'

const SIZE = 8
const MAGIC = 0.75

// Off-screen probe target dimensions. Sample coordinate is derived from
// these (below), not hardcoded, to keep the two coupled - see the readback
// note in the doc comment above `createRenderer` for why that coupling
// matters.
const PROBE_W = 64
const PROBE_H = 64

/**
 * Exp-4: three.js #33795 - is a compute-written StorageTexture visible to the
 * render pass in the SAME frame?
 *
 * Compute writes a known constant into every texel; a fullscreen quad samples
 * it and we read back the centre pixel. three.js submits the compute encoder
 * and the render encoder to the same GPUQueue in order, and storage textures
 * get TEXTURE_BINDING | STORAGE_BINDING | COPY_DST | COPY_SRC, so this should
 * work on the renderer.render path even though the filed repro uses
 * RenderPipeline.
 *
 * Compute node follows the official webgpu_compute_texture pattern: the
 * storage texture is a Fn() parameter, not a closure capture, and the write
 * is chained through .toWriteOnly() so the backend knows the access mode.
 *
 * Readback: renders into an explicit off-screen `THREE.RenderTarget` rather
 * than the canvas's default framebuffer, the same pattern established in
 * exp09-reverse-z.ts. `readRenderTargetPixelsAsync` requires a real
 * `RenderTarget` - its WebGPU backend implementation indexes straight into
 * `renderTarget.textures[textureIndex]`, so a `null` target throws. A
 * canvas-based fallback isn't viable either: once a canvas has been handed to
 * `WebGPURenderer` as its context, `canvas.getContext('2d')` cannot also
 * succeed on that same canvas in a browser.
 *
 * `createRenderer` sizes the renderer to the window and mutates the canvas
 * (e.g. 64x64 -> 3024x1610 at dpr 2). `setPixelRatio(1)` + `setSize(PROBE_W,
 * PROBE_H, false)` immediately after put it back before the RenderTarget and
 * sample coordinate are derived from `PROBE_W`/`PROBE_H` - the same fix
 * applied in exp09-reverse-z.ts and exp01-march-cost.ts. See the doc comment
 * above `createRenderer` in Renderer.ts.
 */
export async function runExp04(): Promise<ExpResult> {
  const canvas = document.createElement('canvas')
  canvas.width = PROBE_W
  canvas.height = PROBE_H

  let renderer: THREE.WebGPURenderer | undefined
  let target: THREE.RenderTarget | undefined
  let storage: THREE.StorageTexture | undefined
  let geometry: THREE.PlaneGeometry | undefined
  let material: THREE.MeshBasicNodeMaterial | undefined

  // A real readback always returns one RGBA pixel. If it doesn't, the probe
  // infrastructure failed - report that plainly rather than letting a -1
  // sentinel silently compare as "not visible" and masquerade as the hazard.
  let readbackOk = true
  let sampled = -1

  try {
    renderer = await createRenderer(canvas)
    renderer.setPixelRatio(1)
    renderer.setSize(PROBE_W, PROBE_H, false)

    target = new THREE.RenderTarget(PROBE_W, PROBE_H)

    storage = new THREE.StorageTexture(SIZE, SIZE)
    storage.type = THREE.FloatType

    const computeTexture = Fn(({ storageTexture }: { storageTexture: THREE.StorageTexture }) => {
      const posX = instanceIndex.mod(SIZE)
      const posY = instanceIndex.div(SIZE)
      textureStore(storageTexture, uvec2(posX, posY), vec4(MAGIC, 0.0, 0.0, 1.0)).toWriteOnly()
    })
    const computeNode = computeTexture({ storageTexture: storage }).compute(SIZE * SIZE)

    const scene = new THREE.Scene()
    geometry = new THREE.PlaneGeometry(2, 2)
    material = new THREE.MeshBasicNodeMaterial()
    material.colorNode = texture(storage)
    scene.add(new THREE.Mesh(geometry, material))
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)

    renderer.setRenderTarget(target)
    await renderer.compileAsync(scene, camera)

    // Compute BEFORE render, same frame - the ordering the issue questions.
    await renderer.computeAsync(computeNode)
    await renderer.renderAsync(scene, camera)

    const px = await renderer.readRenderTargetPixelsAsync(target, PROBE_W >> 1, PROBE_H >> 1, 1, 1)
    if (px.length < 1) {
      readbackOk = false
    } else {
      sampled = px[0]! / 255
    }
  } finally {
    renderer?.setRenderTarget(null)
    target?.dispose()
    storage?.dispose()
    geometry?.dispose()
    material?.dispose()
    renderer?.dispose()
  }

  const expected = MAGIC

  if (!readbackOk) {
    return {
      id: 'exp-04-storage-texture',
      pass: false,
      detail: 'Readback failed: readRenderTargetPixelsAsync returned no pixel data.',
      data: { size: SIZE, order: 'compute-then-render' },
    }
  }

  const ok = Math.abs(sampled - expected) < 0.06

  return {
    id: 'exp-04-storage-texture',
    pass: ok,
    detail: ok
      ? `Compute-written storage texture IS visible to render in the same frame (sampled ${sampled.toFixed(3)} vs expected ${expected}). Issue #33795 is RenderPipeline-specific; design nothing around it.`
      : `Same-frame read returned ${sampled.toFixed(3)}, expected ~${expected}. Insert convertToTexture() and benchmark its per-frame cost before relying on any GPU-driven field.`,
    data: { sampled, expected, size: SIZE, order: 'compute-then-render' },
  }
}
