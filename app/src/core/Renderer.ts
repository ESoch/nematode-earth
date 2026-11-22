import * as THREE from 'three/webgpu'

/**
 * Reverse-Z with a SINGLE frustum.
 *
 * three.js's WebGPU backend implements this today: the renderer accepts
 * reversedDepthBuffer, sets camera._reversedDepth, flips getClearDepth(), and
 * WebGPUPipelineUtils maps material.depthFunc through ReversedDepthFuncs into
 * depthStencil.depthCompare.
 *
 * Do NOT use logarithmicDepthBuffer: it writes frag_depth and kills early-Z.
 * The Cesium-style multi-frustum near/far camera pair is not needed. Spec section 7.2.
 *
 * WARNING: this sizes the renderer to the window and, via `setSize`,
 * MUTATES the canvas element's `width`/`height` to match (e.g. a
 * 256x256 canvas becomes 3024x1610 at devicePixelRatio 2). Any caller doing
 * an offscreen probe — rendering into an explicit `THREE.RenderTarget` sized
 * from `canvas.width`/`canvas.height`, or sampling a pixel coordinate chosen
 * before calling this — must call `renderer.setPixelRatio(1)` and
 * `renderer.setSize(w, h, false)` afterwards to put the canvas back to the
 * probe's intended dimensions before deriving anything from it. Exp-1,
 * Exp-4, and Exp-9 all follow exactly this pattern: each defines its own
 * explicit probe dimensions (`PROBE_W`/`PROBE_H` or equivalent), calls
 * `setPixelRatio(1)` + `setSize` on those dimensions immediately after
 * `createRenderer` returns, and derives both its `RenderTarget` and its
 * sampled pixel coordinate from those same constants — never by reading
 * `canvas.width`/`canvas.height` back after this call. (Exp-2 has no
 * `RenderTarget` or pixel readback at all, so it is unaffected regardless.)
 */
export interface RendererOptions {
  /**
   * Enables GPU pass timing via the `timestamp-query` feature. Self-gates to
   * a no-op when the adapter lacks the feature (three.js ANDs this flag with
   * `hasFeature('timestamp-query')` internally), so it is always safe to
   * request. Off by default: the query pool holds a small persistent buffer
   * that callers with no per-pass timing need shouldn't pay for.
   */
  trackTimestamp?: boolean
}

export async function createRenderer(
  canvas: HTMLCanvasElement,
  opts: RendererOptions = {},
): Promise<THREE.WebGPURenderer> {
  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: true,
    reversedDepthBuffer: true,
    trackTimestamp: opts.trackTimestamp ?? false,
  })
  await renderer.init()
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(window.innerWidth, window.innerHeight)
  return renderer
}
