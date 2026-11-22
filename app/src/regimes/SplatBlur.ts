import * as THREE from 'three/webgpu'
import {
  Fn, Loop, vec2, vec3, vec4, float, int, uniformArray, instanceIndex, varying, positionLocal,
  cameraViewMatrix, cameraProjectionMatrix, viewportSize, texture, uv, floor, mix, clamp, storage,
} from 'three/tsl'
import type { ParticleSystem } from './Particles'
import { gaussianTaps } from '../math/splat'

/**
 * The splat draw pass: deposit + resolve (spec 7.7), using math/splat.ts's
 * discrete-domain kernel directly rather than re-deriving it here.
 *
 * DEPOSIT (`buildDepositMaterial`): each instance draws as a quad spanning
 * the 2x2 PIXELS whose centres bracket the instance's own screen position -
 * `basePixel` to `basePixel+2` in each axis, so all four pixel centres
 * (at `basePixel+0.5` and `basePixel+1.5`) sit STRICTLY INTERIOR to the
 * quad and all four rasterise. This replaced an earlier, WRONG version
 * whose four vertices sat exactly ON the four pixel centres (a 1x1-pixel
 * quad, corners AT its own sample points) - team lead's deposit trace
 * measured that version giving `nonZeroCount` of 0 or 1 (never 4),
 * non-deterministic run to run, tied to whichever way a fill-rule tie at
 * an exactly-on-the-boundary sample point happened to break. The fix
 * proposed first (shift the quad by translating its corner offset) was
 * tested and refuted - see `buildDepositMaterialCornerVariantForTesting`
 * below - because a 1x1 quad has this problem regardless of where it is
 * translated; only enlarging it to 2x2 puts every sample point safely
 * inside.
 *
 * Because the quad is now BIGGER than one pixel, a per-VERTEX bilinear
 * weight (the old design: each of the 4 vertices carries its own corner's
 * weight, interpolated into a varying) no longer works - a fragment
 * anywhere in a 2x2 quad receives a BLEND of all four vertices' varyings,
 * not any single corner's own value. Weights are computed PER-FRAGMENT
 * instead: `cornerSpaceVarying` (`corner*2`, carried through unchanged
 * from vertex to fragment by the rasterizer's own interpolation) equals,
 * at any fragment, that fragment's own screen position minus `basePixel`
 * - `floor()` of it recovers which of the four target pixels this
 * fragment is, and the SAME bilinear formula as before is then evaluated
 * there instead of at the vertex. `fracVarying`/`fluxVarying` carry the
 * per-INSTANCE (not per-corner) quantities through unchanged, since they
 * are identical at all four vertices and therefore trivially correct
 * under interpolation.
 *
 * This needs `vertexNode`, not `positionNode`: `positionNode` still goes
 * through the automatic model-view-projection pipeline afterward (see
 * NodeMaterial.d.ts's own `positionNode` doc:
 * "define local vertex positions... `material.positionNode = positionLocal.add(displace)`",
 * i.e. still local space, still auto-transformed), whereas `vertexNode`
 * replaces vertex-stage logic entirely and is expected to output a final
 * CLIP-SPACE position - confirmed against three.js's own shipped
 * `nodes/utils/SpriteUtils.js`'s `billboarding()`, whose doc example is
 * literally `material.vertexNode = billboarding()` and whose body ends
 * `return cameraProjectionMatrix.mul(modelViewMatrix).mul(positionLocal)`.
 *
 * The corner-snapping works in NDC-then-implicit-pixel-grid space rather
 * than literal screen pixel coordinates, specifically so it never has to
 * assume or verify which way this renderer's NDC-to-pixel Y axis runs:
 * `pixelUnits = (ndc*0.5+0.5) * viewportSize` measures position in a
 * REGULAR GRID whose spacing is one device pixel, without ever comparing
 * that grid's orientation to an external convention. `cornerSpaceVarying`
 * stays inside this SAME self-consistent grid rather than reaching for
 * `screenCoordinate` (WGSL's `@builtin(position)`), which three.js
 * documents as potentially Y-flipped relative to it depending on render
 * target - mixing the two conventions in one shader is exactly the kind
 * of bug this design has avoided since Session 4. The RESOLVE pass reads
 * back the SAME accumulation texture texel-by-texel (adjacent texel
 * indices, not NDC positions), so it inherits whatever orientation the
 * deposit pass used automatically - the two passes only need to agree with
 * EACH OTHER, never with a convention neither of them has to name.
 *
 * RESOLVE (`buildResolveMaterial`): SEPARABLE two-pass (H then V), per
 * math/splat.ts's own fill-rate arithmetic - 2*taps samples instead of
 * taps^2 (42 vs 441 at sigma_pix=3's 21-tap radius), and load-bearing, not
 * just an optimization: a first version summed all `taps x taps` samples as
 * ONE flat unrolled expression, and WGSL's parser hit its maximum
 * recursion depth on that many nested parenthesised terms - the pipeline
 * failed to compile at all. Separable halves the per-pass tap count AND
 * keeps each pass's own sum shallow. That sum is a REAL TSL `Loop` over a
 * `uniformArray(taps, 'float')`, accumulated into a `.toVar()` rather than
 * chained `.add()` calls - fixed WGSL size regardless of tap count, so a
 * later change to `SIGMA_PIX_MAX`/`MAX_TAPS` (math/lod.ts, math/splat.ts)
 * cannot silently reintroduce the same recursion-depth failure.
 *
 * THE BEHIND-CAMERA GUARD (`buildDepositMaterial`'s `behindCamera`/
 * `degenerate`): a `clip.w <= 0` instance has no valid perspective divide.
 * The original guard substituted `safeW = 1e6` and fed it straight into
 * the returned clip-space position - since a record's `clip.xy` is only
 * O(10-1000) at these distances, `clip.xy / 1e6` lands within a fraction
 * of a pixel of NDC (0,0), the screen CENTRE, not the "degenerate,
 * off-screen" position the comment claimed. Every behind-camera particle
 * piled onto the same handful of centre pixels and accumulated additively
 * - measured directly (docs/superpowers/results/
 * 2026-08-17-regime2-deposit-investigation.md): 100% of ~100,000 accepted
 * particles behind the camera at one altitude produced six non-zero
 * deposit pixels, all at the exact screen centre. The fix returns a fixed
 * `vec4(2,2,2,1)` for EVERY corner of a behind-camera instance's quad
 * instead: identical across all four corners, so the quad has zero area
 * (no fragment can be strictly interior to a degenerate triangle) AND
 * outside the clip volume on x, y, AND z (|2| > w=1 fails all three
 * clip-plane tests) - two independent reasons it produces no fragments,
 * not one. `safeW` still guards the (now-unused-when-behind-camera)
 * intermediate pixel/frac math from dividing by a non-positive w; it is no
 * longer what determines the returned position.
 */

type FloatNode = THREE.Node<'float'>
type Vec2Node = THREE.Node<'vec2'>
type Vec4Node = THREE.Node<'vec4'>

/**
 * Only these two fields, not the whole `ParticleSystem` - lets a caller
 * hand this the real particle system OR a minimal stand-in with a single,
 * manually-written record (a known position + known flux) for tracing the
 * deposit/resolve/composite chain with a controlled input. Deliberately
 * NOT duplicated logic for that trace: same function, same vertex math,
 * same blend state, just a different `records`/`compFactorAttr` source -
 * a hand-copied second implementation would risk testing a DIFFERENT
 * shader than the one actually drawing the real particles.
 */
export type DepositSource = Pick<ParticleSystem, 'records' | 'compFactorAttr'>

function buildDepositBlendState(material: THREE.NodeMaterial): void {
  material.transparent = true
  // NOT AdditiveBlending: that is SrcAlpha/One, which reads an alpha
  // channel this material's target (makeAccumTarget's RedFormat, below)
  // does not have - the WebGPU backend refused to create the pipeline at
  // all ("Color blending ... is reading alpha but it is missing from
  // fragment output"). CustomBlending with One/One is straight additive
  // accumulation with no alpha read - also the semantically correct
  // choice, not just the one that compiles: this is summing radiance
  // (flux is already carried in the value itself, per instance), not
  // alpha-compositing translucent layers.
  material.blending = THREE.CustomBlending
  material.blendSrc = THREE.OneFactor
  material.blendDst = THREE.OneFactor
  material.blendEquation = THREE.AddEquation
  material.depthWrite = false
  material.depthTest = false
  material.side = THREE.DoubleSide
}

/**
 * The real, shipped deposit material - see this file's module doc for the
 * full history (a 1x1-pixel-quad predecessor measured broken, a
 * translation fix tried and refuted, this 2x2-quad-plus-per-fragment-
 * weight version confirmed by the same harness: `nonZeroCount==4`,
 * `sum==1.0`, at two independently solved sub-pixel positions).
 */
export function buildDepositMaterial(particles: DepositSource): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial()
  buildDepositBlendState(material)

  // Per-INSTANCE quantities (identical at all four vertices, so trivially
  // correct wherever the rasterizer interpolates them) and the per-
  // FRAGMENT quantity that is not: cornerSpaceVarying is corner*2 carried
  // through unchanged, which the rasterizer interpolates using the exact
  // same weights it uses for the vertex positions themselves - so at any
  // fragment inside the quad, its value equals that fragment's own screen
  // position minus basePixel. See the module doc for why this, not
  // screenCoordinate, is what stays inside the shader's own self-
  // consistent pixel-grid convention.
  const cornerSpaceVarying = varying(vec2(0, 0), 'splatCornerSpace')
  const fracVarying = varying(vec2(0, 0), 'splatFrac')
  const fluxVarying = varying(float(0), 'splatFlux')
  // Budget-clamp energy compensation - see Particles.ts's compFactorBuf
  // declaration. A plain storage read, not a uniform: the value is
  // GPU-computed (finalizeKernel, previous frame) and this is the exact
  // same "storage buffer written by compute, read by the vertex stage"
  // pattern `particles.records` below already uses successfully.
  const compFactor = storage(particles.compFactorAttr, 'float', 1).element(0)

  const vertexFn = Fn((): Vec4Node => {
    const rec = particles.records.element(instanceIndex)
    const instPos = rec.xyz
    const flux = rec.w.mul(compFactor)

    const viewPos = cameraViewMatrix.mul(vec4(instPos, 1.0))
    const clip = cameraProjectionMatrix.mul(viewPos)
    const behindCamera = clip.w.lessThanEqual(0.0)
    // safeW only keeps the pixel/frac math below well-defined (no division
    // by a non-positive w) for a behind-camera instance - it is NOT what
    // makes that instance disappear anymore. It used to be: the return
    // statement fed `safeW` straight into the final clip-space position,
    // so `ndc = clip.xy / 1e6` landed near NDC (0,0) - the screen CENTRE,
    // not "degenerate, off-screen" as the old comment here claimed. Every
    // behind-camera particle piled onto the same centre pixel (see the
    // investigation this fixes: docs/superpowers/results/
    // 2026-08-17-regime2-deposit-investigation.md). The actual off-screen
    // push is `degenerate` below, applied to the RETURNED position only.
    const safeW = behindCamera.select(float(1e6), clip.w)

    const ndc = clip.xy.div(safeW)
    const pixelUnits: Vec2Node = ndc.add(1.0).mul(0.5).mul(viewportSize)

    // positionLocal.xy is this vertex's OWN corner index within the quad
    // (0,0)/(1,0)/(1,1)/(0,1) - see Particles.ts's buildQuadGeometry.
    const corner = positionLocal.xy
    const basePixel = floor(pixelUnits.sub(0.5))
    const frac = pixelUnits.sub(0.5).sub(basePixel)
    // 2x2-PIXEL quad, not 1x1: corner*2 spans (0,0)..(2,2), so all four
    // pixel centres (basePixel+0.5 and basePixel+1.5) sit strictly inside
    // it rather than exactly on its boundary. See module doc for why a
    // 1x1 quad (the old `.add(corner).add(cornerOffset)`) could never
    // fix this by translation alone.
    const cornerOffset2x2 = corner.mul(2.0)
    const cornerPixel = basePixel.add(cornerOffset2x2)
    const cornerNdc = cornerPixel.div(viewportSize).mul(2.0).sub(1.0)

    cornerSpaceVarying.assign(cornerOffset2x2)
    fracVarying.assign(frac)
    fluxVarying.assign(flux)

    // A behind-camera instance is pushed to a position IDENTICAL across all
    // four of its corners (zero area - no fragment can be strictly inside
    // a degenerate triangle, regardless of fill rule) AND outside the clip
    // volume on every axis (|2|>w=1 fails the x, y, AND z clip-plane tests)
    // - two independent guarantees, not one. `positionLocal`/`corner` is
    // deliberately absent from this branch: unlike `normal` below, every
    // corner of the quad must map to the SAME point, not four points that
    // happen to share a plane.
    const degenerate = vec4(2.0, 2.0, 2.0, 1.0)
    const normal = vec4(cornerNdc.mul(safeW), clip.z, safeW)
    return behindCamera.select(degenerate, normal)
  })

  const fragmentFn = Fn((): Vec4Node => {
    // Which of the four target pixels THIS fragment is - see module doc
    // for why cornerSpaceVarying's interpolated value equals this
    // fragment's own (screen position - basePixel).
    const localCorner = floor(cornerSpaceVarying)

    // Same bilinearWeights(fx,fy) formula as before
    // (math/splat.ts) - evaluated per-fragment now, using this fragment's
    // OWN localCorner rather than a vertex-supplied one.
    const wx = mix(float(1.0).sub(fracVarying.x), fracVarying.x, localCorner.x)
    const wy = mix(float(1.0).sub(fracVarying.y), fracVarying.y, localCorner.y)
    return vec4(fluxVarying.mul(wx).mul(wy), 0.0, 0.0, 1.0)
  })

  material.vertexNode = vertexFn()
  material.fragmentNode = fragmentFn()
  return material
}

/**
 * TESTED AND REFUTED - kept for the record, not called by
 * `buildSplatPipeline` or anything else in the real render path (grepped
 * to confirm the only caller is viewer.ts's deposit coverage test). Do
 * NOT re-propose this fix: a quad whose four vertices sit at
 * `basePixel + corner + cornerOffset` is a 1x1-PIXEL SQUARE regardless of
 * `cornerOffset` (corner in {0,1} spans exactly one pixel unit either
 * way) - translating it (this variant's `cornerOffset=0`, vs the old
 * broken `0.5`) only moves WHICH pixel-centre sample points sit on its
 * boundary, never makes any of them interior. Measured by
 * `logDepositCoverageTest`: `nonZeroCount` stayed at exactly 1 (not 4) at
 * both (0.5,0.5) and (0.25,0.75), the same as the original. The real fix
 * (`buildDepositMaterial` above) changes the quad's SIZE to 2x2 pixels
 * instead, plus per-fragment (not per-vertex) weights.
 */
export function buildDepositMaterialCornerVariantForTesting(particles: DepositSource): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial()
  buildDepositBlendState(material)

  const weightedFluxVarying = varying(float(0), 'splatWeightedFluxCornerVariant')
  const compFactor = storage(particles.compFactorAttr, 'float', 1).element(0)

  const vertexFn = Fn((): Vec4Node => {
    const rec = particles.records.element(instanceIndex)
    const instPos = rec.xyz
    const flux = rec.w.mul(compFactor)

    const viewPos = cameraViewMatrix.mul(vec4(instPos, 1.0))
    const clip = cameraProjectionMatrix.mul(viewPos)
    const safeW = clip.w.lessThanEqual(0.0).select(float(1e6), clip.w)

    const ndc = clip.xy.div(safeW)
    const pixelUnits: Vec2Node = ndc.add(1.0).mul(0.5).mul(viewportSize)

    const corner = positionLocal.xy
    const basePixel = floor(pixelUnits.sub(0.5))
    const frac = pixelUnits.sub(0.5).sub(basePixel)
    const cornerOffset = 0 // the refuted fix: was 0.5 (pixel centres) in the original broken version
    const cornerPixel = basePixel.add(corner).add(cornerOffset)
    const cornerNdc = cornerPixel.div(viewportSize).mul(2.0).sub(1.0)

    const wx = mix(float(1.0).sub(frac.x), frac.x, corner.x)
    const wy = mix(float(1.0).sub(frac.y), frac.y, corner.y)
    weightedFluxVarying.assign(flux.mul(wx).mul(wy))

    return vec4(cornerNdc.mul(safeW), clip.z, safeW)
  })

  material.vertexNode = vertexFn()
  material.fragmentNode = vec4(weightedFluxVarying, 0.0, 0.0, 1.0)
  return material
}

/**
 * One 1D pass of the separable resolve: `gaussianTaps(sigmaPix)` samples
 * along `direction` (horizontal or vertical), each already discretely
 * normalized (math/splat.ts), so the sum over all taps is exactly 1 with
 * no separate normalization step. Two of these (H then V, the second
 * reading the first's output) reproduce the full 2D kernel exactly - a
 * separable Gaussian is mathematically the outer product of its two 1D
 * factors, not an approximation of the 2D form.
 */
export function buildResolveMaterial(
  sourceTexture: THREE.Texture,
  sigmaPix: number,
  direction: 'horizontal' | 'vertical',
): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial()
  material.transparent = false
  material.depthWrite = false
  material.depthTest = false

  const taps = gaussianTaps(sigmaPix)
  const radius = (taps.length - 1) / 2
  const weights = uniformArray<'float'>(taps, 'float')
  const dir: Vec2Node = direction === 'horizontal' ? vec2(1, 0) : vec2(0, 1)

  const shade = Fn((): Vec4Node => {
    const texelSize = vec2(float(1.0).div(viewportSize.x), float(1.0).div(viewportSize.y))
    const base = uv()
    const sum: FloatNode = float(0).toVar()

    // A real GPU Loop, not a JS-unrolled chain of .add() calls: the
    // generated WGSL is a `for` statement with a `sum = sum + ...;`
    // assignment inside it, not one deeply nested expression - see this
    // file's doc comment for why that distinction is load-bearing, not
    // cosmetic.
    Loop({ start: int(0), end: int(taps.length), type: 'int' }, ({ i }) => {
      const w = weights.element(i)
      const offsetTaps = float(i).sub(radius)
      const offset = texelSize.mul(dir).mul(offsetTaps)
      const sampleUv = clamp(base.add(offset), 0.0, 1.0)
      // Explicit LOD: this is a straight-line full-screen pass with no
      // non-uniform control flow, so implicit-LOD sampling would be legal
      // here (unlike Shell.ts's march loop) - explicit .level(0) is used
      // anyway for consistency with the project's established caution
      // around texture sampling, and because this texture has no mip
      // chain to select between regardless.
      const s = texture(sourceTexture, sampleUv).level(float(0)).r
      sum.addAssign(s.mul(w))
    })

    return vec4(vec3(sum), 1.0)
  })

  material.fragmentNode = shade()
  return material
}

/**
 * An orthographic camera for a full-viewport `[-1,1]x[-1,1]` quad pass (the
 * standard three.js post-processing pattern), so no vertex-node/billboard
 * tricks are needed for such a pass. `near`/`far` are `-1`/`1` (not `0`/`1`)
 * specifically so a plane at `z=0` sits safely inside the frustum rather
 * than exactly on its near boundary, where clip behaviour is
 * implementation-defined. Stateless (nothing here is updated per-frame),
 * so one instance is shared across every full-screen pass in this file's
 * pipeline (resolve H, resolve V) and the composite pass in viewer.ts.
 */
export function makeFullscreenCamera(): THREE.OrthographicCamera {
  return new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1)
}

export function buildFullscreenQuad(material: THREE.NodeMaterial): { mesh: THREE.Mesh; camera: THREE.OrthographicCamera } {
  const geometry = new THREE.PlaneGeometry(2, 2)
  const mesh = new THREE.Mesh(geometry, material)
  const camera = makeFullscreenCamera()
  return { mesh, camera }
}

/**
 * Composites the resolved (blurred) accumulation texture onto whatever is
 * already on screen, additively - the same premise as the shell's own
 * premultiplied-alpha accumulation (regimes/Shell.ts), just simpler here
 * since the resolve target already holds linear radiance rather than a
 * transmittance-weighted integral. Tinted to match the shell's own veil
 * colour (spec 6.2's bone-white/cold-cyan bioluminescence) so the two
 * regimes read as one continuous glow rather than two different palettes
 * meeting at the handoff.
 */
export function buildCompositeMaterial(
  resolvedTexture: THREE.Texture,
  tint: THREE.Color = new THREE.Color(0.72, 0.86, 0.95),
  exposure = 1,
): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial()
  material.transparent = true
  material.blending = THREE.AdditiveBlending
  material.depthWrite = false
  material.depthTest = false

  const shade = Fn((): Vec4Node => {
    const s = texture(resolvedTexture, uv()).level(float(0)).r
    const color = vec3(tint.r, tint.g, tint.b).mul(s).mul(exposure)
    return vec4(color, 1.0)
  })

  material.fragmentNode = shade()
  return material
}

/**
 * Temporary diagnostic, meant to be kept (not thrown away after use): draws
 * a raw single-channel accumulation target (typically `depositTarget`, but
 * works on any of the pipeline's RedFormat targets) straight to screen,
 * OPAQUE — no tint, no exposure curve, no blending — so it fully overwrites
 * whatever was drawn before it rather than compositing with it. Wired into
 * `viewer.ts` behind the 'v' key.
 *
 * Exists to split "nothing was deposited" from "deposit happened and a
 * later pass (blur/composite) lost it": point it at `depositTarget.texture`
 * directly and compare against the normal (blurred+composited) view. A flat
 * `gain` multiplier is applied before display because individual deposit
 * texels (one bilinear-corner-weighted flux contribution at a time) can be
 * faint enough to read as black even when nonzero — this is a visibility
 * aid, not a calibrated exposure curve like `buildCompositeMaterial`'s.
 */
export function buildDebugViewMaterial(sourceTexture: THREE.Texture, gain = 8): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial()
  material.transparent = false
  material.depthWrite = false
  material.depthTest = false

  const shade = Fn((): Vec4Node => {
    const s = texture(sourceTexture, uv()).level(float(0)).r
    return vec4(vec3(s.mul(gain)), 1.0)
  })

  material.fragmentNode = shade()
  return material
}

function makeAccumTarget(width: number, height: number): THREE.RenderTarget {
  return new THREE.RenderTarget(width, height, {
    format: THREE.RedFormat,
    type: THREE.HalfFloatType,
    magFilter: THREE.LinearFilter,
    minFilter: THREE.LinearFilter,
    generateMipmaps: false,
    depthBuffer: false,
    stencilBuffer: false,
  })
}

export interface SplatPipeline {
  depositTarget: THREE.RenderTarget
  depositMesh: THREE.Mesh
  depositScene: THREE.Scene
  /** Intermediate target: the horizontal pass's output, the vertical pass's input. */
  blurHTarget: THREE.RenderTarget
  blurHMesh: THREE.Mesh
  blurHScene: THREE.Scene
  /** Final target: the vertical pass's output, what the composite pass samples. */
  resolveTarget: THREE.RenderTarget
  resolveMesh: THREE.Mesh
  resolveScene: THREE.Scene
  resolveCamera: THREE.OrthographicCamera
  /** Rebuilds both resolve materials for a new sigma_pix - gaussianTaps' length depends on it, so the materials (not just a uniform) change. */
  setSigmaPix(sigmaPix: number): void
  dispose(): void
}

export function buildSplatPipeline(particles: ParticleSystem, width: number, height: number, initialSigmaPix: number): SplatPipeline {
  const depositMaterial = buildDepositMaterial(particles)
  const depositMesh = new THREE.Mesh(particles.geometry, depositMaterial)
  // depositMaterial.vertexNode computes each quad corner's real position from
  // the particle storage buffer, but the CPU-side geometry (four corner-index
  // vertices, see Particles.ts's buildQuadGeometry) never moves - three.js
  // computes frustumCulled's bounding sphere from THAT static geometry, has
  // no idea the real positions come from the vertex shader, and (worse under
  // floating origin, where a tiny quad near the render origin can look
  // degenerate in clip space) culls the whole indirect draw before it ever
  // reaches the GPU. Same reasoning applies to any future mesh whose
  // positions are computed in vertexNode/positionNode rather than read from
  // its own geometry.
  depositMesh.frustumCulled = false
  const depositScene = new THREE.Scene()
  depositScene.add(depositMesh)
  const depositTarget = makeAccumTarget(width, height)

  // One shared orthographic camera for every full-screen pass in this
  // pipeline (resolve H, resolve V) and viewer.ts's own composite pass -
  // see makeFullscreenCamera's doc for why sharing it is correct, not just
  // convenient.
  const resolveCamera = makeFullscreenCamera()

  let blurHMaterial = buildResolveMaterial(depositTarget.texture, initialSigmaPix, 'horizontal')
  const blurHMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), blurHMaterial)
  const blurHScene = new THREE.Scene()
  blurHScene.add(blurHMesh)
  const blurHTarget = makeAccumTarget(width, height)

  let blurVMaterial = buildResolveMaterial(blurHTarget.texture, initialSigmaPix, 'vertical')
  const resolveMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), blurVMaterial)
  const resolveScene = new THREE.Scene()
  resolveScene.add(resolveMesh)
  const resolveTarget = makeAccumTarget(width, height)

  function setSigmaPix(sigmaPix: number): void {
    blurHMaterial.dispose()
    blurHMaterial = buildResolveMaterial(depositTarget.texture, sigmaPix, 'horizontal')
    blurHMesh.material = blurHMaterial

    blurVMaterial.dispose()
    blurVMaterial = buildResolveMaterial(blurHTarget.texture, sigmaPix, 'vertical')
    resolveMesh.material = blurVMaterial
  }

  // The deposit mesh's geometry ownership stays with
  // Particles.ts's ParticleSystem.dispose() - only the two quad geometries
  // this pipeline allocated itself are disposed here.
  function dispose(): void {
    depositMaterial.dispose()
    blurHMaterial.dispose()
    blurVMaterial.dispose()
    depositTarget.dispose()
    blurHTarget.dispose()
    resolveTarget.dispose()
    blurHMesh.geometry.dispose()
    resolveMesh.geometry.dispose()
  }

  return {
    depositTarget, depositMesh, depositScene,
    blurHTarget, blurHMesh, blurHScene,
    resolveTarget, resolveMesh, resolveScene, resolveCamera,
    setSigmaPix, dispose,
  }
}
