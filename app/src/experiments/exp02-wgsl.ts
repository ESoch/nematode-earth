import { createRenderer } from '../core/Renderer'
import { buildShellMaterial, dumpShellWGSL } from '../regimes/Shell'
import { sigma0, saturationCapTau } from '../math/optical'
import { bakeSyntheticCubeTexture } from '../field/bakeCube'
import type { ExpResult } from './types'
import { uniform } from 'three/tsl'

const N_MAX = 128
const H_S = 32000
const H_MAX = 192000
// Tiny bake: exp-2 only needs a material that compiles and samples a real
// cube texture in the shader, not a meaningful image - a full 512-face bake
// would just slow this experiment down for no benefit here.
const CUBE_FACE_SIZE = 4

// three.js's WGSLNodeBuilder (node_modules/three/src/renderers/webgpu/nodes/
// WGSLNodeBuilder.js:245-249,2561) unconditionally prepends this exact line to
// EVERY fragment shader it compiles, in every browser except Firefox/Deno:
// a permanent, library-wide opt-out of the WGSL derivative_uniformity rule,
// present regardless of whether a given shader's control flow is actually
// non-uniform. It exists precisely because node-graph-generated shaders
// (like the Break() in the march loop below) routinely can't be proven
// uniform, and three.js would rather silence the check than have its own
// codegen fail validation. Confirmed at the pipeline level too: WebGPU
// PipelineUtils.js's `_reportShaderDiagnostics` (line 447) - the one place
// that reads GPUShaderModule.getCompilationInfo() - only runs from the
// popErrorScope('validation') failure path, and this diagnostic class never
// reaches the validator in the first place, so it can never fire here.
//
// A literal `/derivative_uniformity/.test(code)` (as in the brief) is
// therefore always true and would fail Exp-2 unconditionally, on every
// shader this renderer ever produces - a vacuous fail, not a real finding.
// We instead flag only occurrences OUTSIDE that exact suppression line,
// which is the only signal this text channel can actually carry.
const SUPPRESSION_LINE = /^\s*diagnostic\(\s*off\s*,\s*derivative_uniformity\s*\)\s*;?\s*$/m

interface WgslCheck {
  bytes: number
  expCount: number
  unrolled: boolean
  usesImplicitSample: boolean
  usesExplicitLevel: boolean
  anomalousDerivativeText: boolean
  problems: string[]
}

function checkWgsl(code: string, requireExplicitLevelIfSampled: boolean): WgslCheck {
  const usesImplicitSample = /\btextureSample\s*\(/.test(code)
  const usesExplicitLevel = /textureSampleLevel\s*\(/.test(code)
  const anomalousDerivativeText = code
    .split('\n')
    .some((line) => line.includes('derivative_uniformity') && !SUPPRESSION_LINE.test(line))
  const expCount = (code.match(/\bexp\s*\(/g) ?? []).length
  const unrolled = expCount > 16

  const problems: string[] = []
  if (requireExplicitLevelIfSampled && usesImplicitSample && !usesExplicitLevel) {
    problems.push('implicit-LOD textureSample inside non-uniform control flow')
  }
  if (anomalousDerivativeText) problems.push('derivative_uniformity text present outside the expected blanket-suppression directive')
  if (unrolled) problems.push(`loop appears unrolled (${expCount} exp() calls for N_MAX=${N_MAX})`)

  return { bytes: code.length, expCount, unrolled, usesImplicitSample, usesExplicitLevel, anomalousDerivativeText, problems }
}

/**
 * Exp-2: inspect the generated WGSL for the march loop.
 *
 * PASS: explicit-LOD sampling (if any texture sampling occurs at all), no
 * derivative_uniformity diagnostic beyond three.js's blanket suppression
 * line, and the loop is a real loop rather than N_MAX unrolled copies.
 *
 * Session 2 (cube-sampled config): the uniform-density shader below still
 * does no texture sampling at all, so explicit-LOD stays untestable for it,
 * exactly as in Session 1. The cube-sampled shader is a SECOND material this
 * experiment builds and dumps specifically to make that criterion testable:
 * regimes/Shell.ts's cube branch calls `cubeTexture(...).level(0)` inside
 * the march loop precisely because the loop's transmittance Break() makes
 * control flow non-uniform and WGSL forbids implicit-LOD textureSample
 * there (WGSL also has no textureLoad overload for texture_cube at all, so
 * textureSampleLevel is the only legal path regardless). `requireExplicitLevelIfSampled`
 * is what actually enforces that for the cube shader, where it is no longer
 * vacuously true the way it was for the texture-free uniform shader.
 */
export async function runExp02(): Promise<ExpResult> {
  const canvas = document.createElement('canvas')
  canvas.width = 64
  canvas.height = 64

  let renderer: Awaited<ReturnType<typeof createRenderer>> | undefined
  let materialUniform: ReturnType<typeof buildShellMaterial> | undefined
  let materialCube: ReturnType<typeof buildShellMaterial> | undefined
  let cubeTex: ReturnType<typeof bakeSyntheticCubeTexture> | undefined
  let codeUniform = ''
  let codeCube = ''

  try {
    renderer = await createRenderer(canvas)

    const Hs = uniform(H_S)
    const Hmax = uniform(H_MAX)
    const s0 = uniform(sigma0(0.047, H_S, H_MAX))
    materialUniform = buildShellMaterial({ Hs, Hmax, sigma0: s0, nMax: N_MAX })
    codeUniform = await dumpShellWGSL(renderer, materialUniform)

    cubeTex = bakeSyntheticCubeTexture({ size: CUBE_FACE_SIZE })
    materialCube = buildShellMaterial({
      Hs, Hmax, sigma0: s0,
      cube: {
        texture: cubeTex,
        sigma0PerTau: uniform(sigma0(1, H_S, H_MAX)),
        tauCap: uniform(saturationCapTau(H_S)),
      },
      nMax: N_MAX,
    })
    codeCube = await dumpShellWGSL(renderer, materialCube)
  } finally {
    materialUniform?.dispose()
    materialCube?.dispose()
    cubeTex?.dispose()
    renderer?.dispose()
  }

  if (!codeUniform || !codeCube) {
    return {
      id: 'exp-02-wgsl',
      pass: false,
      detail: `Could not retrieve generated WGSL for ${!codeUniform ? 'the uniform-density' : 'the cube-sampled'} material: renderer._pipelines.programs.fragment was empty or the internal accessor path has moved again. The check cannot be performed, so it cannot pass.`,
    }
  }

  const uniformCheck = checkWgsl(codeUniform, false)
  const cubeCheck = checkWgsl(codeCube, true)
  const problems = [
    ...uniformCheck.problems.map((p) => `uniform: ${p}`),
    ...cubeCheck.problems.map((p) => `cube: ${p}`),
  ]

  return {
    id: 'exp-02-wgsl',
    pass: problems.length === 0,
    detail: problems.length === 0
      ? `Uniform-density shader: WGSL compiles clean, no anomalous derivative_uniformity text, loop not unrolled (${uniformCheck.expCount} exp() calls); explicit-LOD is untestable there (no texture sampling occurs). Cube-sampled shader: ${cubeCheck.usesExplicitLevel ? 'uses textureSampleLevel (explicit LOD)' : 'samples no texture'}${cubeCheck.usesImplicitSample ? ', WARNING also emits implicit-LOD textureSample' : ', no implicit-LOD textureSample'}, no anomalous derivative_uniformity text, loop not unrolled (${cubeCheck.expCount} exp() calls).`
      : problems.join('; '),
    data: {
      uniform: uniformCheck,
      cube: cubeCheck,
    },
  }
}
