import * as THREE from 'three/webgpu'
import { uniform } from 'three/tsl'
import { createRenderer } from '../core/Renderer'
import { buildShellMaterial } from '../regimes/Shell'
import { sigma0, arealCoverage, saturationCapTau } from '../math/optical'
import { syntheticTau } from '../field/synthetic'
import { bakeSyntheticCubeTexture } from '../field/bakeCube'
import { FrameTimer } from '../core/Timing'
import { probeAdapter } from '../platform/webgpu'
import type { ExpResult } from './types'

const A = 6378137.0
const N_MAX = 128
const FRAMES = 240
const BUDGET_MS = 2.5

const H_S = 32000
const H_MAX = 6 * H_S
const TAU_UNIFORM = 0.047
const CUBE_FACE_SIZE = 512

// Uniform-density content check: this config's tau is a compile-time
// constant (0.047) at EVERY frame, so the expected nadir coverage is exactly
// arealCoverage(0.047) ~= 4.6% (~12/255) regardless of camera position - a
// fixed threshold is valid here and is unchanged from the original Exp-1.
const BACKGROUND_ALPHA = 3

// Field/cube content check: fraction of the PER-FRAME expected alpha
// (derived below from syntheticTau at that frame's actual nadir direction,
// not a constant) a sampled pixel must clear. See expectedFieldAlpha255's
// doc for why a fixed threshold like BACKGROUND_ALPHA would be a false
// negative for either of these configs. The cube config reuses the SAME
// expected value as the analytic field: both derive tau from the same
// underlying syntheticTau at the same direction, and the cube's log-encoding
// quantization noise (see field/bakeCube.ts) is far smaller than this
// fraction's tolerance band, so a single shared expectation is correct, not
// an approximation of convenience.
const FIELD_ALPHA_FRACTION = 0.5

type ConfigName = 'uniform' | 'field' | 'cube' | 'cube-far-wall'

/**
 * Expected 8-bit alpha at the screen-centre pixel for a field-driven config
 * (analytic or cube-sampled), given the camera position for a frame in this
 * experiment.
 *
 * Every camera in this run looks straight at the world origin, and the shell
 * is centred there, so the screen-centre ray's nadir direction is just
 * `camPos`'s own direction - `syntheticTau` normalizes internally, exactly
 * mirroring what the shader samples for a near-nadir ray in both the
 * analytic (`syntheticTauNode`) and cube-sampled (`cubeTexture(...).level(0)`)
 * branches of `stepSigma0`. `alpha ~= 1 - transmittance ~= 1 - exp(-tau_nadir)`
 * for a near-nadir view, i.e. `arealCoverage(tau)`, which is what the
 * original Exp-1 comment already assumed for its own fixed tau=0.047 case.
 */
function expectedFieldAlpha255(camPos: THREE.Vector3): number {
  const tau = syntheticTau(camPos.x, camPos.y, camPos.z)
  return arealCoverage(tau) * 255
}

/**
 * Content-check verdict for a field-driven config's sampled alpha against
 * its per-frame expected value. Below one 8-bit count of expected signal, a
 * correct render and a broken one are genuinely indistinguishable at this
 * pixel (both read 0 or round to 0/1) - a real quantization floor, not a bug
 * in this check - so that case is reported `inconclusive` rather than forced
 * through a threshold that would produce spurious failures right at the
 * boundary.
 */
function checkFieldContent(sampledAlpha: number, expectedAlpha255: number): {
  failed: boolean
  inconclusive: boolean
  backgroundAlphaThreshold: number | null
} {
  const inconclusive = expectedAlpha255 < 1
  const backgroundAlphaThreshold = inconclusive ? null : expectedAlpha255 * FIELD_ALPHA_FRACTION
  const failed = !inconclusive && backgroundAlphaThreshold !== null && sampledAlpha < backgroundAlphaThreshold
  return { failed, inconclusive, backgroundAlphaThreshold }
}

/**
 * Exp-1: shell march cost at full-globe framing, where the shell covers roughly
 * 10-25% of screen. Highest priority experiment in Session 1.
 *
 * NOTE (fix round 1): at this framing the sphere's angular radius (0.451 rad)
 * exceeds the half-FOV (0.436 rad), so the shell actually overfills the frame
 * and the limb is off-screen - the "~10-25% of screen" comment above is wrong,
 * it is closer to 100%. Left uncorrected here per instruction: do not change
 * the framing, only record the discrepancy for Session 2 to decide on.
 *
 * The 1.5-2.5 ms figure is a HYPOTHESIS TO FALSIFY, not a budget. No public
 * WebGPU raymarching timings exist; the published anchors (Sakmary, Horizon
 * Zero Dawn) do not survive scrutiny as extrapolation bases. Spec section 11.
 *
 * Renderer lifecycle: everything constructed after `createRenderer` succeeds
 * lives inside the try below, and disposal in `finally` is guarded (`?.`) for
 * partially-constructed state, so a throw at any point - including inside
 * `createRenderer` itself - can't leak a WebGPU context into the next
 * experiment in the sequential run.
 *
 * TIMING (Session 2 rework): wall-clock around a forced-sync readback was the
 * only measurement in Session 1, and it turned out to be untrustworthy - an
 * identical field-free shader re-measured at 10.90 ms (p95 30.4 ms) on a
 * later run that had earlier measured 2.50 ms, a 4x swing from GPU contention
 * with no code change. GPU pass timing via the `timestamp-query` feature
 * measures the PASS, not the queue, and is immune to that contention, so it
 * is now the primary number whenever the adapter supports it (`hasTimestamp`
 * below - gated per spec section 10: 97.95% overall but only 85.44% on
 * Safari, so this must degrade to wall-clock, not throw, when absent). Both
 * numbers are still recorded: the gap between them IS the contention signal,
 * and hiding it would throw away exactly the thing that caught the false
 * PASS above. Chrome quantizes GPU timestamps to 100us, so `FrameTimer`
 * keeps averaging ACROSS frames rather than reacting per frame - no adaptive
 * controller is built on top of this here.
 *
 * CONTENT CHECK (Session 1, extended in Session 2): a forced-sync readback
 * alone proves the GPU finished SOMETHING, not that it finished marching a
 * non-empty shell - a compile failure, culled mesh, or depth-rejected frame
 * would all resolve without throwing and cost ~0.2 ms, a false PASS. The
 * alpha channel of the sampled pixel is checked against an expected level
 * before any pass/fail verdict. For the uniform-density config that expected
 * level is the fixed BACKGROUND_ALPHA (tau is a compile-time constant there);
 * for the field-sampled and cube-sampled configs it is derived per-frame from
 * the field itself, via expectedFieldAlpha255/checkFieldContent above.
 *
 * COST COMPARISON (Session 2, extended for the cube config): three otherwise
 * identical configs (Hs/Hmax/sigma0/N_MAX/camera path all shared) - uniform
 * density, the analytic field (~900 sin() calls/pixel: 7 hash calls/fbm x
 * 128 steps), and a field baked into an R16Float cube texture at startup and
 * sampled per step instead. The hypothesis under test: the cube fetch should
 * be nearly free next to those ~900 transcendentals, i.e. cube cost should
 * land close to uniform, not close to analytic. All three configs render
 * back-to-back INSIDE every loop iteration (not as separate sequential runs),
 * so slow drift across the run - thermal ramp, background contention -
 * affects all three equally instead of skewing whichever config happened to
 * run later. Nothing here is tuned toward a particular ratio; the reported
 * numbers are whatever the three configs actually cost. The cube bake itself
 * (baking `syntheticTau` for 512x512x6 texels on the CPU) happens once
 * before the timed loop and is not included in any per-frame measurement.
 *
 * FOURTH CONFIG (cube-far-wall, added when `ShellFarWallOpts` shipped):
 * identical to `cube` except `farWall.enabled` is pinned `true` rather than
 * driven by the viewer's per-frame matter/altitude gate - this experiment
 * measures the far wall's OWN added cost in isolation, not whether the gate
 * itself is wired correctly (that's shell.test.ts's job). At this framing
 * (~8700 km altitude, the shell overfilling the frame per the NOTE above)
 * most pixels' rays genuinely cross the inner sphere, so the far wall's
 * second Loop actually executes for most of the frame - a representative
 * worst case, not a best case where it rarely engages. Purely additive:
 * this fourth config's timing is reported in `data` alongside the other
 * three but does NOT affect `pass`, which still gates on the original
 * uniform/field/cube trio exactly as before - a wrong number here should
 * never silently break the Session 1 budget check. Its own content check is
 * informational only for the same reason (see contentData.cubeFarWall's
 * doc below). UNVERIFIED AT COMMIT TIME: this file needs a live WebGPU
 * adapter to run and none was available in the environment that wrote it -
 * run this experiment (or watch the far wall render in the actual viewer)
 * before trusting these numbers.
 */
export async function runExp01(): Promise<ExpResult> {
  const adapter = await probeAdapter()
  const canvas = document.createElement('canvas')
  canvas.width = 1920
  canvas.height = 1080

  let renderer: THREE.WebGPURenderer | undefined
  let target: THREE.RenderTarget | undefined
  let geometry: THREE.SphereGeometry | undefined
  let materialUniform: THREE.NodeMaterial | undefined
  let materialField: THREE.NodeMaterial | undefined
  let materialCube: THREE.NodeMaterial | undefined
  let materialCubeFarWall: THREE.NodeMaterial | undefined
  let cubeTex: THREE.CubeTexture | undefined

  const submitOnlyTimer = new FrameTimer(FRAMES) // uniform config only, see its doc below
  const lastCamPos = new THREE.Vector3()
  const lastAlpha: Record<ConfigName, number> = { uniform: -1, field: -1, cube: -1, 'cube-far-wall': -1 }

  interface TimedConfig {
    name: ConfigName
    scene: THREE.Scene
    wallTimer: FrameTimer
    gpuTimer: FrameTimer
  }
  let configs: TimedConfig[] = []

  try {
    renderer = await createRenderer(canvas, { trackTimestamp: true })
    renderer.setPixelRatio(1)
    renderer.setSize(1920, 1080, false)

    // Authoritative gate: this is the SAME renderer/backend instance whose
    // resolveTimestampsAsync we're about to call, so it can't drift from
    // probeAdapter's separate requestAdapter() call the way two independent
    // adapter probes theoretically could. adapter.hasTimestamp (from the
    // probe above) is still reported in `data` for consistency with the
    // other experiments' convention, but this is what gates the timing path.
    const hasTimestamp = renderer.hasFeature('timestamp-query')

    target = new THREE.RenderTarget(1920, 1080)

    const Hs = uniform(H_S)
    const Hmax = uniform(H_MAX)
    const s0 = uniform(sigma0(TAU_UNIFORM, H_S, H_MAX))
    const sigma0PerTau = uniform(sigma0(1, H_S, H_MAX))
    const tauCap = uniform(saturationCapTau(H_S))

    materialUniform = buildShellMaterial({ Hs, Hmax, sigma0: s0, nMax: N_MAX })
    materialField = buildShellMaterial({
      Hs, Hmax, sigma0: s0,
      field: { useField: uniform(true), sigma0PerTau, tauCap },
      nMax: N_MAX,
    })

    cubeTex = bakeSyntheticCubeTexture({ size: CUBE_FACE_SIZE })
    materialCube = buildShellMaterial({
      Hs, Hmax, sigma0: s0,
      cube: { texture: cubeTex, sigma0PerTau, tauCap },
      nMax: N_MAX,
    })
    // Same cube texture and every other opt as `cube` above - only
    // `farWall.enabled` differs (pinned true, not gated) - so any timing
    // delta against `cube` is attributable to the far wall alone. See this
    // function's doc for why this fourth config is purely additive.
    materialCubeFarWall = buildShellMaterial({
      Hs, Hmax, sigma0: s0,
      cube: { texture: cubeTex, sigma0PerTau, tauCap },
      farWall: { enabled: uniform(true) },
      nMax: N_MAX,
    })

    geometry = new THREE.SphereGeometry(A * 1.031, 96, 48)

    const makeScene = (material: THREE.NodeMaterial): THREE.Scene => {
      const scene = new THREE.Scene()
      scene.add(new THREE.Mesh(geometry, material))
      return scene
    }

    configs = [
      { name: 'uniform', scene: makeScene(materialUniform), wallTimer: new FrameTimer(FRAMES), gpuTimer: new FrameTimer(FRAMES) },
      { name: 'field', scene: makeScene(materialField), wallTimer: new FrameTimer(FRAMES), gpuTimer: new FrameTimer(FRAMES) },
      { name: 'cube', scene: makeScene(materialCube), wallTimer: new FrameTimer(FRAMES), gpuTimer: new FrameTimer(FRAMES) },
      { name: 'cube-far-wall', scene: makeScene(materialCubeFarWall), wallTimer: new FrameTimer(FRAMES), gpuTimer: new FrameTimer(FRAMES) },
    ]

    // Full-globe framing: ~8700 km altitude, the composition the piece opens on.
    const camera = new THREE.PerspectiveCamera(50, 1920 / 1080, 1, 1e9)
    camera.position.set(0, 0, A + 8.7e6)
    camera.lookAt(0, 0, 0)

    const axis = new THREE.Vector3(0, 1, 0)

    renderer.setRenderTarget(target)
    // Per-config progress, not just a single "compiling" message: pipeline
    // compilation is the one step in this function whose cost genuinely
    // scales with shader complexity (unlike the bounded, per-frame timed
    // loop below), so if this experiment is ever slow enough to look hung,
    // logging WHICH config's compileAsync is in flight - rather than only
    // finding out after all four have resolved - is what turns "the console
    // shows nothing" into an actual diagnostic (see report.ts's
    // runSequentially doc for the matching top-level fix).
    for (const cfg of configs) {
      const tCompile0 = performance.now()
      await renderer.compileAsync(cfg.scene, camera)
      console.log(`[exp-01] compiled ${cfg.name} in ${(performance.now() - tCompile0).toFixed(0)} ms`)
    }

    const PROGRESS_EVERY_FRAMES = 30
    for (let i = 0; i < FRAMES; i++) {
      if (i > 0 && i % PROGRESS_EVERY_FRAMES === 0) {
        console.log(`[exp-01] frame ${i}/${FRAMES}`)
      }
      camera.position.applyAxisAngle(axis, 0.004)
      camera.lookAt(0, 0, 0)

      // Every config renders back-to-back under this SAME camera pose before
      // the pose advances again - the interleaving that keeps drift from
      // skewing one config over the others.
      for (const cfg of configs) {
        const t0 = performance.now()
        await renderer.renderAsync(cfg.scene, camera)
        const tSubmit = performance.now()
        const gpuMs = hasTimestamp ? await renderer.resolveTimestampsAsync('render') : undefined
        // Forces GPU completion: readback cannot resolve before the frame it
        // reads from has actually finished executing. The shell overfills
        // the frame at this camera (see NOTE above), so any pixel is covered.
        const px = await renderer.readRenderTargetPixelsAsync(target, 960, 540, 1, 1)
        const tSync = performance.now()
        lastAlpha[cfg.name] = px[3]!

        if (cfg.name === 'uniform') submitOnlyTimer.push(tSubmit - t0)
        cfg.wallTimer.push(tSync - t0)
        if (gpuMs !== undefined) cfg.gpuTimer.push(gpuMs)
      }

      lastCamPos.copy(camera.position)
    }
  } finally {
    renderer?.setRenderTarget(null)
    target?.dispose()
    geometry?.dispose()
    materialUniform?.dispose()
    materialField?.dispose()
    materialCube?.dispose()
    materialCubeFarWall?.dispose()
    cubeTex?.dispose()
    renderer?.dispose()
  }

  const byName = (name: ConfigName) => {
    const cfg = configs.find((c) => c.name === name)
    if (!cfg) throw new Error(`missing timed config: ${name}`) // unreachable - configs is always fully populated above
    return cfg
  }
  const uniformCfg = byName('uniform')
  const fieldCfg = byName('field')
  const cubeCfg = byName('cube')
  const cubeFarWallCfg = byName('cube-far-wall')

  // Primary number: GPU pass median when the adapter supports it (immune to
  // the contention that corrupted the old wall-clock-only measurement),
  // wall-clock median otherwise. Never silently - which basis is live is
  // recorded in `data.timingBasis`.
  const uniformHasGpu = uniformCfg.gpuTimer.samples > 0
  const uniformPrimaryMs = uniformHasGpu ? uniformCfg.gpuTimer.median : uniformCfg.wallTimer.median

  const uniformContentFailed = lastAlpha.uniform < BACKGROUND_ALPHA

  const expectedAlpha255 = expectedFieldAlpha255(lastCamPos)
  const fieldCheck = checkFieldContent(lastAlpha.field, expectedAlpha255)
  const cubeCheck = checkFieldContent(lastAlpha.cube, expectedAlpha255)

  // Informational only, NOT part of the pass/fail gate below: the far-wall
  // config draws two walls' worth of alpha, so it clears the same
  // single-wall `expectedAlpha255` threshold trivially whenever anything
  // rendered at all - this confirms "something drew", not "the far wall
  // specifically drew the right amount". A real correctness check on its
  // rendered VALUE would need its own two-wall expected-alpha model, which
  // this experiment (a cost comparison, not a correctness one - see
  // shell.test.ts/shellFarWall.test.ts for correctness) does not attempt.
  const cubeFarWallCheck = checkFieldContent(lastAlpha['cube-far-wall'], expectedAlpha255)

  const contentData = {
    uniform: { sampledAlpha: lastAlpha.uniform, backgroundAlphaThreshold: BACKGROUND_ALPHA, failed: uniformContentFailed },
    field: { sampledAlpha: lastAlpha.field, expectedAlpha255, ...fieldCheck },
    cube: { sampledAlpha: lastAlpha.cube, expectedAlpha255, ...cubeCheck },
    cubeFarWall: { sampledAlpha: lastAlpha['cube-far-wall'], expectedAlpha255, ...cubeFarWallCheck, note: 'informational - see doc above, not part of pass/fail' },
  }

  if (uniformContentFailed || fieldCheck.failed || cubeCheck.failed) {
    const failedConfigs = [
      uniformContentFailed ? 'uniform-density' : null,
      fieldCheck.failed ? 'field-sampled' : null,
      cubeCheck.failed ? 'cube-sampled' : null,
    ].filter((s): s is string => s !== null)
    const which = failedConfigs.length > 1
      ? `${failedConfigs.join(', ')} configs'`
      : `${failedConfigs[0] ?? 'unknown'} config's`
    return {
      id: 'exp-01-march-cost',
      pass: false,
      detail: `${which} centre pixel alpha read below its expected level - the shell did not render (compile failure, culling, or depth rejection) for that config. Timing numbers must not be read as a pass or fail against the budget.`,
      data: {
        frames: FRAMES,
        nMax: N_MAX,
        resolution: '1920x1080',
        hasTimestampQueryAdapter: adapter.hasTimestamp,
        hasTimestampQueryRenderer: hasTimestampRendererFromConfigs(configs),
        content: contentData,
      },
    }
  }

  const timingSummary = (cfg: TimedConfig) => {
    const hasGpu = cfg.gpuTimer.samples > 0
    return {
      gpuMedianMs: hasGpu ? cfg.gpuTimer.median : null,
      gpuP05Ms: hasGpu ? cfg.gpuTimer.p05 : null,
      gpuP95Ms: hasGpu ? cfg.gpuTimer.p95 : null,
      gpuStddevMs: hasGpu ? cfg.gpuTimer.stddev : null,
      wallMedianMs: cfg.wallTimer.median,
      wallP05Ms: cfg.wallTimer.p05,
      wallP95Ms: cfg.wallTimer.p95,
      wallStddevMs: cfg.wallTimer.stddev,
    }
  }

  const ratioToUniform = (cfg: TimedConfig) => {
    const cfgHasGpu = cfg.gpuTimer.samples > 0
    const gpu = uniformHasGpu && cfgHasGpu ? cfg.gpuTimer.median / uniformCfg.gpuTimer.median : null
    const wall = cfg.wallTimer.median / uniformCfg.wallTimer.median
    return { gpu, wall }
  }

  const fieldRatio = ratioToUniform(fieldCfg)
  const cubeRatio = ratioToUniform(cubeCfg)
  const cubeFarWallRatio = ratioToUniform(cubeFarWallCfg)
  const cubeHasGpu = cubeCfg.gpuTimer.samples > 0
  const cubePrimaryMs = cubeHasGpu ? cubeCfg.gpuTimer.median : cubeCfg.wallTimer.median
  const fieldHasGpu = fieldCfg.gpuTimer.samples > 0
  const fieldPrimaryMs = fieldHasGpu ? fieldCfg.gpuTimer.median : fieldCfg.wallTimer.median
  const cubeFarWallHasGpu = cubeFarWallCfg.gpuTimer.samples > 0
  const cubeFarWallPrimaryMs = cubeFarWallHasGpu ? cubeFarWallCfg.gpuTimer.median : cubeFarWallCfg.wallTimer.median
  // The far wall's OWN added cost, isolated from the cube-fetch cost cube
  // already pays: cube-far-wall / cube, not vs uniform. At this framing
  // (shell overfilling the frame) most pixels' rays cross the inner sphere,
  // so this is close to the worst-case doubling, not a lucky best case.
  const farWallAddedCostVsCube = {
    gpu: cubeHasGpu && cubeFarWallHasGpu ? cubeFarWallCfg.gpuTimer.median / cubeCfg.gpuTimer.median : null,
    wall: cubeFarWallCfg.wallTimer.median / cubeCfg.wallTimer.median,
  }

  return {
    id: 'exp-01-march-cost',
    pass: uniformPrimaryMs <= BUDGET_MS,
    detail: uniformPrimaryMs <= BUDGET_MS
      ? `Median ${uniformPrimaryMs.toFixed(2)} ms (${uniformHasGpu ? 'GPU pass timing' : 'wall-clock, timestamp-query unavailable'}) at 1080p, N_MAX=${N_MAX}, uniform-density config. Hypothesis survives. Field-sampled: ${fieldPrimaryMs.toFixed(2)} ms (${(fieldRatio.gpu ?? fieldRatio.wall).toFixed(2)}x vs uniform). Cube-sampled: ${cubePrimaryMs.toFixed(2)} ms (${(cubeRatio.gpu ?? cubeRatio.wall).toFixed(2)}x vs uniform). Cube+far-wall: ${cubeFarWallPrimaryMs.toFixed(2)} ms (${(farWallAddedCostVsCube.gpu ?? farWallAddedCostVsCube.wall).toFixed(2)}x vs plain cube, informational - not gated).`
      : `Median ${uniformPrimaryMs.toFixed(2)} ms (${uniformHasGpu ? 'GPU pass timing' : 'wall-clock, timestamp-query unavailable'}) at 1080p exceeds the ${BUDGET_MS} ms hypothesis for the uniform-density config. Adopt half-res plus temporal upsample BEFORE any other optimization.`,
    data: {
      frames: FRAMES,
      nMax: N_MAX,
      resolution: '1920x1080',
      cubeFaceSize: CUBE_FACE_SIZE,
      hasTimestampQueryAdapter: adapter.hasTimestamp,
      hasTimestampQueryRenderer: uniformHasGpu,
      timingBasis: uniformHasGpu ? 'gpu-timestamp-query' : 'wall-clock (timestamp-query unavailable on this adapter)',
      content: contentData,
      framingNote: 'sphere angular radius 0.451 rad exceeds half-FOV 0.436 rad at this camera: the shell overfills the frame (~100%), not the ~10-25% the framing comment claims',
      uniform: {
        ...timingSummary(uniformCfg),
        submitOnlyMs: submitOnlyTimer.median,
        submitOnlyNote: 'submitOnlyMs times renderAsync() alone with no sync - it reads near zero because renderAsync resolves on submission, not completion, and must not be used as the pass/fail number',
        measurement: 'wall-clock is render + 1px readback via readRenderTargetPixelsAsync, forced to sync on GPU completion; it slightly overestimates true march cost by the readback overhead, and (per the doc comment above) can be inflated further by unrelated GPU contention - compare against gpuMedianMs for that signal',
      },
      field: timingSummary(fieldCfg),
      cube: timingSummary(cubeCfg),
      cubeFarWall: timingSummary(cubeFarWallCfg),
      costRatios: {
        field: { ...fieldRatio, note: 'field-sampled median / uniform-density median (gpu basis when both configs have GPU samples, else wall-clock).' },
        cube: { ...cubeRatio, note: 'cube-sampled median / uniform-density median (gpu basis when both configs have GPU samples, else wall-clock).' },
        cubeFarWall: { ...cubeFarWallRatio, vsCube: farWallAddedCostVsCube, note: 'cubeFarWall.gpu/wall are vs uniform-density (for consistency with the other rows); vsCube is the far wall\'s OWN added cost, cube-far-wall median / plain-cube median - the number that actually answers "what does the far wall cost". Informational: NOT gated by `pass` (see this file\'s module doc\'s FOURTH CONFIG section) and unverified at commit time - no WebGPU adapter was available to run this experiment when it was written.' },
        hypothesis: 'the cube fetch is expected to land close to 1x (uniform-like), not close to the field ratio - a baked lookup replacing ~900 sin() calls/pixel should be nearly free by comparison. Not tuned toward this or any other expected value.',
      },
    },
  }
}

/** True if ANY config actually accumulated a GPU timestamp sample - used only for the content-check failure branch's summary flag. */
function hasTimestampRendererFromConfigs(configs: Array<{ gpuTimer: FrameTimer }>): boolean {
  return configs.some((c) => c.gpuTimer.samples > 0)
}
