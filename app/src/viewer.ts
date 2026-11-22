/**
 * Shell preview — NOT part of the falsification harness.
 *
 * Session 1's four experiments render offscreen into a RenderTarget because
 * they measure rather than display. This page exists purely so the shell can
 * be looked at: same `buildShellMaterial`, same constants, drawn to the canvas.
 *
 * Density comes from the cube-sampled path (`src/field/bakeCube.ts`, baked
 * once at startup): Session 2's Exp-1 measured it at 4.4x cheaper than the
 * analytic field port for the same result (1.35x vs 6x the uniform-density
 * baseline), so this page runs the config that's actually shippable rather
 * than the one that first proved the field could be sampled at all.
 *
 * Interactive as of Session 2: drag to orbit, and scroll or pinch to descend,
 * via `ScaleCamera` (logarithmic descent, floating origin). The pinch is
 * later than the rest of that sentence: descent was bound to `wheel` alone
 * for four sessions, which meant that on a phone the interaction this whole
 * piece is built around did not exist at all. `core/pointerGesture.ts` holds
 * the multi-pointer state machine and both inputs reach the camera through
 * the same `scrollBy`, so they cannot diverge. The camera stays
 * pinned at the render origin; the shell mesh is translated by
 * `-cameraWorldPosition` each frame instead - see `regimes/Shell.ts`'s
 * `ShellOpts.cameraWorldPosition` doc for why the shader needs the camera's
 * TRUE planet-relative position through a dedicated uniform once
 * `camera.position` itself is pinned to zero.
 *
 * Session 3 adds the sweep: an opaque "matter" mesh (`sweep/Matter.ts`) -
 * real ETOPO 2022 elevation and relief, real land-fraction-blended
 * hypsometric colour, real bathymetry, not a placeholder ball - that
 * dissolves layer by layer (atmosphere, then water, then rock) via three
 * spatially propagating fronts (`sweep/Dissolve.ts`), scheduled by a
 * scrubbable five-beat timeline (`sweep/Timeline.ts`). The governing rule
 * (spec 6.1) is structural, not cosmetic: the veil (the shell material
 * above) is NEVER touched by the sweep - same Hs/Hmax/sigma0 wiring as
 * before, always computing the same density it always would. What the
 * sweep controls is only whether the matter mesh is still standing in
 * front of it. See sweep/Matter.ts's doc for how that occlusion is
 * literally implemented (opaque depth test/write plus dithered discard,
 * never alpha blending) and for how the collapse beat (surface dropping to
 * reveal bathymetry) is driven by water's own front.
 *
 * Session 4 adds regime 2's FOUNDATION: compute-emitted, world-anchored
 * particles (`regimes/Particles.ts`), the Nelson Max partition coupling
 * (`regimes/Blend.ts`, wired into the shell additively via
 * `shellPartitionShare` - the shell material itself is otherwise
 * unchanged), and the discrete-domain splat draw pass
 * (`regimes/SplatBlur.ts`). Press `r` to cycle shell-only / particles-only /
 * blended at whatever altitude the descent is currently at - that is this
 * session's side-by-side comparison config. Several real simplifications
 * are documented in `regimes/Particles.ts`'s own doc comment (single-level
 * lattice, a symmetric candidate box rather than a frustum-fitted one,
 * direct rather than incremental altitude, vertical rather than slant-path
 * transmittance) and the budget here (200k, not spec 7.7's 4e6) is sized
 * for a comparison tool on one machine, not the shipped scale. Press `v` to
 * view `depositTarget` raw (unblurred, uncomposited) — a permanent
 * diagnostic (`SplatBlur.ts`'s `buildDebugViewMaterial`) that separates
 * "nothing was deposited" from "deposit happened and a later pass lost it".
 * Press `p` to log the indirect-draw args and emission atomic counters
 * straight off the GPU to the console (`logParticleBuffers` below) — also
 * permanent, for checking whether the draw itself is issued at all. Press
 * `t` to trace one known-flux, known-position instance through deposit ->
 * resolve -> composite and print the MEASURED value at each stage
 * (`logSingleInstanceTrace` below) — also permanent; three separate
 * defects this session turned out to be silent zeroing/scaling in exactly
 * this class of layer, and instrumenting adjacent stages is what caught
 * all three, not algebra. `t` also runs `logDepositCoverageTest`
 * afterward, camera frozen for both: solves for specific known sub-pixel
 * offsets and compares `nonZeroCount`/`sum` between the shipped deposit
 * material and a corner-vs-centre test variant
 * (`buildDepositMaterialCornerVariantForTesting`, SplatBlur.ts) - the
 * discriminator for a coverage-bug hypothesis a non-deterministic first
 * version couldn't reliably test.
 */
import * as THREE from 'three/webgpu'
import { uniform, exp, storage } from 'three/tsl'
import { isWebGPUAvailable } from './platform/webgpu'
import { createRenderer } from './core/Renderer'
import { buildShellMaterial } from './regimes/Shell'
import { loadSamplePoints, type SamplePointsLayer } from './regimes/SamplePoints'
import { bakeSyntheticCubeTexture, bakeSyntheticDensityArrayTexture } from './field/bakeCube'
import { loadCubeTexture, sampleCubeDensity, type LoadedCube } from './field/loadCube'
import { sampleDensityArrayNode } from './field/cubeArrayNodes'
import { probeGeoLocation, KNOWN_PROBES, latLonToDirection, type GeoProbeResult } from './field/geoProbe'
import {
  cellMaxDisplacementM, interpolatedDisplacementM, sampleElevationCube,
  MATTER_WIDTH_SEGMENTS, MATTER_HEIGHT_SEGMENTS,
} from './field/groundHeight'
import { matterAtlasBindings } from './sweep/matterAtlases'
import { replaceMaterial } from './core/materialSwap'
import { ScaleCamera } from './core/ScaleCamera'
import { OpeningExperience, openingControlLayout } from './core/openingExperience'
import { createViewerPolicy } from './core/viewerPolicy'
import { dragToOrbit, wheelDeltaToTicks, projectedUp } from './core/orbitInput'
import { PointerGesture } from './core/pointerGesture'
import { verticalFovForViewport, frameCameraToRegion } from './core/viewFraming'
import { zoomQualityForView, type ZoomQuality } from './core/zoomQuality'
import './hud/viewer.css'
import { planAtlasLadder, readClientBudgetHints } from './platform/atlasLadder'
import { sigma0, saturationCapTau, E_A } from './math/optical'
import { WGS84, geodeticToGeocentricLat } from './math/wgs84'
import { Timeline, COBB_LINE, T_END, farWallVisibilityAt } from './sweep/Timeline'
import { playbackSeconds } from './sweep/pacing'
import { createSweepGuide } from './hud/sweepGuide'
import { createDissolveUniforms } from './sweep/dissolveNodes'
import { buildMatterMaterial } from './sweep/Matter'
import { buildParticleSystem, type ParticleSystem } from './regimes/Particles'
import {
  buildSplatPipeline, buildCompositeMaterial, buildDebugViewMaterial, buildDepositMaterial,
  buildDepositMaterialCornerVariantForTesting, buildResolveMaterial, type SplatPipeline,
} from './regimes/SplatBlur'
import { gaussianTaps } from './math/splat'
import { particleWindowIntersectsShell, partitionAt, REGIME2_TOP_ALT, REGIME2_FULL_ALT } from './regimes/Blend'
import type { TslFieldPort } from './field/probeField'
import { exaggerationAt } from './math/descent'
import { sigmaPixAt, cellSizeForBudget } from './math/lod'
import { COMPOSITION_CAPTION } from './field/cobb'
// `MEASUREMENT_CAPTION` is no longer read here: its nested-parenthesis form
// was the defect, and hud/copy.ts now carries the same figures across three
// typographic tiers instead. The constant stays exported from field/cobb.ts
// beside the coefficients it describes.
import {
  HUD_COPY, KEY_GROUP_TITLES, ALWAYS_VISIBLE_HINTS,
  ABOUT_LINK_TEXT, ABOUT_LINK_HREF, NO_WEBGPU_COPY,
} from './hud/copy'
import { syntheticTau } from './field/synthetic'
import {
  UNCERTAINTY_STOPS, UNCERTAINTY_CV_LOW, UNCERTAINTY_CV_HIGH, formatUncertaintyCv,
} from './field/terrain'
import { estimatedCountInView, formatCount, humanComparison, GLOBAL_MEAN_DENSITY_PER_SQM } from './field/countEstimate'

const hud = document.getElementById('hud')!
const viewerPolicy = createViewerPolicy()

/**
 * Put the no-WebGPU page on screen, optionally replacing its reason line.
 *
 * The page itself is STATIC MARKUP in viewer.html — `<main id="no-webgpu">`,
 * revealed by a `data-webgpu` attribute a classic script in `<head>` stamps
 * before the body is parsed. This module builds none of it, deliberately: it
 * is an ES module with top-level `await` in it, and a browser old enough to
 * lack WebGPU is exactly the browser that may fail to run it. A fallback
 * assembled by the script that cannot run is not a fallback.
 *
 * What this function is for is the OTHER failure: `navigator.gpu` present but
 * no adapter or device obtainable. The head script cannot see that — only an
 * attempted `requestAdapter` can — so this stamps the same attribute by hand
 * and swaps in the reason that actually applies.
 */
const showNoWebGPU = (reason?: string): void => {
  document.documentElement.setAttribute('data-webgpu', 'no')
  if (reason === undefined) return
  const reasonEl = document.getElementById('no-webgpu-reason')
  // A missing element here means the markup and this module have drifted
  // apart. Say so rather than silently showing the wrong reason.
  if (reasonEl) reasonEl.textContent = reason
  else console.error('no-webgpu-reason element is missing from viewer.html')
}

if (!isWebGPUAvailable()) {
  // Already on screen: the head script stamped data-webgpu="no" before this
  // module was even fetched. This call is the belt to that braces, and costs
  // one attribute write.
  showNoWebGPU()
} else {
  const TAU_MEAN = 0.047
  // The count-in-view estimator's global-mean density for SYNTHETIC mode -
  // field/countEstimate.ts's GLOBAL_MEAN_DENSITY_PER_SQM is the real
  // pipeline's published-total/area figure, meaningless against the
  // synthetic field, so this is its analogue: TAU_MEAN converted to a
  // density via tau = E_A * N_A (math/optical.ts).
  const SYNTHETIC_MEAN_DENSITY_PER_SQM = TAU_MEAN / E_A
  const CUBE_FACE_SIZE = 512
  // Sized for the LARGEST the descent schedule ever asks for (Hs maxes out
  // at 32 km, Hmax = 6*Hs = 192 km - see math/descent.ts's scaleHeightAt),
  // so it stays a safe outer bound as Hs/Hmax shrink live with descent: the
  // march's own analytic outer-bound test already uses the LIVE Hmax
  // uniform, independent of this fixed mesh radius.
  const H_MAX_INITIAL = 192000

  const canvas = document.createElement('canvas')
  document.body.appendChild(canvas)
  /**
   * `navigator.gpu` being present is not the same as a device being
   * obtainable: `requestAdapter` returns null on a blocklisted driver, and
   * device creation throws on a GPU process that has died. That visitor used
   * to get an unhandled rejection and a permanently black page — strictly
   * worse than the one sentence the no-WebGPU branch used to show, because
   * there was nothing on it at all. Both failures now land on the same page.
   */
  let renderer: THREE.WebGPURenderer
  try {
    renderer = await createRenderer(canvas)
  } catch (err: unknown) {
    console.error('WebGPU is present but no device could be created:', err)
    canvas.remove()
    showNoWebGPU(NO_WEBGPU_COPY.adapterFailure)
    // Nothing below this line can run without a renderer, and there is no
    // `return` to be had at module top level. Module evaluation stops here
    // with the fallback already on screen, which is the point; the throw is
    // what carries the cause into the console and window.__rejections.
    throw new Error(
      `WebGPU device creation failed: ${err instanceof Error ? err.message : String(err)}`,
    )
  }

  const scene = new THREE.Scene()
  // Keep space opaque for consistent tone mapping with and without the particle composite pass.
  scene.background = new THREE.Color(0x000000)

  const Hs = uniform(32000)
  const Hmax = uniform(H_MAX_INITIAL)
  const s0 = uniform(sigma0(TAU_MEAN, 32000, H_MAX_INITIAL))
  const sigma0PerTau = uniform(sigma0(1, 32000, H_MAX_INITIAL))
  const tauCap = uniform(saturationCapTau(32000))
  const cameraWorldPosition = uniform(new THREE.Vector3())
  // regimes/Blend.ts's `1 - handoffWeight(altitude)`, live - see
  // ShellPartitionOpts's doc for why this single uniform is the entire
  // Nelson Max coupling on the shell side. 1 = unpartitioned (Session 1-3
  // behaviour); updated each frame once regime-2 comparison mode is active.
  const shellPartitionShare = uniform(1)
  const farWallEnabled = uniform(false)
  const farWallVisibility = uniform(0)
  // The Cobb inversion slider (spec 6.3, corrected — see Shell.ts's
  // ShellOpts.urban doc and docs/superpowers/results/2026-08-16-urban-
  // claim-ruling.md): 0 = the measurement (no urban effect), 1 = Cobb's
  // 1914 prediction. Only ever applied to the REAL shell material, never
  // synthetic — GHSL built-up data has no synthetic-field analogue, and
  // synthetic stays byte-identical to its tested-reference behaviour.
  const cobbMixUniform = uniform(0)
  // The model-uncertainty overlay (`u`): 0 = off, 1 = on. A uniform, not a
  // material rebuild — at 0 the overlay term is an exact no-op (Shell.ts's
  // ShellUncertaintyOpts). Real shell only: the B channel it reads is a
  // measured bootstrap CV, and the synthetic field has no uncertainty to
  // report because it is not a measurement.
  const uncertaintyStrength = uniform(0)

  // Forward hooks into the Cobb control, which is built far below but has to
  // be told about two things that happen above it: the active data source
  // changing, and the GHSL urban atlas landing. Both are the difference
  // between the slider doing something and the slider being inert, and the
  // HUD used to claim the former in both cases. No-ops until the control
  // exists, so an atlas that resolves unusually early cannot throw.
  let updateCobbAvailability: () => void = () => {}
  let markUrbanAtlasReady: () => void = () => {}

  // ~350-450 ms of synchronous CPU work (benchmarked in Session 2's cube-bake
  // task) - a one-off startup cost, fine for a preview page, not something
  // that belongs in the per-frame render loop.
  const cubeTex = bakeSyntheticCubeTexture({ size: CUBE_FACE_SIZE })
  // Same bytes as cubeTex, packed as a DataArrayTexture instead of six cube
  // faces — regime 2's compute-kernel field port (below) samples THIS, not
  // cubeTex: CubeTextureNode cannot run inside compute (see
  // field/cubeArrayNodes.ts's doc for the confirmed mechanism). Regime 1
  // (the shell material just below) keeps sampling cubeTex directly, in the
  // fragment stage, where cubeTexture() works fine and gets seamless
  // cross-face filtering for free.
  const densityArrayTex = bakeSyntheticDensityArrayTexture({ size: CUBE_FACE_SIZE })

  const syntheticMaterial = buildShellMaterial({
    Hs, Hmax, sigma0: s0,
    cube: { texture: cubeTex, sigma0PerTau, tauCap },
    cameraWorldPosition,
    partition: { shellFraction: shellPartitionShare },
    farWall: { enabled: farWallEnabled, visibility: farWallVisibility },
    nMax: 128,
  })
  // Outer shell bound as geometry: the march runs between this backface and
  // the camera, so the sphere supplies the bound with no custom code.
  // Material starts synthetic (always instantly available) and is swapped to
  // the real atlas once it loads — see the "data source" block below.
  const shell = new THREE.Mesh(
    new THREE.SphereGeometry(WGS84.a * (1 + H_MAX_INITIAL / WGS84.a), 128, 64),
    syntheticMaterial,
  )
  scene.add(shell)

  // Matter: opaque, rendered before the transparent shell (three.js's
  // default draw order already puts opaque objects first), so it occludes
  // the veil via normal depth test/write until the dissolve discards it away.
  // Real ETOPO elevation + land fraction, real relief, real hypsometric
  // colour, real bathymetry — see sweep/Matter.ts's doc.
  //
  // Three independent fronts, one per dissolve layer (atmosphere/water/rock),
  // each a SEPARATE DissolveUniforms instance (same seed, own `front` value)
  // since a single instance only carries one front - see Matter.ts's doc for
  // how each is used (atmosphere -> rim intensity, water -> ocean
  // discard+bathymetry displacement, rock -> land discard).
  const atmosphereDissolve = createDissolveUniforms()
  const waterDissolve = createDissolveUniforms()
  const rockDissolve = createDissolveUniforms()
  const bathymetryUniform = uniform(0)
  // k(alt): math/descent.ts's exaggerationAt, ramped 3.0(ship)..1.0 on
  // descent, same schedule the shell reports as `state.k` in the HUD.
  const exaggerationUniform = uniform(exaggerationAt(1.15e7))

  // Segment counts come from field/groundHeight.ts, which reproduces this
  // exact vertex grid on the CPU so the camera's ground floor is measured
  // against the mesh that is actually drawn rather than against the atlas.
  const MATTER_GEOMETRY = new THREE.SphereGeometry(WGS84.a, MATTER_WIDTH_SEGMENTS, MATTER_HEIGHT_SEGMENTS)

  // Instantly-available placeholder textures (2x2, flat/zero) so matter
  // material construction never blocks on the network - matches the
  // shell's synthetic-first pattern. Swapped for the real elevation +
  // density (land-fraction) cubes once both load, a few requests below.
  //
  // Both `elevationCubeTex` and `realCube` are declared here (not lazily
  // where each first loads) specifically so `matterMaterial`'s FIRST,
  // SYNCHRONOUS call below (building the instant placeholder) can close
  // over them safely - a `let` declared later in the same block would
  // still be in its temporal dead zone at that point, unlike the async
  // `.then()` callbacks elsewhere in this file, which only ever run after
  // the whole synchronous script body (all `let`s included) has completed.
  const placeholderTerrainCube = bakeSyntheticCubeTexture({ size: 2 })
  let texelAngularWidthRad = 0.01 // overwritten once the real elevation sidecar loads
  // The WHOLE LoadedCube for elevation, not just its texture: `faces` is the
  // CPU-side copy the camera's ground floor samples (field/groundHeight.ts).
  // It costs nothing extra to retain — the elevation atlas skips the tau
  // re-encode, so its DataTextures already hold these very arrays.
  let elevationCube: LoadedCube | null = null
  let urbanCubeTex: THREE.CubeTexture | null = null
  let realCube: LoadedCube | null = null
  const matterMaterial = () => {
    // Per-slot, independent (sweep/matterAtlases.ts): whatever has landed is
    // bound now, and a slot still in flight — or permanently failed — costs
    // only its own placeholder, never anyone else's atlas.
    const bindings = matterAtlasBindings({
      elevation: elevationCube?.texture ?? null,
      density: realCube?.texture ?? null,
      // Built-up surface drives the city lights and the Cobb illustration.
      urban: urbanCubeTex,
    }, placeholderTerrainCube)
    return buildMatterMaterial({
      atmosphere: atmosphereDissolve,
      water: waterDissolve,
      rock: rockDissolve,
      bathymetry: bathymetryUniform,
      exaggeration: exaggerationUniform,
      elevationTexture: bindings.elevation,
      densityTexture: bindings.density,
      urbanTexture: bindings.urban,
      texelAngularWidthRad,
    })
  }

  // Held in its own binding, not read back off the mesh: `Mesh.material` is
  // typed as the base `Material` (or an array of them), and the ladder needs
  // the concrete NodeMaterial to hand to `replaceMaterial`.
  let matterMaterialCurrent = matterMaterial()
  const matter = new THREE.Mesh(MATTER_GEOMETRY, matterMaterialCurrent)
  scene.add(matter)

  /**
   * Rebuilds matter's material against whatever atlases have landed. Called
   * from each of the three independent `.then()`s below, EVERY time one
   * succeeds — material binding never waits for unrelated atlases. The
   * opening's playback gate separately waits for density and elevation.
   *
   * There used to be: `if (!(elevationCubeTex && realCube && urbanCubeTex))
   * return`, an all-or-nothing AND across three unrelated network loads. A
   * failed density cube or a failed urban ladder silently threw away a
   * fully-downloaded elevation atlas — 192 MiB fetched, decoded, land-mask
   * validated, and then dropped, with nothing on screen changing. See
   * `sweep/matterAtlases.ts` for the tested contract that replaced it, and
   * `logAtlasBind` for why every successful bind now leaves a trace: three
   * separate defects in this atlas path have been silent losses rather than
   * crashes.
   */
  const rebuildMatter = (): void => {
    // Each rebuild is a whole new NodeMaterial (and, on first draw, a whole
    // new compiled pipeline), so the superseded one has to be released or
    // every ladder tier leaks one. Rebuild-then-dispose, never the reverse -
    // core/materialSwap.ts.
    matterMaterialCurrent = replaceMaterial(matterMaterialCurrent, matterMaterial(), (next) => {
      matter.material = next
    })
  }

  /**
   * One line per atlas that actually reached the GPU. Not decoration: the
   * failure mode this whole path keeps producing is a load that succeeds
   * and is then discarded without a word, which is indistinguishable from a
   * slow network unless the successes are as visible as the failures.
   */
  const logAtlasBind = (what: string, url: string, faceSize: number): void => {
    // eslint-disable-next-line no-console
    console.log(`[atlas] bound ${what} from ${url} (${faceSize}-texel faces)`)
  }

  /**
   * Elevation atlas tiers, coarsest first — a progressive ladder, not one
   * fetch. The sharp tier is what actually fixes visible blur (1024 faces =
   * 9.8 km/texel against 256's 39.1 km/texel, a 4x linear gain in both the
   * hypsometric ramp and, more visibly, `terrainNormalNode`'s relief
   * shading), but it is also ~16x the bytes, and a cold open that stares at
   * a flat placeholder while it downloads is a worse first impression than
   * a slightly soft planet that is on screen immediately.
   *
   * Sequential, not `Promise.all`: the tiers compete for the same
   * connection, and fetching the coarse one alongside the sharp one delays
   * exactly the thing it exists to make fast. Sequencing also makes "a
   * later tier never loses to an earlier one that resolved out of order"
   * true by construction rather than by a comparison at assignment time.
   */
  /**
   * The 6,825 places someone actually put a corer in the ground, from van den
   * Hoogen et al. 2020 (Sci Data 7:103, CC0). Off by default and bound to `s`:
   * this is a "show me the measurements" mode, not part of the piece's own
   * imagery, and it undercuts the render deliberately when you turn it on —
   * 86.6 KB of measurement against 81 MB of interpolation between them.
   *
   * Failure here is non-fatal by design. The overlay is commentary on the
   * globe rather than part of it, so a missing artefact should cost you the
   * commentary and nothing else.
   */
  const timeline = new Timeline()
  const scaleCamera = new ScaleCamera()
  const opening = new OpeningExperience(scaleCamera, timeline)
  let samplePoints: SamplePointsLayer | null = null
  let showSamples = false
  let showUncertainty = false
  let sampleLoadFailed = false
  let updateLayerStatus = (): void => {}
  loadSamplePoints('/data/nematode_samples_v2').then((layer) => {
    samplePoints = layer
    layer.points.visible = showSamples
    scene.add(layer.points)
    updateLayerStatus()
  }).catch((err: unknown) => {
    sampleLoadFailed = true
    updateLayerStatus()
    console.error('sample points failed to load — the measurement overlay is unavailable:', err)
  })

  /**
   * Which atlas tiers this client is allowed to fetch — decided ONCE, here,
   * from a single snapshot of the viewport and connection.
   *
   * Once, and never again, is the whole point. The reading is taken before
   * any atlas request goes out and is not revisited on resize, so a phone
   * turned from portrait to landscape cannot start a 31.7 MiB download
   * mid-session. See `platform/atlasLadder.ts` for the ruling itself and the
   * measurement behind it.
   */
  const atlasLadder = planAtlasLadder(readClientBudgetHints())
  const ladderSizes = (tiers: ReadonlyArray<{ faceSize: number }>): string =>
    tiers.map((t) => t.faceSize).join('/')
  console.info(
    `[atlas] ladder cap=${atlasLadder.cap} · elevation ${ladderSizes(atlasLadder.elevation)}` +
      ` · urban ${ladderSizes(atlasLadder.urban)}`,
  )

  // Coarsest first: 256 paints fast, 1024 is the working tier, 2048 is the
  // top and is withheld from small or slow clients by the plan above.
  void opening.loadEssential('elevation', atlasLadder.elevation.map(({ url }) => url), async (url) => {
    try {
      const loaded = await loadCubeTexture(url)
      const superseded = elevationCube
      elevationCube = loaded
      if (loaded.meta.km_per_texel !== undefined) {
        texelAngularWidthRad = (loaded.meta.km_per_texel * 1000) / WGS84.R_MEAN
      }
      rebuildMatter()
      logAtlasBind('elevation', url, loaded.meta.face_size)
      // Only safe AFTER the rebuild above: until then the live material
      // is still sampling this texture.
      superseded?.texture.dispose()
    } catch (err: unknown) {
      // A failed upgrade leaves the last bound tier usable. The opening
      // controller reports failure only if the entire ladder is unavailable.
      console.error(`elevation tier ${url} failed to load:`, err)
      throw err
    }
  })

  /**
   * Urban tiers, coarsest first — the same progressive ladder the elevation
   * atlas uses, and for the same reason: the sharp tier is what makes cities
   * read as cities rather than as smudges, but it should not hold up first
   * paint.
   *
   * This is the largest source-to-baked gap in the project. GHSL's source is
   * 30 arc-second (0.927 km/px) and the 512 tier bakes it at 19.55 km/texel,
   * discarding about 21x of linear detail; 1024 halves that to 9.77 km/texel.
   * City lights are the sharpest-featured thing on screen and the setup half
   * of the piece's central rhyme (spec 6.3), so this is where sharpening buys
   * the most per byte — and it is the honest kind: a finer SAMPLING of data
   * already shipped, not a new dataset and not a new claim. The provenance
   * string and DOI are unchanged.
   *
   * Contrast the density cube, which must NOT be re-baked finer: it is
   * already finer than its source at face corners, and its source's N-S axis
   * is a 2.4% row sample.
   *
   * Chained with `.then()` rather than an async IIFE deliberately: the
   * callbacks below read `activeSource`/`realMaterial`/`buildRealShellMaterial`,
   * which are declared further down this function, and a `.then()` callback
   * is the form the checker treats as genuinely deferred.
   */
  // Built-up fraction is mostly exact zeros, so it compresses ~29x and the
  // whole ladder costs ~9 MiB brotli against elevation's ~41 MiB. That is why
  // `atlasLadder` leaves this one at full length on a small screen and cuts
  // it only when the visitor has actually asked for fewer bytes — the
  // argument is written out in platform/atlasLadder.ts.
  atlasLadder.urban.reduce(
    (chain, { url }) => chain.then(() => loadCubeTexture(url).then((loaded) => {
      const superseded = urbanCubeTex
      urbanCubeTex = loaded.texture
      rebuildMatter()
      // The Cobb slider's real material is built with whatever urbanCubeTex
      // was available AT THAT TIME (see buildRealShellMaterial's doc), so it
      // is rebuilt per tier — otherwise the slider stays bound to the coarse
      // one after the sharp tier lands. Each rebuild supersedes a whole
      // NodeMaterial and its pipeline; this used to drop them on the floor
      // (two leaked per session) while the elevation path beside it disposed
      // correctly. Same rebuild-then-dispose ordering as everywhere else now.
      if (realCube) {
        realMaterial = replaceMaterial(realMaterial, buildRealShellMaterial(realCube), (next) => {
          if (activeSource === 'real') shell.material = next
        })
      }
      logAtlasBind('urban (GHSL)', url, loaded.meta.face_size)
      // The Cobb axis has something to enrich from here on, so the control
      // stops saying it hasn't.
      markUrbanAtlasReady()
      // Only after both consumers above are rebuilt: until then the live
      // materials are still sampling the superseded texture.
      superseded?.dispose()
    })).catch((err: unknown) => {
      console.error(`urban (GHSL) tier ${url} failed to load:`, err)
    }),
    Promise.resolve(),
  )

  // --- Regime 2: compute-instanced particles, foundation pass ------------
  //
  // NOT the shipped scale: budget/maxCandidates here are sized for a
  // comparison harness on one machine, not spec 7.7's 4e6 budget - see
  // regimes/Particles.ts's doc comment for the full list of simplifications
  // (single-level lattice, symmetric candidate box, direct altitude,
  // vertical transmittance). Bound to the SYNTHETIC cube always (not
  // whichever of real/synthetic is currently active for the shell): the
  // compute kernel's field sampling is fixed at construction time, and the
  // synthetic field is always available immediately, which matters more
  // for a comparison tool than matching the shell's current data source.
  // Array-sampled, not cube-sampled — see densityArrayTex's doc above.
  const syntheticFieldPort: TslFieldPort = {
    tauNadirAt: (dir) => exp(sampleDensityArrayNode(densityArrayTex, dir)),
  }
  const PARTICLE_BUDGET = 200_000
  const particles: ParticleSystem = buildParticleSystem({
    field: syntheticFieldPort,
    budget: PARTICLE_BUDGET,
    maxCandidates: 800_000,
  })
  const splat: SplatPipeline = buildSplatPipeline(particles, window.innerWidth, window.innerHeight, 3)
  const compositeMaterial = buildCompositeMaterial(splat.resolveTarget.texture)
  // Reuses splat.resolveCamera rather than a fresh one - see
  // SplatBlur.ts's makeFullscreenCamera doc for why every full-screen pass
  // in this pipeline shares one stateless orthographic camera.
  const compositeMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), compositeMaterial)
  const compositeScene = new THREE.Scene()
  compositeScene.add(compositeMesh)

  // Diagnostic, kept permanently (not removed after use) — see
  // SplatBlur.ts's buildDebugViewMaterial doc: draws depositTarget's raw,
  // unblurred, uncomposited content straight to screen behind the 'v' key,
  // to separate "nothing was deposited" from "deposit happened and the
  // resolve pass lost it".
  const debugDepositMaterial = buildDebugViewMaterial(splat.depositTarget.texture)
  const debugDepositMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), debugDepositMaterial)
  const debugDepositScene = new THREE.Scene()
  debugDepositScene.add(debugDepositMesh)
  let debugDeposit = false
  let particleDrawReady = false
  const particleKernels = [particles.resetKernel, particles.emitKernel, particles.finalizeKernel]

  type RegimeMode = 'shell' | 'particles' | 'blended'
  let regimeMode: RegimeMode = 'blended'

  document.body.classList.add('viewer-exhibit')
  const camera = new THREE.PerspectiveCamera(
    verticalFovForViewport(window.innerWidth, window.innerHeight),
    window.innerWidth / window.innerHeight, 1, 1e9,
  )
  let zoomQuality: ZoomQuality | null = null
  let projectionViewport = { width: window.innerWidth, height: window.innerHeight }
  const syncZoomQuality = (): void => {
    // CSS dimensions and the just-framed projection, never the DPR-scaled
    // render buffer. Reserve accepted soil detail while it loads; if a
    // coarser soil fallback is bound, its actual face size tightens the floor.
    const next = zoomQualityForView(camera, projectionViewport.width, projectionViewport.height, realCube?.meta.face_size)
    if (!next) return
    zoomQuality = next
    scaleCamera.qualityFloorM = next.floorM
  }
  syncZoomQuality()

  /**
   * The camera's ground floor (`core/ScaleCamera.ts`'s `groundHeightAt`).
   *
   * Mirrors what the vertex shader is doing THIS frame, not what the atlas
   * says in the abstract: `exaggerationUniform` and `bathymetryUniform` are
   * the very uniforms `sweep/Matter.ts` displaces by, so the CPU floor and
   * the GPU surface cannot drift apart as the descent schedule or the
   * collapse beat move them. Bound to `elevationCube` lazily, so it costs
   * nothing until an atlas has landed and it follows whichever ladder tier
   * is currently on screen.
   *
   * The physical floor releases once matter is gone. The separate dataset
   * quality floor remains throughout the sweep, including over empty ocean.
   */
  let groundFloorEnabled = true
  scaleCamera.groundHeightAt = (dir) => {
    // null, not 0 — see GroundHeightAt's doc. "No atlas yet" and "matter is
    // gone" are both "nothing to stand on", not "the ground is at sea level".
    if (!groundFloorEnabled || !elevationCube) return null
    if (timeline.state.matter <= 0) return null
    return cellMaxDisplacementM(elevationCube.faces, elevationCube.meta, dir, {
      widthSegments: MATTER_WIDTH_SEGMENTS,
      heightSegments: MATTER_HEIGHT_SEGMENTS,
      exaggeration: exaggerationUniform.value,
      bathymetryProgress: bathymetryUniform.value,
    })
  }

  addEventListener('resize', () => {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    // Splat targets use CSS pixels; the composite scales to the drawing
    // buffer. Keep the same CSS grid after a resize or orientation change.
    splat.depositTarget.setSize(window.innerWidth, window.innerHeight)
    splat.blurHTarget.setSize(window.innerWidth, window.innerHeight)
    splat.resolveTarget.setSize(window.innerWidth, window.innerHeight)
  })

  // --- interaction: drag to orbit, pinch or scroll to descend ---------
  //
  // A gentle idle orbit begins at first paint. Gestures immediately stop it,
  // pause the sweep, and yield the view; playback keeps the camera still.
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)')
  opening.setReducedMotion(reducedMotion.matches)
  reducedMotion.addEventListener('change', (event) => opening.setReducedMotion(event.matches))

  /**
   * One finger orbits, two descend — the state machine lives in
   * `core/pointerGesture.ts`, free of DOM types so it can be tested against a
   * real `ScaleCamera` without a browser. This is only the wiring.
   *
   * Descent used to be bound to `wheel` alone, so on a phone the interaction
   * the whole piece is built around did not exist: visitors could spin the
   * globe and nothing else. Both inputs now produce TICKS and reach the
   * camera through the same `scrollBy`, so they cannot drift apart and the
   * ground floor in `ScaleCamera.groundFloor` binds both.
   *
   * viewer.html's `canvas { touch-action: none }` is load-bearing here:
   * without it the browser claims the two-finger gesture for its own page
   * zoom and cancels the pointers part-way through the descent.
   */
  const gesture = new PointerGesture()

  canvas.addEventListener('pointerdown', (e) => {
    opening.explore()
    gesture.down({ id: e.pointerId, x: e.clientX, y: e.clientY })
    // Per pointer id, so a second finger is captured independently of the
    // first. Capture keeps delivering move/up to the canvas even once a
    // pointer leaves its bounds mid-gesture, so no window-level listener is
    // needed for that case.
    canvas.setPointerCapture(e.pointerId)
  })
  canvas.addEventListener('pointermove', (e) => {
    const { dxPx, dyPx, ticks } = gesture.move({ id: e.pointerId, x: e.clientX, y: e.clientY })
    if (ticks !== 0) scaleCamera.scrollBy(ticks)
    if (dxPx !== 0 || dyPx !== 0) {
      const { dLon, dLat } = dragToOrbit(dxPx, dyPx, Math.min(window.innerWidth, window.innerHeight))
      scaleCamera.orbitBy(dLon, dLat)
    }
  })
  const endPointer = (e: PointerEvent): void => {
    gesture.up(e.pointerId)
    // hasPointerCapture guards the double-release that `pointercancel`
    // followed by `pointerup` would otherwise attempt.
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId)
  }
  canvas.addEventListener('pointerup', endPointer)
  canvas.addEventListener('pointercancel', endPointer)
  // Belt to the two above. The spec guarantees every captured pointer ends in
  // an up or a cancel, but capture can also be lost without either — the
  // element being removed, or the browser reclaiming the pointer — and a
  // pointer left in the map when the finger is gone makes the tracker believe
  // two fingers are down forever, which kills orbit until a reload. `up` is
  // idempotent, so on the ordinary path this fires and does nothing.
  canvas.addEventListener('lostpointercapture', (e) => { gesture.up(e.pointerId) })

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault() // canvas fills the viewport; nothing else should scroll
    opening.explore()
    scaleCamera.scrollBy(wheelDeltaToTicks(e.deltaY))
  }, { passive: false })

  // --- HUD ---------------------------------------------------------------
  // Built with DOM nodes rather than innerHTML. Every value here is locally
  // computed so there is no injection path, but textContent is free and the
  // project already ruled against innerHTML for generated content. Live
  // values get their own persistent element and are updated via textContent
  // each frame, rather than rebuilding the whole HUD subtree every frame.
  const line = (...parts: Array<string | Node>): HTMLDivElement => {
    const div = document.createElement('div')
    for (const p of parts) div.append(p)
    return div
  }
  const val = (s = ''): HTMLElement => {
    const b = document.createElement('b')
    b.textContent = s
    return b
  }
  // The dim tier. Was `style.opacity = '0.65'`, which composited to
  // rgb(83,95,105) over the page background — 3.10:1, below WCAG AA's 4.5:1
  // before anything was even drawn behind it. An opacity is also unmeasurable
  // on its own, since its effective colour depends on whatever it lands over;
  // an explicit token can be checked once and holds everywhere. See
  // viewer.html's palette block for the measured figures.
  const faint = (s: string): HTMLElement => {
    const span = document.createElement('span')
    span.className = 'hud-dim'
    span.textContent = s
    return span
  }
  const el = (tag: string, className: string, text = ''): HTMLElement => {
    const node = document.createElement(tag)
    node.className = className
    if (text) node.textContent = text
    return node
  }

  const formatAltitude = (m: number): string => {
    if (m >= 1000) return `${(m / 1000).toFixed(m >= 1e5 ? 0 : 1)} km`
    return `${m.toFixed(m >= 1 ? 1 : 3)} m`
  }

  const altVal = val()
  const hsVal = val()
  const kVal = val()
  const tauVal = val()
  const qVal = val()
  const beatVal = val()
  const cobbVal = val()
  const dataSourceVal = val()
  const dataStatusFaint = faint('')
  // The two bindings that change what the piece CLAIMS, kept beside the
  // readout they change rather than inside the `?` panel — see the key panel
  // block below for the argument.
  const evidenceHints = faint('')
  const probesLine = document.createElement('div')
  const regimeModeVal = val()
  const regimeStatsFaint = faint('')
  const cobbModeVal = val()

  const expLine = document.createElement('div')

  // --- count in view (spec 8.3): "the piece's emotional device" ----------
  //
  // Everything else in the HUD is said to serve this one number - watching
  // it fall from ~1e19 at orbit to a handful at a centimetre is what turns
  // 4.4e20 from a statistic into a feeling. See field/countEstimate.ts for
  // the estimator itself.
  //
  // The label and the comparison now sit on their own lines at their own
  // sizes rather than running inline at 0.65 opacity beside the value: the
  // comparison is a whole sentence and was the least legible text in the
  // HUD while being the part that does the actual translating.
  const countVal = el('span', 'hud-count-value')
  const countCompare = el('span', 'hud-count-compare')
  const countLine = el('div', 'hud-count')
  const countLabel = el('span', 'hud-count-label', HUD_COPY.countLabel)
  countLine.append(countLabel, countVal, countCompare)

  // --- diagnostics: everything past what a first-time viewer needs -------
  //
  // Hidden by default (press i to reveal) - this was right while debugging
  // and is wrong for a viewer, per the brief: "this deserves to be the
  // thing you see," not one line among ten.
  const diagnostics = document.createElement('div')
  diagnostics.style.display = 'none'
  diagnostics.append(
    // Was the visible title's second half ("Nematode Earth — Regime 1 shell,
    // interactive"). A build note is not a title; it belongs here.
    line(faint('Regime 1 shell, interactive')),
    line(
      'H_s ', hsVal,
      '   k ', kVal,
      '   tau_rendered ', tauVal,
      '   Q ', qVal,
    ),
    line(faint('density atlas'), dataStatusFaint),
    probesLine,
    line(faint('City lights show built-up surface from GHSL. The soil estimates come from a separate map.')),
    line(faint(COMPOSITION_CAPTION)),
    line('sweep ', beatVal),
    line('regime 2 mode ', regimeModeVal, regimeStatsFaint),
    expLine,
  )

  // --- uncertainty overlay legend (`u`) ----------------------------------
  //
  // A colour ramp with no scale communicates "some places are more orange"
  // and nothing else, so the swatch below is drawn from the SAME
  // UNCERTAINTY_STOPS the shader ramps through (field/terrain.ts) and
  // labelled with the actual quantity in its actual units — a bootstrap
  // coefficient of variation, i.e. the standard deviation of the 100-
  // iteration bootstrap distribution as a percentage of its own mean.
  // Hidden unless the overlay is on.
  const uncertaintyLine = document.createElement('div')
  uncertaintyLine.style.display = 'none'
  const rampSwatch = document.createElement('span')
  rampSwatch.style.cssText = [
    'display:inline-block', 'width:7rem', 'height:0.7em', 'vertical-align:middle',
    'margin:0 0.4rem', 'border:1px solid rgba(255,255,255,0.25)',
    `background:linear-gradient(to right, ${UNCERTAINTY_STOPS.map(([cv, r, g, b]) => {
      const pct = ((cv - UNCERTAINTY_CV_LOW) / (UNCERTAINTY_CV_HIGH - UNCERTAINTY_CV_LOW)) * 100
      const ch = (c: number): number => Math.round(c * 255)
      return `rgb(${ch(r)},${ch(g)},${ch(b)}) ${pct.toFixed(1)}%`
    }).join(', ')})`,
  ].join(';')
  uncertaintyLine.append(
    'Variation across model runs ',
    faint('(coefficient of variation) '),
    val(formatUncertaintyCv(UNCERTAINTY_CV_LOW)),
    rampSwatch,
    val(formatUncertaintyCv(UNCERTAINTY_CV_HIGH)),
    faint(' · oceans are outside the soil model'),
  )

  // --- the finding -------------------------------------------------------
  //
  // Was one run-on line with NESTED parentheses, shown only while the Cobb
  // slider sat at its data end:
  //   cobb inversion data (our own measurement: 243,294 native-resolution
  //   pixels · 20 cities · a = -0.0085 (95% CI -0.035, 0.019) · R² = -0.0003)
  // Three tiers now do the work the parentheses were doing, and it is a
  // standing statement rather than a caption on a control: a lay reader
  // stops after the second line and has the whole point; the third line
  // keeps every figure, including the two unflattering ones, so it stays
  // falsifiable. See hud/copy.ts and test/hudCopy.test.ts.
  const finding = el('div', 'hud-finding')
  /**
   * The link out, attached to the FINDING rather than to the hint line.
   *
   * It used to live at the end of `.hud-hints`, which is the last child of
   * #hud and therefore the first thing #hud's overflow clip removes -- see
   * the ranking argument in viewer.html, which puts the key hints at the
   * bottom of the list deliberately because they are the utility tier. The
   * about link is not a hint and does not belong in that tier: it is the only
   * route out of the viewer, and on a short viewport -- a phone held
   * sideways, most obviously -- ranking it as utility deleted it. It renders
   * exactly as before; it is only clipped later, and now sits directly under
   * the claim whose methods it explains.
   */
  const about = document.createElement('a')
  about.className = 'hud-about'
  about.href = ABOUT_LINK_HREF
  about.hash = 'city-comparison'
  about.textContent = ABOUT_LINK_TEXT
  finding.append(
    el('div', 'hud-finding-lede', HUD_COPY.findingPrediction),
    el('div', 'hud-finding-result', HUD_COPY.findingResult),
    el('div', 'hud-finding-stats', HUD_COPY.findingStatistics),
    (() => {
      const row = el('div', 'hud-finding-link')
      row.append(about)
      return row
    })(),
  )

  // --- key bindings ------------------------------------------------------
  //
  // Grouped rows behind `?`, filtered by the same policy as the handler.
  // Development-only source/regime controls and GPU diagnostics are never
  // advertised in the public viewer.
  const keyPanel = el('div', 'hud-keys')
  keyPanel.style.display = 'none'
  for (const group of ['evidence', 'view', 'dev'] as const) {
    const bindings = viewerPolicy.keyBindings.filter((b) => b.group === group)
    if (bindings.length === 0) continue
    const section = el('div', 'hud-keys-group')
    section.append(el('div', 'hud-keys-title', KEY_GROUP_TITLES[group]))
    for (const b of bindings) {
      const row = el('div', 'hud-keys-row')
      row.append(el('kbd', '', b.key === ' ' ? 'Space' : b.key), b.description)
      section.append(row)
    }
    keyPanel.append(section)
  }

  const quote = el('div', 'hud-quote')
  quote.append(cobbVal) // typed in character-by-character during the cold open (spec 6.1)
  // Held back until the line it attributes has finished typing — an
  // attribution that arrives before its quotation reads as a stray caption.
  // `visibility`, not `display`, so it reserves its space and the block below
  // does not jump when it appears.
  const quoteAttrib = el('div', 'hud-attrib', HUD_COPY.quotationAttribution)
  const quoteSource = document.createElement('a')
  quoteSource.href = 'https://archive.org/details/yoa1914/page/472/mode/2up'
  quoteSource.target = '_blank'
  quoteSource.rel = 'noopener noreferrer'
  quoteSource.textContent = HUD_COPY.quotationAttribution
  quoteSource.title = 'Read Cobb’s original passage, USDA Yearbook 1914, page 472'
  quoteAttrib.replaceChildren(quoteSource)
  quoteAttrib.style.visibility = 'hidden'
  quote.append(quoteAttrib)

  const readouts = el('div', 'hud-readouts')
  const modelSource = document.createElement('a')
  modelSource.href = '/about.html#credits'
  modelSource.title = 'Research, data sources, and limitations'
  modelSource.append(dataSourceVal)
  readouts.append(
    line(altVal, faint(' above sea level')),
    line(modelSource),
  )

  const introduction = el('div', 'hud-introduction')
  introduction.append(
    el('h1', 'hud-headline', 'What if only nematodes remained?'),
    el('p', 'hud-deck', 'Explore the distribution of soil nematodes, microscopic roundworms that live beneath our feet.'),
  )
  const layerStatus = el('div', 'hud-layer-status')
  layerStatus.setAttribute('aria-live', 'polite')
  const story = el('div', 'hud-story')
  story.append(introduction, quote)

  hud.replaceChildren(
    story,
    countLine,
    readouts,
    layerStatus,
  )

  // --- data source: real (default) vs synthetic ---------------------------
  //
  // Both materials share the SAME Hs/Hmax/sigma0/sigma0PerTau/tauCap/
  // cameraWorldPosition uniforms (only `cube.texture` differs), so the
  // per-frame updates in the render loop below apply to whichever is active
  // with no extra bookkeeping — only `shell.material` itself is swapped.
  // Synthetic is up and rendering immediately (baked synchronously above,
  // the tested reference), so there is never a black screen while the real
  // atlas — six requests, ~768 KiB — is in flight; once it lands, "real"
  // (the default) takes over automatically.
  // Full-row density + bootstrap CV; coarse composition has its own support.
  // Acceptance record: docs/superpowers/results/2026-09-22-local-soil-extraction.md.
  const REAL_CUBE_URL = '/data/soil_plain_512_v2'
  type DataSource = 'real' | 'synthetic'
  let preferredSource: DataSource = 'real'
  let activeSource: DataSource = 'synthetic'
  // realCube itself is declared earlier (with the matter setup, above) —
  // matterMaterial()'s first, synchronous call needs it in scope already.
  let realMaterial: THREE.NodeMaterial | null = null
  let realLoadStatus: 'loading' | 'ready' | 'error' = 'loading'
  let realLoadError = ''
  let probeResults: GeoProbeResult[] | null = null

  const updateDataHud = (): void => {
    countLabel.textContent = activeSource === 'real' ? HUD_COPY.countLabel : 'Illustrative count from a synthetic pattern'
    dataSourceVal.textContent = activeSource === 'real'
      ? 'Soil abundance model · van den Hoogen et al., 2019'
      : realLoadStatus === 'loading' ? 'Loading the soil model…' : 'Synthetic test pattern'
    const swapHint = viewerPolicy.sourceHint(activeSource)
    evidenceHints.textContent = [swapHint, HUD_COPY.uncertaintyHint].filter(Boolean).join('  ·  ')
    if (activeSource === 'real' && realCube) {
      dataStatusFaint.textContent = ` ${realCube.meta.face_size}-texel RGBA16Float cube, van den Hoogen et al. 2019`
    } else if (realLoadStatus === 'loading') {
      dataStatusFaint.textContent = ' Showing a synthetic pattern while the soil map loads.'
    } else if (realLoadStatus === 'error') {
      dataStatusFaint.textContent = ` Showing a synthetic pattern. Soil map error: ${realLoadError}`
    } else {
      dataStatusFaint.textContent = ` ${CUBE_FACE_SIZE}-texel R16Float cube, analytic placeholder`
    }
    // The Cobb axis is only meaningful against the real atlas, so its
    // availability changes with this — see updateCobbHud.
    updateCobbAvailability()
    updateLayerStatus()
  }

  // Orientation check (spec-mandated): these four should read back with
  // enormous contrast if the cube's face convention and log-decode are both
  // right. Computed once, CPU-side, off the same decoded bytes the GPU
  // texture was built from — see field/geoProbe.ts.
  const updateProbeHud = (): void => {
    if (!probeResults) return
    const parts: Array<string | Node> = [faint('orientation probes (geodetic, real data): ')]
    probeResults.forEach((p, i) => {
      if (i > 0) parts.push(faint('  ·  '))
      parts.push(val(`${p.name} ${p.density.toExponential(2)}/m²`))
    })
    probesLine.replaceChildren(...parts)
  }

  const applyActiveSource = (source: DataSource): void => {
    activeSource = source
    shell.material = source === 'real' && realMaterial ? realMaterial : syntheticMaterial
    updateDataHud()
  }

  const toggleDataSource = (): void => {
    if (activeSource === 'synthetic' && realCube && realMaterial) {
      preferredSource = 'real'
      applyActiveSource('real')
    } else if (activeSource === 'real') {
      preferredSource = 'synthetic'
      applyActiveSource('synthetic')
    }
    // Otherwise (still on synthetic, real not loaded/failed yet): nothing to switch to.
  }

  updateDataHud()

  // Factory (not a one-off construction) so the real material can be
  // REBUILT with the actual GHSL texture once urban loads, in case it
  // resolves after the density cube - urban is the largest of the three
  // atlases (12 MB) and has no ordering guarantee against this one. Until
  // then, an all-zero placeholder makes the Cobb multiplier an exact no-op
  // (cobbMultiplier(0) = dataMultiplier(0) = 1 either way - see cobb.ts),
  // so there is nothing to visually glitch while it's substituted in.
  const buildRealShellMaterial = (loaded: LoadedCube): THREE.NodeMaterial => buildShellMaterial({
    Hs, Hmax, sigma0: s0,
    cube: { texture: loaded.texture, sigma0PerTau, tauCap },
    cameraWorldPosition,
    partition: { shellFraction: shellPartitionShare },
    farWall: { enabled: farWallEnabled, visibility: farWallVisibility },
    urban: { texture: urbanCubeTex ?? placeholderTerrainCube, cobbMix: cobbMixUniform },
    uncertainty: { strength: uncertaintyStrength },
    nMax: 128,
  })

  void opening.loadEssential('density', [REAL_CUBE_URL], async (url) => {
    realLoadStatus = 'loading'
    realLoadError = ''
    updateDataHud()
    try {
      const loaded = await loadCubeTexture(url)
      realCube = loaded
      syncZoomQuality()
      realMaterial = buildRealShellMaterial(loaded)
      probeResults = KNOWN_PROBES.map((p) => probeGeoLocation(loaded.faces, loaded.meta, p))
      updateProbeHud()
      realLoadStatus = 'ready'
      if (preferredSource === 'real') applyActiveSource('real')
      else updateDataHud()
      // Matter also wants this cube (for vegetationIndex) — bound now, on its
      // own, regardless of whether the other two atlases have landed or ever will.
      rebuildMatter()
      logAtlasBind('density', REAL_CUBE_URL, loaded.meta.face_size)
    } catch (err: unknown) {
      realLoadStatus = 'error'
      realLoadError = err instanceof Error ? err.message : String(err)
      updateDataHud()
      console.error('real cube load failed:', err)
      throw err
    }
  })

  // --- Sweep controls: scrubbable, not merely playable -------------------
  //
  // A SEPARATE top-level element, not appended inside #hud: #hud is
  // deliberately `pointer-events: none` (CSS Session 2) so it never steals
  // the canvas's drag/wheel handlers, and pointer-events is an INHERITED
  // property, so real controls need their own container outside that
  // subtree. Placed at the top of the screen, away from the bottom HUD
  // text block it would otherwise overlap (HUD spans the full width there).
  //
  // The slider - not the cinematic - is what people actually play with, so
  // it is visible and usable from t=0, not gated behind SweepState.uiOpacity
  // (which models the CINEMATIC's own controls fading in only at release,
  // spec 6.1). Wiring that in would work against the slider being the
  // primary interaction surface, so it is deliberately left unused here.
  //
  // The playback dock reserves room for the guide and controls together.
  // Secondary data controls live in the dialog; the header is navigation.
  const topbar = document.createElement('div')
  topbar.id = 'topbar'

  const controls = el('div', 'rail-row')
  controls.id = 'transport'
  controls.setAttribute('aria-label', 'Earth sweep playback')
  const sweepGuide = createSweepGuide()
  // Startup is synthetic; the frame loop refreshes this after the real load.
  sweepGuide.update(timeline.state, false, opening.showIntro)
  const transportRow = el('div', 'transport-row')

  const playButton = document.createElement('button')
  const replayButton = document.createElement('button')
  const scrub = document.createElement('input')
  scrub.type = 'range'
  scrub.min = '0'
  scrub.max = String(T_END)
  scrub.step = '0.01'
  scrub.setAttribute('aria-label', 'Sweep from Earth to nematode veil')
  const scrubVal = el('span', 'rail-label')
  scrubVal.classList.add('transport-time')

  replayButton.textContent = '↺'
  replayButton.className = 'replay-button'
  replayButton.setAttribute('aria-label', 'Replay sweep from opening view')
  replayButton.title = 'Replay sweep from opening view'
  playButton.className = 'play-button'
  replayButton.addEventListener('click', () => { opening.replay() })
  playButton.addEventListener('click', () => { opening.togglePlayback() })
  scrub.addEventListener('input', () => { opening.scrubTo(Number(scrub.value)) })

  const openingControls = el('div', 'opening-controls')
  const openingMenuActions = el('div', 'opening-menu-actions')
  const readinessStatus = el('p', 'opening-readiness')
  readinessStatus.id = 'opening-readiness'
  readinessStatus.setAttribute('role', 'status')
  readinessStatus.setAttribute('aria-live', 'polite')
  readinessStatus.setAttribute('aria-atomic', 'true')
  playButton.setAttribute('aria-describedby', readinessStatus.id)
  replayButton.setAttribute('aria-describedby', readinessStatus.id)
  const retryData = document.createElement('button')
  retryData.type = 'button'
  retryData.className = 'retry-data'
  retryData.textContent = 'Retry data loading'
  retryData.setAttribute('aria-describedby', readinessStatus.id)
  retryData.addEventListener('click', () => {
    const { canRetry, retrying } = opening.readiness
    if (!canRetry || retrying) return
    const recovery = opening.retryDataLoading()
    updateOpeningControls()
    void recovery.then(updateOpeningControls)
  })
  const exploreView = document.createElement('button')
  exploreView.textContent = 'Explore'
  exploreView.setAttribute('aria-label', 'Pause the sweep and explore the globe')
  exploreView.addEventListener('click', () => {
    opening.explore()
    if (explore.open) explore.close()
  })
  const returnView = document.createElement('button')
  returnView.textContent = 'Return to opening'
  returnView.setAttribute('aria-label', 'Return to opening view')
  returnView.title = 'Restore the opening view and keep the current sweep time'
  returnView.addEventListener('click', () => {
    opening.returnToOpening()
    if (explore.open) explore.close()
  })
  const openingActions = el('div', 'opening-actions')
  openingActions.append(exploreView, returnView)
  // Recovery stays beside the visible failure status, even when secondary
  // actions move into the small-screen menu.
  openingControls.append(readinessStatus, retryData, openingActions)
  const updateOpeningControls = (): void => {
    const { canBegin, canRetry, retrying, retryExhausted, message } = opening.readiness
    const layout = openingControlLayout(window.innerWidth, window.innerHeight, canBegin)
    readinessStatus.classList.toggle('sweep-sr-only', !layout.showStatus)
    openingControls.dataset.empty = String(layout.actions === 'menu' && !layout.showStatus)
    controls.dataset.compactGuide = String(layout.compactGuide)
    const actionsParent = layout.actions === 'menu' ? openingMenuActions : openingControls
    if (openingActions.parentElement !== actionsParent) {
      const focused = document.activeElement
      const hadFocus = focused instanceof HTMLElement && openingActions.contains(focused)
      actionsParent.append(openingActions)
      // Rotation/resize can relocate a focused button into a closed dialog.
      // Return focus to its opener instead of leaving it on hidden content.
      if (hadFocus) {
        if (layout.actions === 'menu' && !explore.open) exploreButton.focus({ preventScroll: true })
        else focused.focus({ preventScroll: true })
      }
    }
    playButton.disabled = !canBegin
    replayButton.disabled = !canBegin
    const showRetry = canRetry || retrying || retryExhausted
    if (!showRetry && document.activeElement === retryData) {
      const nextControl = canBegin ? playButton : scrub
      nextControl.focus({ preventScroll: true })
    }
    retryData.hidden = !showRetry
    // aria-disabled keeps keyboard focus on the control while an attempt is
    // running. The click guard above also covers native Enter/Space activation.
    retryData.setAttribute('aria-disabled', String(!canRetry || retrying))
    retryData.textContent = retrying ? 'Retrying data loading…'
      : retryExhausted && !canRetry ? 'Retry limit reached' : 'Retry data loading'
    // Avoid repeated live-region announcements on every animation frame.
    if (readinessStatus.textContent !== message) readinessStatus.textContent = message
    playButton.textContent = timeline.playing ? 'Pause sweep' : timeline.t === 0 ? 'Begin the sweep' : timeline.t >= T_END ? 'Watch again' : 'Continue sweep'
  }
  updateOpeningControls()

  const timelineTrack = el('div', 'timeline-track')
  const timelineLabels = el('div', 'timeline-labels')
  timelineLabels.append(el('span', '', 'Earth'), el('span', '', 'Nematode veil'))
  timelineTrack.append(timelineLabels, scrub)
  transportRow.append(playButton, timelineTrack, scrubVal, replayButton)
  controls.append(sweepGuide.element, transportRow, openingControls)

  // --- Cobb inversion slider (spec 6.3, CORRECTED — see field/cobb.ts and
  // docs/superpowers/results/2026-08-16-urban-claim-ruling.md) -----------
  //
  // Data end (0) applies no multiplier at all — dragging toward it doesn't
  // invert the cities, it makes them stop mattering, because the source
  // model cannot see urban land use. Cobb end (1) is his 1914 prediction,
  // given deliberately DIFFERENT visual treatment (italic, warm) so it can
  // never be mistaken for measurement.
  //
  // The LABELS changed, not that behaviour. `data (measured: no effect)` is
  // a finding written in the grammar of a disabled control, and it read as
  // one: the strongest idea in the piece looked like a dead slider. Both
  // ends now distinguish "Soil model" from "Cobb illustration", with the
  // verdict moved into the caption, where
  // it reads as a result. hud/copy.ts holds the strings; the test there
  // fails if on/off vocabulary comes back into either end label.
  const cobbControls = el('div', 'rail-row')
  const cobbCaption = el('div', 'rail-caption')

  const cobbDataLabel = el('span', 'rail-label', HUD_COPY.cobbAxisMeasured)
  const cobbSlider = document.createElement('input')
  cobbSlider.type = 'range'
  cobbSlider.min = '0'
  cobbSlider.max = '1'
  cobbSlider.step = '0.001'
  cobbSlider.value = '0'
  cobbSlider.setAttribute('aria-label', 'Compare soil model with an illustration of Cobb’s prediction')
  const cobbPredictionLabel = el('span', 'rail-label hud-accent', HUD_COPY.cobbAxisPredicted)

  /**
   * Why the axis may be inert, if it is.
   *
   * `synthetic` is the bug this replaces: `syntheticMaterial` is built with
   * no `urban` option at all, so `Shell.ts`'s `applyCobbMix` early-returns
   * and the slider is a COMPLETE no-op — while the HUD went on reporting a
   * blend percentage as though something were happening. `urban-loading` is
   * the same class of dishonesty from a different cause: the GHSL atlas is a
   * separate, later download, and until it lands there is nothing to enrich.
   */
  type CobbAvailability = 'live' | 'synthetic' | 'urban-loading'
  let urbanAtlasReady = false
  const cobbAvailability = (): CobbAvailability =>
    activeSource !== 'real' ? 'synthetic' : urbanAtlasReady ? 'live' : 'urban-loading'

  const updateCobbHud = (): void => {
    const availability = cobbAvailability()
    const live = availability === 'live'
    cobbControls.classList.toggle('is-disabled', !live)
    cobbSlider.disabled = !live
    if (!live) {
      // Deliberately NOT a blend percentage. The slider keeps its position
      // (so returning to real data restores where you left it) but the HUD
      // stops claiming that position is doing anything.
      cobbModeVal.textContent = 'unavailable'
      cobbCaption.textContent = availability === 'synthetic'
        ? viewerPolicy.syntheticCaption
        : HUD_COPY.cobbUnavailableUrbanLoading
      return
    }
    const mix = cobbMixUniform.value
    // Silent at both ends: the end LABELS already name them, and the thumb is
    // sitting on one. Repeating the same label beside it adds no information.
    // A blend is the only position neither the
    // labels nor the thumb express precisely, so that is the only one shown.
    cobbModeVal.textContent = mix < 0.02 || mix > 0.98 ? '' : `blend ${(mix * 100).toFixed(0)}%`
    cobbCaption.textContent = mix < 0.05
      ? HUD_COPY.cobbCaptionMeasured
      : mix > 0.95 ? HUD_COPY.cobbCaptionPrediction : ''
  }
  // Published to the two callers above that exist before this control does.
  updateCobbAvailability = updateCobbHud
  markUrbanAtlasReady = (): void => { urbanAtlasReady = true; updateCobbHud() }

  cobbSlider.addEventListener('input', () => {
    cobbMixUniform.value = Number(cobbSlider.value)
    updateCobbHud()
  })

  cobbControls.append(cobbDataLabel, cobbSlider, cobbPredictionLabel, cobbModeVal)

  // Secondary controls live in an accessible dialog: the globe and the sweep
  // stay visible on first visit, while every layer is reachable without keys.
  const explore = document.createElement('dialog')
  explore.id = 'explore-dialog'
  explore.setAttribute('aria-labelledby', 'explore-title')
  const exploreTitle = el('h2', '', 'Explore the data')
  exploreTitle.id = 'explore-title'
  const closeExplore = document.createElement('button')
  closeExplore.className = 'dialog-close'
  closeExplore.textContent = 'Close'
  closeExplore.addEventListener('click', () => explore.close())
  const dialogHeading = el('div', 'dialog-heading')
  dialogHeading.append(exploreTitle, closeExplore)
  const openExplore = (): void => {
    opening.explore()
    if (!explore.open) explore.showModal()
  }

  const layerControls = el('div', 'layer-controls')
  const makeLayer = (title: string, description: string, onChange: (on: boolean) => void): HTMLInputElement => {
    const label = document.createElement('label')
    label.className = 'layer-control'
    const input = document.createElement('input')
    input.type = 'checkbox'
    const text = el('span', 'layer-copy')
    text.append(el('span', 'layer-title', title), el('span', 'layer-description', description))
    input.addEventListener('change', () => onChange(input.checked))
    label.append(text, input)
    layerControls.append(label)
    return input
  }
  const syncLayers = (): void => {
    sampleToggle.checked = showSamples
    uncertaintyToggle.checked = showUncertainty
    updateLayerStatus()
  }
  const setSamplesVisible = (on: boolean): void => {
    showSamples = on
    if (samplePoints) samplePoints.points.visible = on
    syncLayers()
  }
  const setUncertaintyVisible = (on: boolean): void => {
    showUncertainty = on
    uncertaintyStrength.value = on ? 1 : 0
    syncLayers()
  }
  const sampleToggle = makeLayer('Sample locations', 'Show where researchers collected the 6,825 soil samples in the 2020 dataset.', setSamplesVisible)
  const uncertaintyToggle = makeLayer('Model uncertainty', 'Colour the veil by how much the soil estimates varied across repeated model runs.', setUncertaintyVisible)
  updateLayerStatus = (): void => {
    const layers = []
    if (showSamples) layers.push(samplePoints ? 'Sample locations' : sampleLoadFailed ? 'Sample locations unavailable' : 'Loading sample locations…')
    if (showUncertainty) layers.push(activeSource === 'real' ? 'Model uncertainty' : 'Uncertainty requires the soil model')
    layerStatus.textContent = layers.join(' · ')
    uncertaintyLine.style.display = showUncertainty && activeSource === 'real' ? '' : 'none'
  }
  const layerNote = el('p', 'dialog-note', 'The uncertainty colours are easiest to see at the end of the sweep.')
  const comparison = el('section', 'comparison-section')
  comparison.append(el('h3', '', 'Would cities leave a trace?'), finding, cobbControls, cobbCaption)
  const keyDetails = document.createElement('details')
  keyDetails.className = 'keyboard-details'
  keyDetails.append(el('summary', '', 'Keyboard shortcuts'), keyPanel)
  keyPanel.style.display = ''
  const dialogActions = el('div', 'dialog-actions')
  const creditsLink = document.createElement('a')
  creditsLink.href = '/about.html#credits'
  creditsLink.textContent = 'Sources & credits'
  dialogActions.append(creditsLink)
  const fullscreenStatus = el('p', 'fullscreen-status')
  fullscreenStatus.setAttribute('role', 'status')
  if (document.fullscreenEnabled) {
    const fullscreenButton = document.createElement('button')
    fullscreenButton.className = 'dialog-close'
    const syncFullscreen = (): void => {
      fullscreenButton.textContent = document.fullscreenElement ? 'Exit full screen' : 'Use full screen'
    }
    fullscreenButton.addEventListener('click', () => {
      const change = document.fullscreenElement
        ? document.exitFullscreen()
        : document.documentElement.requestFullscreen()
      change.then(() => { fullscreenStatus.textContent = ''; explore.close() })
        .catch(() => { fullscreenStatus.textContent = 'The browser could not switch full-screen mode.' })
    })
    document.addEventListener('fullscreenchange', syncFullscreen)
    syncFullscreen()
    dialogActions.append(fullscreenButton)
  }
  explore.append(dialogHeading, openingMenuActions, layerControls, layerNote, uncertaintyLine, comparison, keyDetails, dialogActions, fullscreenStatus, diagnostics)
  explore.addEventListener('click', (event) => {
    if (event.target !== explore) return
    const rect = explore.getBoundingClientRect()
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) explore.close()
  })

  const navigation = el('nav', 'viewer-nav')
  navigation.setAttribute('aria-label', 'Explore Nematode Earth')
  const methodsLink = about.cloneNode(true) as HTMLAnchorElement
  methodsLink.href = ABOUT_LINK_HREF
  methodsLink.textContent = 'Science & methods'
  const exploreButton = document.createElement('button')
  exploreButton.textContent = 'Explore the data'
  exploreButton.setAttribute('aria-haspopup', 'dialog')
  exploreButton.addEventListener('click', openExplore)
  navigation.append(methodsLink, exploreButton)
  topbar.append(el('div', 'viewer-wordmark', HUD_COPY.wordmark), navigation)
  const gestureHint = ALWAYS_VISIBLE_HINTS.slice(0, 2).join(' · ')
  const hints = el('div', 'viewer-gestures', gestureHint)
  hints.setAttribute('role', 'status')
  hints.setAttribute('aria-live', 'polite')
  hints.setAttribute('aria-atomic', 'true')
  let showingQualityHint = false
  const sceneSpace = el('div', 'viewer-scene-space')
  sceneSpace.setAttribute('aria-hidden', 'true')
  document.body.append(topbar, controls, hints, explore, sceneSpace)
  updateCobbHud()

  // --- keeping the two blocks out of each other ------------------------
  //
  // The bottom HUD is bottom-anchored and the top rail is top-anchored, and
  // nothing previously stopped them meeting in the middle: at a 336px-tall
  // viewport the HUD stood 263px tall and drew through the rail for 64px of
  // overlap. Both heights are content-dependent — the rail's caption wraps,
  // the HUD's finding wraps — so neither can be a constant in the stylesheet.
  // Measured instead, and published as custom properties: `--topbar-height`
  // caps the HUD's max-height, `--hud-height` sizes the scrim behind it.
  //
  // The observer cannot run away: --topbar-height feeds the HUD's max-height
  // and --hud-height feeds only #hud-scrim, which is not observed, so a
  // topbar change settles after one extra pass rather than oscillating.
  /**
   * Right edge of the widest line of ACTUAL TEXT in the HUD, in px from the
   * viewport's left edge.
   *
   * Not `hud.getBoundingClientRect().width`: every child is a block that
   * spans the full HUD box regardless of how short its text is, so the
   * element boxes all report the same full width and say nothing about where
   * the glyphs stop. A Range over each block's contents measures the inline
   * content instead, which is what the scrim actually needs to cover — the
   * first version of that scrim used the viewport width and spent most of its
   * area dimming empty sky.
   */
  const measureTextRight = (): number => {
    const range = document.createRange()
    let right = 0
    const walk = (node: Element): void => {
      for (const child of node.children) {
        if (!(child instanceof HTMLElement) || child.offsetParent === null) continue
        if (child.childElementCount > 0 && getComputedStyle(child).display.includes('block')) {
          walk(child)
          continue
        }
        range.selectNodeContents(child)
        const r = range.getBoundingClientRect()
        if (r.width > 0 && r.right > right) right = r.right
      }
    }
    walk(hud)
    range.detach()
    return right
  }

  const syncLayoutMetrics = (): void => {
    const root = document.documentElement.style
    root.setProperty('--topbar-height', `${Math.ceil(topbar.getBoundingClientRect().height)}px`)
    root.setProperty('--transport-height', `${Math.ceil(controls.getBoundingClientRect().height)}px`)
    root.setProperty('--hud-height', `${Math.ceil(hud.getBoundingClientRect().height)}px`)
    root.setProperty('--hud-right', `${Math.ceil(hud.getBoundingClientRect().right)}px`)
    root.setProperty('--hud-top', `${Math.floor(hud.getBoundingClientRect().top)}px`)
    root.setProperty('--hud-text-width', `${Math.ceil(measureTextRight())}px`)
    // Fades the bottom edge only while the max-height above is actually
    // cutting content off. A mask changes no layout, so setting it from
    // inside the observer cannot feed back into another resize.
    //
    // NOT `scrollHeight > clientHeight`, which was the first version of this
    // line and could not work: overflow out of a flex container is not
    // reachable by scrolling, so scrollHeight stays pinned to clientHeight
    // and the test reads false whether or not anything is being cut off —
    // an instrument that returns the same answer either way. Measured off
    // the children's own boxes instead, which genuinely differ.
    const box = hud.getBoundingClientRect()
    let contentBottom = box.top
    for (const child of hud.children) {
      const r = child.getBoundingClientRect()
      if (r.height > 0 && r.bottom > contentBottom) contentBottom = r.bottom
    }
    hud.classList.toggle('is-clipped', contentBottom > box.bottom + 0.5)
    if (frameCameraToRegion(camera, window.innerWidth, window.innerHeight, sceneSpace.getBoundingClientRect())) {
      projectionViewport = { width: window.innerWidth, height: window.innerHeight }
      syncZoomQuality()
    }
  }
  const layoutObserver = new ResizeObserver(syncLayoutMetrics)
  layoutObserver.observe(topbar)
  layoutObserver.observe(controls)
  layoutObserver.observe(hud)
  layoutObserver.observe(sceneSpace)
  addEventListener('resize', syncLayoutMetrics)
  syncLayoutMetrics()

  // Exposure lifts the faint veil for display. It leaves the optical depth
  // and the two shell segments' transmittance calculations unchanged.
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  let exposure = 8
  const applyExposure = (): void => {
    renderer.toneMappingExposure = exposure
    expLine.replaceChildren(
      faint('Exposure '), val(`${exposure.toFixed(0)}×`), faint(' relative to the unamplified render'),
    )
  }
  const REGIME_MODES: readonly RegimeMode[] = ['blended', 'shell', 'particles']
  let showDiagnostics = false
  let showKeys = false

  // Latest per-frame windowIn/windowOut (set in the render loop below,
  // read by logParticleBuffers) - plain CPU numbers already, no GPU
  // readback needed for these two.
  let lastWindowIn = 0
  let lastWindowOut = 0
  // Tracks the last sigma_pix actually applied via splat.setSigmaPix, so
  // the render loop only rebuilds the resolve materials when the rounded
  // schedule value changes - matches buildSplatPipeline's initial value.
  let lastSigmaPix = 3

  interface TargetStats {
    peak: number
    /** Mean over NON-ZERO texels only - a mean over the whole region would be dragged toward 0 by empty sky/space and say nothing about how bright the particles themselves are. */
    mean: number
    /** Non-zero texel count as a fraction of the sampled region - low peak + low fraction means under-gained; high peak + low fraction means concentrated-but-tiny splats (sigma_pix too small), not a gain problem. */
    nonZeroFrac: number
    /**
     * TOTAL over the sampled region, and the count it came from. For the
     * single-instance deposit trace these two are the load-bearing numbers,
     * not `peak`: the deposit fragment writes bilinear weights, and bilinear
     * weights PARTITION UNITY, so `sum` must equal `flux * compFactor`
     * exactly and `nonZeroCount` must be 4 (or fewer only where the splat
     * straddles the sampled region's edge). That single invariant separates
     * a coverage bug (count < 4, sum short by the missing weights) from a
     * magnitude bug (count == 4, sum uniformly scaled) from an interpolation
     * bug (count == 4, sum right, individual weights wrong) - which `peak`
     * alone cannot distinguish between at all.
     */
    sum: number
    nonZeroCount: number
    /**
     * Absolute (whole-target, not region-relative) pixel coordinates of the
     * peak texel, or -1/-1 if every texel in the region was zero. Team
     * lead's ask: comparing this against the CPU mirror's predicted
     * `basePixelX/Y` is what tells "deposit is elsewhere" apart from
     * "prediction is elsewhere" - an all-zero region alone is ambiguous
     * between the two.
     */
    argmaxX: number
    argmaxY: number
  }

  /**
   * Reads back a `size x size` region centred on `target`, decodes it (16Float
   * -> half-bits Uint16Array -> real floats, same `DataUtils.fromHalfFloat`
   * pattern `loadCube.ts`/`bakeCube.ts` already use), and reduces to
   * peak/mean/non-zero-fraction OF CHANNEL 0 (R) of each texel.
   * `readRenderTargetPixelsAsync` pads each row to a 256-byte GPU alignment
   * boundary (`WebGPUTextureUtils.js`'s `copyTextureToBuffer`) — NOT a flat
   * `size*size` array — so the row stride is computed explicitly rather than
   * assumed contiguous. `texelChannels` is 1 for the pipeline's RedFormat
   * targets (deposit/blurH/resolve) or 4 for an RGBA16Float target (the
   * probe composite target below) — same decode, just a different stride
   * between texels and only R examined either way.
   */
  /**
   * Shared low-level readback: fetches the raw half-float bits for a
   * `w x h` region and decodes channel 0 of every texel into a
   * `[localY][localX]` grid, where `localY=0` is ABSOLUTE row `y0` and
   * `localY` increases with increasing absolute row - i.e. the grid a
   * CALLER gets back is already in the coordinate convention the rest of
   * this shader/pipeline uses (`basePixel.y`, `pixelUnits.y`, etc).
   *
   * `readRenderTargetPixelsAsync`'s OWN row order is inverted relative to
   * that convention - confirmed live, not assumed: team lead's deposit
   * coverage test pinned it exactly via two UNIQUE weight values
   * (`w01=0.5625` only possible at localCorner y=1, `w10=0.0625` only
   * possible at y=0) landing on the OPPOSITE absolute rows from what an
   * un-flipped reading predicted. `(h-1-nativeRow)` is the fix that
   * reproduces exactly that swap - see the message this responds to for
   * the full derivation. Every caller goes through this one function so
   * the fix cannot be applied to one readback and missed on another (this
   * file previously had two independent, duplicated decode loops - the
   * flip was fixed in neither until now).
   */
  async function readRegionDecoded(
    target: THREE.RenderTarget, x0: number, y0: number, w: number, h: number, texelChannels = 1,
  ): Promise<number[][]> {
    const cx0 = Math.max(0, Math.min(x0, target.width - w))
    const cy0 = Math.max(0, Math.min(y0, target.height - h))
    // The row flip below reverses the ORDER of rows within the region, but
    // the region's own ORIGIN still has to be converted: `y0` arrives in
    // top-down space and `readRenderTargetPixelsAsync` addresses bottom-up,
    // so a region occupying top-down rows [y0, y0+h) begins at native row
    // `target.height - (y0 + h)`. Flipping inside the region while reading
    // it from the un-converted origin leaves a constant offset of
    // `target.height - 2*y0 - h`, which is 0 whenever the region is exactly
    // centred and the target's height is even — and 1 when it is odd. That
    // is exactly the residual one-row error the coverage test showed after
    // the flip itself was fixed: 805 - (338 + 128) = 339, not 338.
    const nativeY0 = target.height - (cy0 + h)
    const raw = await renderer.readRenderTargetPixelsAsync(target, cx0, nativeY0, w, h) as Uint16Array
    const bytesPerTexel = 2 * texelChannels
    const rowStrideElems = Math.ceil((w * bytesPerTexel) / 256) * (256 / 2)
    const grid: number[][] = Array.from({ length: h }, () => new Array<number>(w).fill(NaN))
    for (let nativeRow = 0; nativeRow < h; nativeRow++) {
      const localY = h - 1 - nativeRow // the row-order flip
      for (let x = 0; x < w; x++) {
        const bits = raw[nativeRow * rowStrideElems + x * texelChannels]
        if (bits !== undefined) {
          const row = grid[localY]
          if (row) row[x] = THREE.DataUtils.fromHalfFloat(bits)
        }
      }
    }
    return grid
  }

  async function readTargetStats(target: THREE.RenderTarget, size = 128, texelChannels = 1): Promise<TargetStats> {
    const w = Math.min(size, target.width)
    const h = Math.min(size, target.height)
    const x0 = Math.max(0, Math.floor((target.width - w) / 2))
    const y0 = Math.max(0, Math.floor((target.height - h) / 2))
    const grid = await readRegionDecoded(target, x0, y0, w, h, texelChannels)

    let peak = 0
    let sum = 0
    let nonZero = 0
    let argmaxX = -1
    let argmaxY = -1
    for (let localY = 0; localY < h; localY++) {
      const row = grid[localY] ?? []
      for (let x = 0; x < w; x++) {
        const v = row[x]
        if (v === undefined || Number.isNaN(v)) continue
        if (v > peak) { peak = v; argmaxX = x0 + x; argmaxY = y0 + localY }
        if (v !== 0) { sum += v; nonZero++ }
      }
    }
    const total = w * h
    return {
      peak,
      mean: nonZero > 0 ? sum / nonZero : 0,
      nonZeroFrac: total > 0 ? nonZero / total : 0,
      sum,
      nonZeroCount: nonZero,
      argmaxX,
      argmaxY,
    }
  }

  /**
   * Same readback/decode as `readTargetStats` but returns the full grid
   * instead of reducing it - for printing individual pixel values against
   * their theoretical bilinear weight (team lead's "report the four pixel
   * values" ask), not just peak/sum/count. Row-order-corrected the same
   * way `readTargetStats` is - see `readRegionDecoded`.
   */
  async function readRegionGrid(
    target: THREE.RenderTarget, x0: number, y0: number, w: number, h: number, texelChannels = 1,
  ): Promise<number[][]> {
    return readRegionDecoded(target, x0, y0, w, h, texelChannels)
  }

  /**
   * CPU mirror of `buildDepositMaterial`'s vertex-stage pixel/corner-weight
   * math (SplatBlur.ts), used ONLY to know WHICH pixel to read back and
   * what the THEORETICAL weight at each corner should be for a given
   * world position - never treated as the measurement itself.
   * `worldPos.clone().project(camera)` does exactly
   * `applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix)`
   * with the perspective divide included, i.e. the same
   * `cameraProjectionMatrix.mul(cameraViewMatrix.mul(vec4(instPos,1)))` the
   * shader computes, just evaluated on the CPU instead of the GPU - camera
   * sits pinned at the render origin under floating origin, so this needs
   * no extra translation handling to match.
   */
  function computeExpectedWeights(worldPos: THREE.Vector3, vw: number, vh: number): {
    basePixelX: number
    basePixelY: number
    fracX: number
    fracY: number
    /** [w(0,0), w(1,0), w(0,1), w(1,1)] - same corner order as Particles.ts's buildQuadGeometry / positionLocal.xy. */
    weights: [number, number, number, number]
  } {
    const ndc = worldPos.clone().project(camera)
    const pixelX = (ndc.x * 0.5 + 0.5) * vw
    const pixelY = (ndc.y * 0.5 + 0.5) * vh

    // Snap-then-floor, not a bare floor: `pixelX/Y - 0.5` is meant to land
    // EXACTLY on an integer whenever solveWorldPosForFrac targets a frac
    // of 0 (or, symmetrically, 1) on an axis - float64 accumulated through
    // project()'s matrix chain does not reliably land exactly there (e.g.
    // 401.999999999997 instead of 402.0), and a bare floor() sends that
    // one row low. Team lead's own deposit trace caught this precisely:
    // "predicted pixel=(755,401)" against a measured, theory-matching
    // deposit at (755,402) - a diagnostic bug, not a pipeline one (see
    // that message). EPSILON is far smaller than any genuine fractional
    // position (1e-6 px), so this cannot mask a real non-boundary frac.
    const EPSILON = 1e-6
    const rawX = pixelX - 0.5
    const rawY = pixelY - 0.5
    const basePixelX = Math.floor(rawX + EPSILON)
    const basePixelY = Math.floor(rawY + EPSILON)
    // Clamped into [0,1) explicitly (team lead's ask), not just assumed -
    // the snap above handles the exact-boundary case, this is the safety
    // net for whatever residual float noise it doesn't fully absorb.
    const clamp01Open = (v: number): number => (v < 0 ? 0 : v >= 1 ? 1 - Number.EPSILON : v)
    const fracX = clamp01Open(rawX - basePixelX)
    const fracY = clamp01Open(rawY - basePixelY)
    const w00 = (1 - fracX) * (1 - fracY)
    const w10 = fracX * (1 - fracY)
    const w01 = (1 - fracX) * fracY
    const w11 = fracX * fracY
    return { basePixelX, basePixelY, fracX, fracY, weights: [w00, w10, w01, w11] }
  }

  // Human-readable names for the handful of THREE.* blend constants this
  // trace prints - THREE.CustomBlending/OneFactor/AddEquation etc. are
  // plain numbers (5, 201, 100, ...), unreadable in a console line without
  // this. Values confirmed directly from the installed three.js package,
  // not guessed.
  const BLEND_CONST_NAMES: Record<number, string> = {
    [THREE.NoBlending]: 'NoBlending', [THREE.NormalBlending]: 'NormalBlending',
    [THREE.AdditiveBlending]: 'AdditiveBlending', [THREE.SubtractiveBlending]: 'SubtractiveBlending',
    [THREE.MultiplyBlending]: 'MultiplyBlending', [THREE.CustomBlending]: 'CustomBlending',
    [THREE.ZeroFactor]: 'ZeroFactor', [THREE.OneFactor]: 'OneFactor',
    [THREE.SrcAlphaFactor]: 'SrcAlphaFactor', [THREE.OneMinusSrcAlphaFactor]: 'OneMinusSrcAlphaFactor',
    [THREE.AddEquation]: 'AddEquation',
  }
  const blendName = (n: number): string => BLEND_CONST_NAMES[n] ?? String(n)

  /**
   * Traces ONE known-flux, known-position instance through deposit -> resolve
   * -> composite, printing the MEASURED value at each stage - team lead's
   * explicit instruction after three separate defects this session turned
   * out to be silent zeroing/scaling in exactly this class of layer (the
   * urban cube's density->tau re-encode, a tau/density unit mix-up behind
   * min(tau,tauCap), renderAsync resolving on submission not completion) -
   * none of which algebra alone would have caught; all three were found by
   * measuring adjacent stages. This does not derive an expected value and
   * compare - it prints what is actually there at each stage and leaves the
   * comparison to whoever reads the numbers.
   *
   * Bypasses particle emission entirely: writes ONE record directly with a
   * chosen flux/position, so the CODE under test (`buildDepositMaterial`/
   * `buildResolveMaterial`/`buildCompositeMaterial`, all imported unmodified
   * - see `DepositSource` in SplatBlur.ts) is real, only the INPUT is
   * synthetic. Builds and disposes its own full-screen-sized targets each
   * call - cheap, infrequent, and avoids resize-tracking for a manually
   * triggered diagnostic.
   */
  async function logSingleInstanceTrace(): Promise<void> {
    const TEST_FLUX = 1.0
    const TEST_DIST = 2000 // m, camera-relative - inside the windowIn..windowOut band

    // Known render-space (camera-relative) position: straight ahead of the
    // camera, so it lands near screen centre regardless of orientation -
    // matches what records.xyz means (Particles.ts's own doc for that
    // field), NOT a camera-local/view-space offset.
    const forward = new THREE.Vector3()
    camera.getWorldDirection(forward)
    /**
     * OFF-AXIS on purpose. Straight-ahead was the obvious choice and it is
     * the wrong one: `forward * TEST_DIST` projects to exactly NDC (0,0),
     * so it lands at screen centre both when the projection works AND when
     * the projection is collapsing everything to the centre. It cannot tell
     * those apart, which matters right now because the real particle system
     * has ~100,000 records spread over hundreds of metres in every axis
     * (measured) landing on six pixels at the screen centre (also
     * measured).
     *
     * Offsetting along the camera's right vector gives the trace a known
     * answer that is NOT the centre, so a single instance reproduces — or
     * clears — the collapse with no dependence on the particle system at
     * all. `TEST_RIGHT` is a fraction of the half-width at `TEST_DIST`, so
     * it stays comfortably on screen at any aspect ratio.
     */
    const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize()
    const halfHeight = TEST_DIST * Math.tan((camera.fov * Math.PI) / 360)
    const TEST_RIGHT = halfHeight * camera.aspect * 0.4
    const probePos = forward.clone().multiplyScalar(TEST_DIST).addScaledVector(right, TEST_RIGHT)

    const probeRecordsAttr = new THREE.StorageBufferAttribute(
      new Float32Array([probePos.x, probePos.y, probePos.z, TEST_FLUX]), 4,
    )
    const probeRecords = storage(probeRecordsAttr, 'vec4', 1)
    const probeCompFactorAttr = new THREE.StorageBufferAttribute(new Float32Array([1]), 1)

    // Deposit/blurH/resolve match the REAL splat pipeline's own sizing
    // (viewer.ts's buildSplatPipeline(..., window.innerWidth,
    // window.innerHeight, ...) - CSS pixels, confirmed by grep, team
    // lead's dpr concern does not apply to these three). The composite
    // target does NOT: the real composite pass renders to the canvas at
    // its DEVICE-pixel drawing buffer size (renderer.setPixelRatio in
    // Renderer.ts), not window.innerWidth/innerHeight - matching that
    // here is what lets this trace actually measure whether energy
    // survives the CSS-pixel-target -> device-pixel-canvas upsample team
    // lead flagged as a live, unmeasured suspect (a real, separate
    // question from the coverage bug - see the printed sizes/sums below).
    const w = window.innerWidth
    const h = window.innerHeight
    const drawingBufferSize = new THREE.Vector2()
    renderer.getDrawingBufferSize(drawingBufferSize)
    const canvasW = Math.round(drawingBufferSize.x)
    const canvasH = Math.round(drawingBufferSize.y)

    const makeProbeTarget = (format: THREE.PixelFormat, tw: number, th: number): THREE.RenderTarget => new THREE.RenderTarget(tw, th, {
      format, type: THREE.HalfFloatType, magFilter: THREE.LinearFilter, minFilter: THREE.LinearFilter,
      generateMipmaps: false, depthBuffer: false, stencilBuffer: false,
    })
    const probeDepositTarget = makeProbeTarget(THREE.RedFormat, w, h)
    const probeBlurHTarget = makeProbeTarget(THREE.RedFormat, w, h)
    const probeResolveTarget = makeProbeTarget(THREE.RedFormat, w, h)
    const probeCompositeTarget = makeProbeTarget(THREE.RGBAFormat, canvasW, canvasH)

    // Same 4-vertex/6-index quad shape as Particles.ts's (private)
    // buildQuadGeometry, but a PLAIN draw - no indirect wiring needed:
    // RenderObject.getDrawParameters() gives a plain BufferGeometry
    // instanceCount=1 (the same fact Particles.ts's own module doc cites
    // for the real geometry), so a single instance draws correctly here
    // with no indirect buffer at all.
    const probeGeometry = new THREE.BufferGeometry()
    probeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(
      [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], 3,
    ))
    probeGeometry.setIndex(new THREE.Uint16BufferAttribute([0, 1, 2, 0, 2, 3], 1))

    const probeDepositMaterial = buildDepositMaterial({ records: probeRecords, compFactorAttr: probeCompFactorAttr })
    const probeDepositMesh = new THREE.Mesh(probeGeometry, probeDepositMaterial)
    probeDepositMesh.frustumCulled = false
    const probeDepositScene = new THREE.Scene()
    probeDepositScene.add(probeDepositMesh)

    // Shared stateless orthographic camera - see SplatBlur.ts's
    // makeFullscreenCamera doc for why every full-screen pass reuses one.
    const fsCamera = splat.resolveCamera
    const probeBlurHMaterial = buildResolveMaterial(probeDepositTarget.texture, lastSigmaPix, 'horizontal')
    const probeBlurHMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), probeBlurHMaterial)
    const probeBlurHScene = new THREE.Scene()
    probeBlurHScene.add(probeBlurHMesh)

    const probeResolveMaterial = buildResolveMaterial(probeBlurHTarget.texture, lastSigmaPix, 'vertical')
    const probeResolveMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), probeResolveMaterial)
    const probeResolveScene = new THREE.Scene()
    probeResolveScene.add(probeResolveMesh)

    const probeCompositeMaterial = buildCompositeMaterial(probeResolveTarget.texture)
    const probeCompositeMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), probeCompositeMaterial)
    const probeCompositeScene = new THREE.Scene()
    probeCompositeScene.add(probeCompositeMesh)

    renderer.setRenderTarget(probeDepositTarget)
    renderer.setClearColor(0x000000, 0)
    renderer.clear()
    renderer.render(probeDepositScene, camera)

    renderer.setRenderTarget(probeBlurHTarget)
    renderer.clear()
    renderer.render(probeBlurHScene, fsCamera)

    renderer.setRenderTarget(probeResolveTarget)
    renderer.clear()
    renderer.render(probeResolveScene, fsCamera)

    renderer.setRenderTarget(probeCompositeTarget)
    renderer.clear()
    renderer.render(probeCompositeScene, fsCamera)

    renderer.setRenderTarget(null)

    const [depositStats, resolveStats, compositeStats] = await Promise.all([
      // FULL-target, not a 128x128 centre crop. The probe is deliberately
      // off-axis now (see probePos above), so it lands hundreds of pixels
      // from the centre and a centre crop would report "nothing found"
      // whether the projection worked or not - turning the one test that
      // distinguishes those two cases back into a test that cannot.
      readTargetStats(probeDepositTarget, Math.max(probeDepositTarget.width, probeDepositTarget.height)),
      readTargetStats(probeResolveTarget, Math.max(probeResolveTarget.width, probeResolveTarget.height)),
      readTargetStats(probeCompositeTarget, Math.max(probeCompositeTarget.width, probeCompositeTarget.height), 4),
    ])

    // Cheap, often-silent checks team lead named explicitly: the ACTUAL
    // live blend state of the material used in this trace (not what the
    // source code is assumed to say), and the ACTUAL tap-weight sum for
    // the CURRENTLY-APPLIED sigma_pix (not assumed to be 1 just because a
    // unit test elsewhere says gaussianTaps normalizes - measuring it here
    // catches a per-pass-vs-overall normalization mismatch, exactly the
    // "two orders of magnitude sink" team lead flagged as plausible).
    const taps = gaussianTaps(lastSigmaPix)
    const tapSum = taps.reduce((a, b) => a + b, 0)
    const m = probeDepositMaterial

    // Predicted vs found location - team lead's ask: an all-zero (or
    // wrong-looking) region alone cannot distinguish "deposit is
    // elsewhere" from "the CPU mirror's prediction is elsewhere". Printing
    // both closes that gap directly. `probePos` is the off-axis position
    // above (forward*TEST_DIST + right*TEST_RIGHT), not screen centre.
    const predicted = computeExpectedWeights(probePos, w, h)

    // Sign-check for the clip.w instrument in logParticleBuffers, against a
    // case measured to be correct: THIS probe renders at its predicted pixel
    // with energy conserved, so it is unambiguously in front of the camera.
    // If the same element math reports w <= 0 for it, the instrument's sign
    // is inverted and its "100% of particles are behind the camera" reading
    // is an artefact rather than a finding. Validating an instrument on a
    // known-good input is the step this whole investigation kept skipping.
    const pv = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
    const pe = pv.elements
    const probeClipW = (pe[3] ?? 0) * probePos.x + (pe[7] ?? 0) * probePos.y
      + (pe[11] ?? 0) * probePos.z + (pe[15] ?? 0)

    // Team lead's hypothesis 1, measured rather than assumed: `records.xyz`
    // is documented as camera-relative, and `buildDepositMaterial`'s vertex
    // does `cameraViewMatrix.mul(vec4(instPos,1))` - if `cameraViewMatrix`
    // (== camera.matrixWorldInverse, the uniform TSL's cameraViewMatrix
    // reads) carries any translation, every camera-relative position gets
    // displaced by it, which would push far-off-axis positions toward
    // wherever that translation points rather than collapsing them to the
    // centre specifically - still worth ruling out with a real number.
    // viewer.ts pins `camera.position.set(0,0,0)` every frame (see the doc
    // at the top of this file), so IF that write and this manually-
    // triggered trace are on the same cadence, matrixWorldInverse should be
    // rotation-only. Read it AFTER the renders above so this is the exact
    // matrix that produced them, not a re-derivation of what it should be.
    const viewTranslation = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorldInverse)
    const worldTranslation = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld)

    // eslint-disable-next-line no-console
    console.log(
      `[trace] known input: flux=${TEST_FLUX} at dist=${TEST_DIST}m `
      + `predicted pixel=(${predicted.basePixelX},${predicted.basePixelY})..+1 frac=(${predicted.fracX.toFixed(3)},${predicted.fracY.toFixed(3)}) `
      // Team lead's live, unmeasured suspect: deposit/resolve run at
      // window.innerWidth/innerHeight (CSS px), the real composite runs at
      // the canvas's device-pixel drawing buffer size - printed here, not
      // assumed, so the scale factor is visible directly. Whether energy
      // survives that upsample is a SEPARATE question from the coverage
      // bug (deposit sum vs composite sum, below) - do not conflate them.
      + `| deposit/resolve target=${w}x${h} (CSS px) composite target=${canvasW}x${canvasH} (device px, dpr~${(canvasW / w).toFixed(2)}) `
      + `| camera.position=(${camera.position.x.toFixed(3)},${camera.position.y.toFixed(3)},${camera.position.z.toFixed(3)}) `
      + `camera.matrixWorld translation=(${worldTranslation.x.toFixed(3)},${worldTranslation.y.toFixed(3)},${worldTranslation.z.toFixed(3)}) `
      + `cameraViewMatrix translation=(${viewTranslation.x.toFixed(3)},${viewTranslation.y.toFixed(3)},${viewTranslation.z.toFixed(3)}) `
      + `| SIGN CHECK probe clip.w=${probeClipW.toFixed(3)} (must be > 0: this probe demonstrably renders) `
      + `| MEASURED deposit peak=${depositStats.peak} at (${depositStats.argmaxX},${depositStats.argmaxY}) `
      // The partition-of-unity check, which is the whole point: bilinear
      // weights sum to 1, so sum MUST equal flux*compFactor and count MUST
      // be 4. Anything else names which stage is wrong (see TargetStats.sum).
      + `sum=${depositStats.sum} nonZeroCount=${depositStats.nonZeroCount} `
      + `sum/flux=${depositStats.sum / TEST_FLUX} `
      + `| MEASURED resolve peak=${resolveStats.peak} at (${resolveStats.argmaxX},${resolveStats.argmaxY}) sum=${resolveStats.sum} `
      + `| MEASURED composite(screen) peak=${compositeStats.peak} at (${compositeStats.argmaxX},${compositeStats.argmaxY}) sum=${compositeStats.sum} `
      + `| live blend state: blending=${blendName(m.blending)} blendSrc=${blendName(m.blendSrc)} `
      + `blendDst=${blendName(m.blendDst)} blendEquation=${blendName(m.blendEquation)} `
      + `deposit target format=RedFormat `
      + `| sigma_pix=${lastSigmaPix} tap count=${taps.length} tap sum (should be 1.0 if unit-area)=${tapSum}`,
    )

    probeDepositTarget.dispose()
    probeBlurHTarget.dispose()
    probeResolveTarget.dispose()
    probeCompositeTarget.dispose()
    probeDepositMaterial.dispose()
    probeBlurHMaterial.dispose()
    probeResolveMaterial.dispose()
    probeCompositeMaterial.dispose()
    probeGeometry.dispose()
  }

  /**
   * ONE Newton correction from `startPos` toward the render-space position
   * that projects to the EXACT target frac, against the camera's CURRENT
   * (assumed frozen for the duration of the caller's test) state. The local
   * mapping from a small world-space offset to pixel position is smooth
   * and very nearly affine at a few-pixel range and a few-thousand-metre
   * distance, so one finite-difference-Jacobian step lands within a tiny
   * fraction of a pixel in practice - `right`/`up` need not be exactly
   * orthogonal, just linearly independent, since the Jacobian is inverted
   * numerically rather than assumed diagonal.
   */
  function solveWorldPosForFrac(
    startPos: THREE.Vector3, targetFracX: number, targetFracY: number,
    right: THREE.Vector3, up: THREE.Vector3, vw: number, vh: number,
  ): THREE.Vector3 {
    // Shortest signed distance mod 1 - handles the case where the straight
    // frac difference would cross a pixel boundary in the wrong direction.
    const wrap = (a: number, b: number): number => { const d = a - b; return d - Math.round(d) }

    const stepM = 1
    const p0 = computeExpectedWeights(startPos, vw, vh)
    const pr = computeExpectedWeights(startPos.clone().addScaledVector(right, stepM), vw, vh)
    const pu = computeExpectedWeights(startPos.clone().addScaledVector(up, stepM), vw, vh)

    // Continuous pixel coordinate (basePixel+frac, not frac alone) so a
    // finite-difference sample landing in the next pixel over doesn't look
    // like a wraparound in the derivative itself.
    const f0x = p0.basePixelX + p0.fracX
    const f0y = p0.basePixelY + p0.fracY
    const jRX = (pr.basePixelX + pr.fracX - f0x) / stepM
    const jRY = (pr.basePixelY + pr.fracY - f0y) / stepM
    const jUX = (pu.basePixelX + pu.fracX - f0x) / stepM
    const jUY = (pu.basePixelY + pu.fracY - f0y) / stepM

    const dx = wrap(targetFracX, p0.fracX)
    const dy = wrap(targetFracY, p0.fracY)

    const det = jRX * jUY - jRY * jUX
    if (Math.abs(det) < 1e-12) return startPos.clone() // degenerate basis - bail to the starting guess rather than divide by ~0
    const dRight = (dx * jUY - dy * jUX) / det
    const dUp = (jRX * dy - jRY * dx) / det

    return startPos.clone().addScaledVector(right, dRight).addScaledVector(up, dUp)
  }

  /**
   * The discriminator team lead asked for, corrected per their own
   * follow-up: the first version derived the probe's screen position from
   * the LIVE (auto-rotating) camera, so seven repeat runs at "the same"
   * input landed at seven different sub-pixel offsets by luck - exactly
   * what produced the non-deterministic 0-or-1 `nonZeroCount`. This
   * version (1) freezes the camera for its duration, (2) SOLVES for
   * specific, named target fracs (0.5,0.5 and 0.25,0.75, team lead's own
   * cases) via `solveWorldPosForFrac` rather than hoping a sweep crosses
   * them, and (3) runs BOTH the SHIPPED deposit material and team lead's
   * hypothesised fix (vertices at pixel corners, not centres -
   * `buildDepositMaterialCornerVariantForTesting`, SplatBlur.ts, not used
   * by the real pipeline) at each target, side by side. If the corner
   * variant reliably gives nonZeroCount=4 and sum=1.0 where the shipped
   * one does not, that confirms the hypothesis and names the one-line fix;
   * if not, the varying-interpolation explanation is next. DEPOSIT ONLY -
   * no blur/composite, no precision question here, see team lead's own
   * "do not fix precision yet" note.
   *
   * Does NOT freeze the camera itself - the 't' key handler freezes it
   * once around BOTH `logSingleInstanceTrace` and this call, so neither
   * function's numbers drift relative to the other and freezing/restoring
   * doesn't happen twice in quick succession for one keypress.
   */
  async function logDepositCoverageTest(): Promise<void> {
    {
      const TEST_FLUX = 1.0
      const TEST_DIST = 2000

      const forward = new THREE.Vector3()
      camera.getWorldDirection(forward)
      const right = forward.clone().cross(camera.up).normalize()
      const up = right.clone().cross(forward).normalize()
      const basePos = forward.clone().multiplyScalar(TEST_DIST)

      const w = window.innerWidth
      const h = window.innerHeight

      const probeRecordsAttr = new THREE.StorageBufferAttribute(new Float32Array(4), 4)
      const probeRecords = storage(probeRecordsAttr, 'vec4', 1)
      const probeCompFactorAttr = new THREE.StorageBufferAttribute(new Float32Array([1]), 1)

      const probeGeometry = new THREE.BufferGeometry()
      probeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], 3,
      ))
      probeGeometry.setIndex(new THREE.Uint16BufferAttribute([0, 1, 2, 0, 2, 3], 1))

      const variants: { name: string, material: THREE.NodeMaterial }[] = [
        { name: 'shipped(centres)', material: buildDepositMaterial({ records: probeRecords, compFactorAttr: probeCompFactorAttr }) },
        { name: 'test(corners)', material: buildDepositMaterialCornerVariantForTesting({ records: probeRecords, compFactorAttr: probeCompFactorAttr }) },
      ]

      const target = new THREE.RenderTarget(w, h, {
        format: THREE.RedFormat, type: THREE.HalfFloatType,
        magFilter: THREE.LinearFilter, minFilter: THREE.LinearFilter,
        generateMipmaps: false, depthBuffer: false, stencilBuffer: false,
      })

      const targetFracs: [number, number][] = [[0.5, 0.5], [0.25, 0.75]]
      const GRID = 8 // generous margin either side of the theoretical 2x2 block

      const lines: string[] = []
      for (const [fx, fy] of targetFracs) {
        // Solve once from the SAME starting point for both variants - the
        // two materials differ only in cornerPixel's constant offset, not
        // in the ndc/pixelUnits/frac math this solve depends on, so one
        // target position is valid for both.
        let pos = solveWorldPosForFrac(basePos, fx, fy, right, up, w, h)
        pos = solveWorldPosForFrac(pos, fx, fy, right, up, w, h) // second correction for residual non-linearity
        const expected = computeExpectedWeights(pos, w, h)

        probeRecordsAttr.array.set([pos.x, pos.y, pos.z, TEST_FLUX])
        probeRecordsAttr.needsUpdate = true

        lines.push(
          `  target frac=(${fx},${fy}) -> solved frac=(${expected.fracX.toFixed(5)},${expected.fracY.toFixed(5)}) `
          + `expectedW=[${expected.weights.map((v) => v.toFixed(4)).join(',')}]`,
        )

        for (const { name, material } of variants) {
          const mesh = new THREE.Mesh(probeGeometry, material)
          mesh.frustumCulled = false
          const scene = new THREE.Scene()
          scene.add(mesh)

          renderer.setRenderTarget(target)
          renderer.setClearColor(0x000000, 0)
          renderer.clear()
          renderer.render(scene, camera)
          renderer.setRenderTarget(null)

          const gx0 = expected.basePixelX - Math.floor((GRID - 2) / 2)
          const gy0 = expected.basePixelY - Math.floor((GRID - 2) / 2)
          // eslint-disable-next-line no-await-in-loop
          const grid = await readRegionGrid(target, gx0, gy0, GRID, GRID)
          const cx = expected.basePixelX - gx0
          const cy = expected.basePixelY - gy0
          const rowAt = (r: number): number[] => grid[r] ?? []
          const measured: [number, number, number, number] = [
            rowAt(cy)[cx] ?? NaN, rowAt(cy)[cx + 1] ?? NaN,
            rowAt(cy + 1)[cx] ?? NaN, rowAt(cy + 1)[cx + 1] ?? NaN,
          ]
          let sum = 0
          let nonZeroCount = 0
          // ALL non-zero pixels found in the region, in ABSOLUTE
          // coordinates - not just the four PREDICTED coordinates above.
          // Team lead's discriminator: if sum/nonZeroCount are right but
          // measured4 has zeros where expectedW is non-zero, the missing
          // energy is somewhere else in the grid - this list says exactly
          // where, so it can be compared directly against the four
          // predicted coordinates rather than guessed at.
          const foundPixels: string[] = []
          for (let ry = 0; ry < grid.length; ry++) {
            const row = grid[ry] ?? []
            for (let rx = 0; rx < row.length; rx++) {
              const v = row[rx]
              if (v !== undefined && !Number.isNaN(v) && v !== 0) {
                sum += v
                nonZeroCount++
                foundPixels.push(`(${gx0 + rx},${gy0 + ry})=${v.toExponential(4)}`)
              }
            }
          }
          const predictedCoords = [
            `(${expected.basePixelX},${expected.basePixelY})`,
            `(${expected.basePixelX + 1},${expected.basePixelY})`,
            `(${expected.basePixelX},${expected.basePixelY + 1})`,
            `(${expected.basePixelX + 1},${expected.basePixelY + 1})`,
          ]

          lines.push(
            `    ${name}: measured4=[${measured.map((v) => v.toExponential(4)).join(',')}] `
            + `sum=${sum.toExponential(4)} nonZeroCount=${nonZeroCount} (expect 4, sum=1.0) `
            + `predicted coords=[${predictedCoords.join(',')}] `
            + `found (nonzero) coords=[${foundPixels.join(', ')}]`,
          )
        }
      }

      // eslint-disable-next-line no-console
      console.log(`[deposit coverage] flux=${TEST_FLUX}, camera frozen:\n${lines.join('\n')}`)

      target.dispose()
      variants.forEach((v) => v.material.dispose())
      probeGeometry.dispose()
    }
  }

  // Diagnostic, kept permanently: reads back the indirect-draw args buffer
  // (the 5 words geometry.setIndirect's draw actually consumes) and the
  // emission atomic counters, straight from the GPU. Both attributes were
  // created via createIndirectStorageAttribute/createStorageAttribute,
  // which three.js's WebGPU backend gives GPUBufferUsage.COPY_SRC by
  // default (WebGPUBackend.js), so getArrayBufferAsync needs no extra
  // staging-buffer plumbing — confirmed against that source, not guessed.
  // Triggered by 'p'; logs to console since this is a one-shot inspection,
  // not a per-frame HUD value.
  async function logParticleBuffers(): Promise<void> {
    // Full-target reads alongside the existing 128x128-centre ones - team
    // lead's flagged anomaly: nonZeroFrac=3.66e-4 in a centre crop is
    // exactly 6 non-zero pixels against thousands naively expected from
    // 99,908 emitted particles. That could mean the particles are
    // genuinely concentrated off-centre (plausible: near the ground,
    // looking roughly forward, the window's own projection need not be
    // centred on screen) or it could mean the centre crop was never
    // representative. Printing both totals side by side answers which,
    // directly, rather than building an energy argument on the centre
    // number alone.
    const [
      argsBuf, counterBuf, sampleBuf, probeBuf, recordsBuf,
      depositStats, resolveStats, depositFullStats, resolveFullStats,
    ] = await Promise.all([
      renderer.getArrayBufferAsync(particles.argsAttr),
      renderer.getArrayBufferAsync(particles.counterAttr),
      renderer.getArrayBufferAsync(particles.sampleAttr),
      renderer.getArrayBufferAsync(particles.probeAttr),
      // The REAL records buffer, not a synthetic probe - team lead's ask:
      // confirm the screen-space collapse independently of the deposit
      // readback, by checking whether the 3D positions emitKernel wrote
      // are themselves collapsed (upstream, emitKernel's problem) or well
      // spread (downstream, the projection/deposit's problem). `.value` is
      // the StorageInstancedBufferAttribute instancedArray wraps -
      // confirmed directly in StorageBufferNode.js (`this.value...`), same
      // COPY_SRC guarantee as every other storage readback in this file
      // (Attributes.js dispatches ANY storage-type attribute through
      // createStorageAttribute, not by its specific class).
      renderer.getArrayBufferAsync(particles.records.value),
      readTargetStats(splat.depositTarget),
      readTargetStats(splat.resolveTarget),
      readTargetStats(splat.depositTarget, Math.max(splat.depositTarget.width, splat.depositTarget.height)),
      readTargetStats(splat.resolveTarget, Math.max(splat.resolveTarget.width, splat.resolveTarget.height)),
    ])
    const [indexCount, instanceCount, firstIndex, baseVertex, firstInstance] = new Uint32Array(argsBuf)

    // Order matches Particles.ts's CTR_* constants exactly.
    const [emitted, overflow, ran, rejectedPastLiveCount, rejectedTaper, rejectedAltitude] = new Uint32Array(counterBuf)
    const [sampleTau] = new Float32Array(sampleBuf)
    // [dist, z] from the box-centre candidate - see Particles.ts's
    // probeBuf doc for why the centre, not instanceIndex 0.
    const [centreDist, centreZ] = new Float32Array(probeBuf)
    const accepted = (emitted ?? 0) - (overflow ?? 0)

    // Spread of the ACTUAL written positions/flux, over only the
    // `accepted` records that are valid this frame (records beyond that
    // are stale leftovers from whenever they were last overwritten, never
    // cleared - reading them would silently mix frames, the same class of
    // mistake as the staleness this whole thread just chased down).
    const recordsFloats = new Float32Array(recordsBuf)
    const validRecords = Math.max(0, Math.min(accepted, Math.floor(recordsFloats.length / 4)))
    const spread = (getComponent: (i: number) => number): { min: number, max: number, mean: number, stddev: number } => {
      if (validRecords === 0) return { min: NaN, max: NaN, mean: NaN, stddev: NaN }
      let min = Infinity
      let max = -Infinity
      let sum = 0
      for (let i = 0; i < validRecords; i++) {
        const v = getComponent(i)
        if (v < min) min = v
        if (v > max) max = v
        sum += v
      }
      const mean = sum / validRecords
      let sqSum = 0
      for (let i = 0; i < validRecords; i++) {
        const d = getComponent(i) - mean
        sqSum += d * d
      }
      return { min, max, mean, stddev: Math.sqrt(sqSum / validRecords) }
    }
    const spreadX = spread((i) => recordsFloats[i * 4] ?? NaN)
    const spreadY = spread((i) => recordsFloats[i * 4 + 1] ?? NaN)
    const spreadZ = spread((i) => recordsFloats[i * 4 + 2] ?? NaN)
    const spreadFlux = spread((i) => recordsFloats[i * 4 + 3] ?? NaN)
    const fmtSpread = (s: { min: number, max: number, mean: number, stddev: number }): string => (
      `min=${s.min.toFixed(3)} max=${s.max.toFixed(3)} mean=${s.mean.toFixed(3)} stddev=${s.stddev.toFixed(3)}`
    )

    /**
     * Tests one specific hypothesis: that `instanceIndex` does not vary
     * across the indirect draw, so every instance reads `records[0]` and
     * ~100k quads stack on one spot.
     *
     * Positions are known-good (the spread above) and projection is
     * known-good (the off-axis single-instance trace lands within a pixel
     * of prediction, energy conserved), so the remaining difference between
     * the working path and the broken one is that the real draw is indirect
     * and instanced while the probe is a single non-instanced quad.
     *
     * Two falsifiable predictions, printed here rather than argued:
     *  - if every instance reads record 0, the deposited total is
     *    `instanceCount * records[0].w`, so `depositSum / instanceCount`
     *    equals `records[0].w`;
     *  - and the deposit argmax is record 0's own projection, not any
     *    centroid of the field.
     * Both are printed beside their measured counterparts. If either misses,
     * the hypothesis is wrong and should be dropped rather than patched.
     */
    let rec0 = 'records[0]: unavailable'
    if (validRecords > 0) {
      const x0 = recordsFloats[0] ?? NaN
      const y0 = recordsFloats[1] ?? NaN
      const z0 = recordsFloats[2] ?? NaN
      const w0 = recordsFloats[3] ?? NaN
      const proj = computeExpectedWeights(
        new THREE.Vector3(x0, y0, z0), splat.depositTarget.width, splat.depositTarget.height,
      )
      const impliedW0 = depositFullStats.sum / Math.max(1, instanceCount ?? 1)
      rec0 = `records[0]=(${x0.toFixed(2)},${y0.toFixed(2)},${z0.toFixed(2)}) w=${w0.toExponential(4)} `
        + `projects to (${proj.basePixelX},${proj.basePixelY}) vs deposit argmax (${depositFullStats.argmaxX},${depositFullStats.argmaxY}) `
        + `| depositSum/instanceCount=${impliedW0.toExponential(4)} vs records[0].w=${w0.toExponential(4)} `
        + `ratio=${(impliedW0 / w0).toFixed(4)}`
    }

    /**
     * The quantitative test of the `safeW` guard, which inspection alone
     * cannot settle.
     *
     * `buildDepositMaterial` computes `safeW = clip.w <= 0 ? 1e6 : clip.w`
     * and then `ndc = clip.xy / safeW`. Its comment claims that pushes
     * behind-camera quads somewhere "off-screen"; for a point a few hundred
     * metres out `clip.xy` is O(500), so the result is `ndc ~ 5e-4`, and
     * NDC (0,0) is the screen CENTRE. Every behind-camera particle would
     * therefore land on the same pixel and accumulate additively.
     *
     * Two numbers decide it. `behind` is how many accepted records actually
     * have `clip.w <= 0` — if that is a large share of the population the
     * mechanism is real, not merely plausible. And the in-FRONT records get
     * their screen-space extent reported separately, because the guard
     * explains a centre pile-up but says nothing about the rest: if the
     * front subset also fails to spread, there is a second defect and
     * fixing the guard alone would leave it hidden behind a picture that
     * suddenly looks right.
     */
    let clipStats = 'clip: unavailable'
    if (validRecords > 0) {
      const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
      const e = m.elements
      const vw = splat.depositTarget.width
      const vh = splat.depositTarget.height
      let behind = 0
      let onScreen = 0
      let minPx = Infinity; let maxPx = -Infinity; let minPy = Infinity; let maxPy = -Infinity
      for (let i = 0; i < validRecords; i++) {
        const x = recordsFloats[i * 4] ?? 0
        const y = recordsFloats[i * 4 + 1] ?? 0
        const z = recordsFloats[i * 4 + 2] ?? 0
        const cw = (e[3] ?? 0) * x + (e[7] ?? 0) * y + (e[11] ?? 0) * z + (e[15] ?? 0)
        if (cw <= 0) { behind++; continue }
        const cx = (e[0] ?? 0) * x + (e[4] ?? 0) * y + (e[8] ?? 0) * z + (e[12] ?? 0)
        const cy = (e[1] ?? 0) * x + (e[5] ?? 0) * y + (e[9] ?? 0) * z + (e[13] ?? 0)
        const px = ((cx / cw) * 0.5 + 0.5) * vw
        const py = ((cy / cw) * 0.5 + 0.5) * vh
        if (px >= 0 && px < vw && py >= 0 && py < vh) {
          onScreen++
          if (px < minPx) minPx = px
          if (px > maxPx) maxPx = px
          if (py < minPy) minPy = py
          if (py > maxPy) maxPy = py
        }
      }
      // clip.w itself, because "100% behind" is a claim about a quantity
      // nobody has looked at. If w is uniformly a small negative number the
      // cloud is genuinely behind; if it is a large constant the records are
      // not in the space this matrix expects.
      let wLo = Infinity; let wHi = -Infinity; let wSum = 0
      for (let i = 0; i < validRecords; i++) {
        const x = recordsFloats[i * 4] ?? 0
        const y = recordsFloats[i * 4 + 1] ?? 0
        const z = recordsFloats[i * 4 + 2] ?? 0
        const cw = (e[3] ?? 0) * x + (e[7] ?? 0) * y + (e[11] ?? 0) * z + (e[15] ?? 0)
        if (cw < wLo) wLo = cw
        if (cw > wHi) wHi = cw
        wSum += cw
      }
      const pct = (n: number) => `${((n / validRecords) * 100).toFixed(1)}%`
      clipStats = `clip.w[min=${wLo.toFixed(3)} max=${wHi.toFixed(3)} mean=${(wSum / validRecords).toFixed(3)}] `
        + `viewFwdRow=(${(e[3] ?? 0).toFixed(4)},${(e[7] ?? 0).toFixed(4)},${(e[11] ?? 0).toFixed(4)},${(e[15] ?? 0).toFixed(4)}) `
        + `behind(w<=0)=${behind} (${pct(behind)}) inFrontAndOnScreen=${onScreen} (${pct(onScreen)}) `
        + (onScreen > 0
          ? `onScreen px extent x[${minPx.toFixed(0)}..${maxPx.toFixed(0)}] y[${minPy.toFixed(0)}..${maxPy.toFixed(0)}]`
          : 'no on-screen records')
    }
    // A single flattened string, not object/array args: some console
    // capture paths stringify multi-arg console.log calls by naive
    // concatenation rather than inspecting objects, which is what silently
    // turned the previous version's `{...}` into `[object Object]`.
    // eslint-disable-next-line no-console
    console.log(
      `[particles] indirect args: indexCount=${indexCount} instanceCount=${instanceCount} `
      + `firstIndex=${firstIndex} baseVertex=${baseVertex} firstInstance=${firstInstance} `
      + `| histogram: ran=${ran} rejectedPastLiveCount=${rejectedPastLiveCount} rejectedTaper=${rejectedTaper} `
      + `rejectedAltitude=${rejectedAltitude} emitted=${emitted} overflow=${overflow} accepted=${accepted} `
      + `| field sample at dir (1,0,0) = ${sampleTau} `
      + `| window: windowIn=${lastWindowIn} windowOut=${lastWindowOut} sigmaPix=${lastSigmaPix} `
      + `| centre candidate: dist=${centreDist} z=${centreZ} `
      + `| deposit(128x128 centre): peak=${depositStats.peak} mean=${depositStats.mean} nonZeroFrac=${depositStats.nonZeroFrac} sum=${depositStats.sum} `
      + `| resolve(128x128 centre): peak=${resolveStats.peak} mean=${resolveStats.mean} nonZeroFrac=${resolveStats.nonZeroFrac} sum=${resolveStats.sum} `
      + `| deposit(FULL ${splat.depositTarget.width}x${splat.depositTarget.height}): peak=${depositFullStats.peak} `
      + `mean=${depositFullStats.mean} nonZeroFrac=${depositFullStats.nonZeroFrac} sum=${depositFullStats.sum} `
      + `argmax=(${depositFullStats.argmaxX},${depositFullStats.argmaxY}) `
      + `| resolve(FULL ${splat.resolveTarget.width}x${splat.resolveTarget.height}): peak=${resolveFullStats.peak} `
      + `mean=${resolveFullStats.mean} nonZeroFrac=${resolveFullStats.nonZeroFrac} sum=${resolveFullStats.sum} `
      + `argmax=(${resolveFullStats.argmaxX},${resolveFullStats.argmaxY}) `
      // ACTUAL written record positions (render-space, camera-relative
      // metres) and flux, over the `validRecords` accepted this frame -
      // team lead's collapse-independent check. If x/y/z spread is near
      // zero here, the collapse is upstream in emitKernel, not the deposit
      // path - see this function's own comment above.
      + `| ${rec0} `
      + `| ${clipStats} `
      + `| records spread (n=${validRecords}): x[${fmtSpread(spreadX)}] y[${fmtSpread(spreadY)}] `
      + `z[${fmtSpread(spreadZ)}] flux[${fmtSpread(spreadFlux)}]`,
    )
  }

  addEventListener('keydown', (e) => {
    const occupied = explore.open || (e.target instanceof Element && Boolean(
      e.target.closest('button, a, input, textarea, select, summary, [contenteditable]:not([contenteditable="false"]), [role="button"]'),
    ))
    if (opening.handleSpace(e, occupied)) { e.preventDefault(); return }
    if (e.ctrlKey || e.metaKey || e.altKey) return
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return
    if (!viewerPolicy.allowsKey(e.key)) return
    if (e.key === 'p' || e.key === 'P') {
      renderFrame(0, true).then(logParticleBuffers).catch((err: unknown) => console.error('particle buffer readback failed:', err))
      return
    }
    if (e.key === 't' || e.key === 'T') {
      // Stop diagnostic camera drift for both calls - a run must be
      // reproducible to be falsifiable (team lead's own instruction after
      // the first version's numbers turned out to depend on the idle
      // autorotate drift). Sequential, not concurrent: both issue
      // renderer.setRenderTarget/render calls of their own, and running
      // them one after the other keeps console output in a predictable
      // order and avoids any risk of interleaved GPU state. Do not restore
      // drift asynchronously: a gesture during readback must keep control.
      opening.setAutoRotate(false)
      logSingleInstanceTrace()
        .then(() => logDepositCoverageTest())
        .catch((err: unknown) => console.error('deposit trace/coverage test failed:', err))
      return
    }
    if (e.key === 'd' || e.key === 'D') { toggleDataSource(); return }
    if (e.key === 's' || e.key === 'S') {
      setSamplesVisible(!showSamples)
      return
    }
    if (e.key === 'u' || e.key === 'U') {
      setUncertaintyVisible(!showUncertainty)
      return
    }
    if (e.key === 'i' || e.key === 'I') {
      showDiagnostics = !showDiagnostics
      diagnostics.style.display = showDiagnostics ? '' : 'none'
      if (showDiagnostics) openExplore()
      return
    }
    if (e.key === 'r' || e.key === 'R') {
      regimeMode = REGIME_MODES[(REGIME_MODES.indexOf(regimeMode) + 1) % REGIME_MODES.length]!
      return
    }
    if (e.key === 'v' || e.key === 'V') {
      debugDeposit = !debugDeposit
      return
    }
    // `?` (and `/`, so it works without reaching for shift) opens the full
    // key list that used to run onto a single line in the HUD.
    if (e.key === '?' || e.key === '/') {
      showKeys = !showKeys
      keyDetails.open = showKeys
      if (showKeys) openExplore()
      return
    }
    if (e.key === ']') exposure = Math.min(64, exposure * 2)
    else if (e.key === '[') exposure = Math.max(1, exposure / 2)
    else return
    applyExposure()
  })
  applyExposure()

  /**
   * Development-only measurement handle for visual checks — the same class
   * of thing as the `p`/`t` key traces above, and kept for the same reason.
   *
   * Every browser figure this project has argued about came from a
   * screenshot taken at a pose nobody could state exactly: the old idle
   * drift made a capture taken "at the default camera" ambiguous: it was
   * really a capture at whatever longitude the page happened to reach while
   * 200 MiB of atlases finished downloading. Two of this session's four
   * defects were mis-diagnosed from exactly that. `setPose` therefore SETS
   * the pose, stops the drift, and RETURNS what it reads back, so a caller
   * can assert the state it measured in the same step it measured it.
   *
   * `probeGround` exists to answer a question no screenshot can: whether a
   * camera-vs-terrain crossover tracks the elevation atlas or the 156 km
   * linear interpolation between `MATTER_GEOMETRY`'s vertices. It reports
   * both, at the same direction, off the same atlas the GPU is sampling.
   */
  const toDeg = (rad: number): number => (rad * 180) / Math.PI
  const nematodeDiagnostics = {
    /** Absolute pose in GEODETIC degrees (the convention `geoProbe.ts` uses). Stops the idle drift. */
    setPose(lonDeg: number, latDeg: number, altM: number) {
      opening.explore()
      scaleCamera.setOrientation((lonDeg * Math.PI) / 180, geodeticToGeocentricLat((latDeg * Math.PI) / 180))
      scaleCamera.setAltitude(altM)
      return this.pose()
    },
    pose() {
      return {
        lonDeg: toDeg(scaleCamera.longitude),
        latDegGeocentric: toDeg(scaleCamera.latitude),
        altM: scaleCamera.altitude,
        requestedAltM: scaleCamera.requestedAltitude,
        groundFloorM: scaleCamera.groundFloor,
        qualityFloorM: scaleCamera.qualityFloorM,
        atQualityFloor: scaleCamera.atQualityFloor,
        zoomQuality: zoomQuality && {
          ...zoomQuality,
          viewportWidthCssPx: projectionViewport.width,
          viewportHeightCssPx: projectionViewport.height,
          projectedTexelCssPx: zoomQuality.worstTexelM * zoomQuality.focalCssPx / scaleCamera.altitude,
        },
        autoRotate: opening.autoRotate,
        readiness: opening.readiness,
        sweepT: timeline.t,
        matter: timeline.state.matter,
        exaggeration: exaggerationUniform.value,
        bathymetry: bathymetryUniform.value,
        activeSource,
        densityFaceSize: realCube?.meta.face_size ?? null,
        densitySchema: realCube?.meta.schema ?? null,
        elevationFaceSize: elevationCube?.meta.face_size ?? null,
        urbanBound: urbanCubeTex !== null,
        groundFloorEnabled,
      }
    },
    setAutoRotate(on: boolean) { opening.setAutoRotate(on); return this.pose() },
    setSweep(t: number) { timeline.pause(); timeline.scrubTo(t); return this.pose() },
    /** A/B control for the ground floor itself — measuring the defect it fixes requires being able to turn it off. */
    setGroundFloor(on: boolean) { groundFloorEnabled = on; return this.pose() },
    /** What the mesh under a geodetic lat/lon is actually doing, atlas value and drawn surface side by side. */
    probeGround(lonDeg: number, latDeg: number) {
      if (!elevationCube) return null
      const dir = latLonToDirection(latDeg, lonDeg)
      const opts = {
        widthSegments: MATTER_WIDTH_SEGMENTS,
        heightSegments: MATTER_HEIGHT_SEGMENTS,
        exaggeration: exaggerationUniform.value,
        bathymetryProgress: bathymetryUniform.value,
      }
      const { faces, meta } = elevationCube
      const atlas = sampleElevationCube(faces, meta, dir)
      return {
        faceSize: meta.face_size,
        atlasElevationM: atlas.elevationM,
        landFraction: atlas.landFraction,
        // Three surfaces, deliberately reported side by side, because a
        // measured camera-vs-terrain crossover has to be compared against
        // all three to say WHICH of them it tracks: the atlas at the point,
        // a bilinear reading of the cell's displaced vertices, and the
        // cell-maximum bound the floor is actually built on.
        bilinearVertexM: interpolatedDisplacementM(faces, meta, dir, opts),
        cellMaxM: cellMaxDisplacementM(faces, meta, dir, opts),
        exaggeration: opts.exaggeration,
      }
    },
    /**
     * Draws `n` frames synchronously-on-demand at dt = 0, independent of
     * `requestAnimationFrame` (which a backgrounded harness tab never fires)
     * and independent of the idle drift and the timeline, both of which are
     * driven by dt. More than one because the particle pass is deliberately
     * one frame behind (regimes/Particles.ts): the first frame after a pose
     * change still composites the previous pose's records.
     */
    async renderOnce(n = 3) {
      for (let i = 0; i < n; i++) await renderFrame(0)
      return this.pose()
    },
    /** Resolves after `n` rendered frames, so a capture is never taken mid-change. */
    waitFrames(n = 3) {
      return new Promise<void>((resolve) => {
        let left = n
        const step = (): void => { if (--left <= 0) resolve(); else requestAnimationFrame(step) }
        requestAnimationFrame(step)
      })
    },
  }
  viewerPolicy.exposeDiagnostics(globalThis, nematodeDiagnostics)

  let lastFrameMs = performance.now()

  /**
   * One frame, given the wall-clock step to advance by. Split out of
   * `setAnimationLoop` so a frame can also be drawn ON DEMAND, at `dtSec = 0`
   * — see `nematodeDiagnostics.renderOnce`.
   *
   * That is not a convenience. `requestAnimationFrame` does not fire in a
   * backgrounded tab, and an automated harness runs the page backgrounded, so
   * every capture taken that way is of a canvas nothing ever drew to: black,
   * indistinguishable from the genuinely-black frame this session was sent to
   * investigate. Driving the frame directly removes the harness's own
   * visibility state from the measurement entirely.
   */
  const renderFrame = async (dtSec: number, measureParticles = false): Promise<void> => {
    // The sweep is a pure function of one number (Timeline.ts) - advancing
    // it here just moves that number forward while playing; scrubbing
    // (the range input's own 'input' listener) sets it directly and pauses,
    // and neither path knows the other exists. dtSec is clamped here (not
    // inside Timeline) so a backgrounded tab or a mid-sweep material
    // rebuild stall can't jump straight to T_END in one step.
    opening.advance(dtSec)
    const sweepState = timeline.state
    // One front per matter layer - see Matter.ts's doc for how each drives
    // the terrain shader (atmosphere -> rim intensity, water -> ocean
    // discard + the bathymetry collapse, rock -> land discard).
    atmosphereDissolve.front.value = sweepState.frontAtmosphere
    waterDissolve.front.value = sweepState.frontWater
    rockDissolve.front.value = sweepState.frontRock
    matter.visible = sweepState.frontWater < 1 || sweepState.frontRock < 1
    bathymetryUniform.value = sweepState.bathymetry

    // Descent schedule, live: Hs/Hmax shrink (and sigma0PerTau/tauCap follow)
    // as altitude drops, per math/descent.ts - this is what makes the shell
    // actually thin out on descent rather than just reporting a number.
    const state = scaleCamera.state
    Hs.value = state.Hs
    Hmax.value = state.Hmax
    s0.value = sigma0(state.tauRendered, state.Hs, state.Hmax)
    sigma0PerTau.value = sigma0(1, state.Hs, state.Hmax)
    tauCap.value = saturationCapTau(state.Hs)
    // state.k IS math/descent.ts's exaggerationAt(altitude) - the same
    // ramp Matter's terrain displacement/relief-normal use, so mountains
    // and the HUD's own `k` readout always agree.
    exaggerationUniform.value = state.k

    // Floating origin: the camera stays at the render origin; the shell
    // (whose true centre is the world origin) is placed at
    // worldToRender([0,0,0]), i.e. -cameraWorldPosition, computed in f64 on
    // the CPU. cameraWorldPosition (the camera's TRUE planet-relative
    // position) still has to reach the shader some other way, since TSL's
    // own cameraPosition builtin now reads (0,0,0) - see Shell.ts. Matter's
    // own shading (Matter.ts) uses normalWorld instead, which needs no such
    // fallback - see that file's doc.
    const worldPos = scaleCamera.worldPosition
    cameraWorldPosition.value.set(worldPos[0], worldPos[1], worldPos[2])

    // Sample points carry their own CPU horizon test rather than relying on
    // the depth buffer: the shell is a volumetric pass with depthWrite off,
    // so there is no populated depth to test far-side points against (see
    // SamplePoints.ts's update() doc). Only worth the O(n) pass when the
    // layer is actually on screen.
    if (samplePoints && showSamples) samplePoints.update(worldPos)

    camera.position.set(0, 0, 0)
    const originRenderPos = scaleCamera.worldToRender([0, 0, 0])
    shell.position.set(originRenderPos[0], originRenderPos[1], originRenderPos[2])
    matter.position.copy(shell.position)

    const up = projectedUp(worldPos)
    camera.up.set(up[0], up[1], up[2])
    camera.lookAt(originRenderPos[0], originRenderPos[1], originRenderPos[2])

    altVal.textContent = formatAltitude(scaleCamera.altitude)
    // Announce only the transition, not every render/overscroll event. Reuse
    // the quiet gesture line so the limit adds no panel or layout shift.
    if (showingQualityHint !== scaleCamera.atQualityFloor) {
      showingQualityHint = scaleCamera.atQualityFloor
      hints.textContent = showingQualityHint ? 'Maximum detail for this dataset' : gestureHint
      hints.dataset.qualityLimit = String(showingQualityHint)
    }

    // --- count in view (spec 8.3) ---------------------------------------
    //
    // Local density sampled at the camera's own nadir direction: `worldPos`
    // normalized IS that direction, since this app's camera always looks at
    // the planet's centre regardless of orbital position (see the floating-
    // origin block above), so whatever's "closest below" is always exactly
    // where the camera is pointed. Data source matches `activeSource` (real
    // cube probe vs synthetic tau -> density) so the readout never disagrees
    // with what's actually on screen.
    const wpLen = Math.hypot(worldPos[0], worldPos[1], worldPos[2]) || 1
    const nadirDir: readonly [number, number, number] = [worldPos[0] / wpLen, worldPos[1] / wpLen, worldPos[2] / wpLen]
    const localDensityPerSqm = activeSource === 'real' && realCube
      ? sampleCubeDensity(realCube.faces, realCube.meta, nadirDir).density
      : syntheticTau(nadirDir[0], nadirDir[1], nadirDir[2]) / E_A
    const meanDensityPerSqm = activeSource === 'real' && realCube
      ? GLOBAL_MEAN_DENSITY_PER_SQM
      : SYNTHETIC_MEAN_DENSITY_PER_SQM
    const countInView = estimatedCountInView(
      scaleCamera.altitude, localDensityPerSqm, meanDensityPerSqm,
      (camera.fov * Math.PI) / 180, camera.aspect,
    )
    countVal.textContent = formatCount(countInView)
    countCompare.textContent = humanComparison(countInView)

    hsVal.textContent = `${(state.Hs / 1000).toFixed(2)} km`
    kVal.textContent = state.k.toFixed(2)
    tauVal.textContent = state.tauRendered.toFixed(3)
    qVal.textContent = state.Q.toFixed(2)
    beatVal.textContent = `${sweepState.beat} (${playbackSeconds(sweepState.t).toFixed(1)}s / ${playbackSeconds(T_END)}s)`
    cobbVal.textContent = COBB_LINE.slice(0, sweepState.titleChars)
    quoteAttrib.style.visibility = sweepState.titleChars >= COBB_LINE.length ? '' : 'hidden'
    introduction.hidden = sweepState.t > 0
    quote.hidden = sweepState.t === 0

    updateOpeningControls()
    sweepGuide.update(sweepState, activeSource === 'real', opening.showIntro)
    scrub.value = String(timeline.t)
    scrub.setAttribute('aria-valuetext', `${playbackSeconds(timeline.t).toFixed(1)} seconds of ${playbackSeconds(T_END)}`)
    scrub.style.setProperty('--progress', `${timeline.t / T_END * 100}%`)
    scrubVal.textContent = `0:${Math.floor(playbackSeconds(timeline.t)).toString().padStart(2, '0')}`

    // --- Regime 2 partition + particle frame state --------------------
    //
    // The comparison modes control the partition purely through
    // shellPartitionShare - 'shell' pins it at 1 (unpartitioned, exactly
    // Session 1-3 behaviour) and 'particles' pins it at 0 (shell
    // contributes nothing) so the SAME shader path is exercised in all
    // three modes; only the uniform's value differs. 'blended' uses the
    // real regimes/Blend.ts partition.
    const partition = partitionAt(scaleCamera.altitude, TAU_MEAN)
    shellPartitionShare.value = regimeMode === 'particles' ? 0 : regimeMode === 'shell' ? 1 : (1 - partition.w)
    // The last solid fragment disappears before the timeline's settle ends.
    // Reveal the background during that interval and reverse it when scrubbing.
    farWallVisibility.value = farWallVisibilityAt(sweepState.t)
    farWallEnabled.value = !matter.visible && farWallVisibility.value > 0 && shellPartitionShare.value >= 1

    const windowOut = Math.min(Math.max(scaleCamera.altitude * 0.5, 500), 50000)
    const drawParticles = regimeMode !== 'shell' && (
      regimeMode === 'particles' || debugDeposit || particleWindowIntersectsShell(wpLen, windowOut, state.Hmax)
    )
    renderer.render(scene, camera)
    regimeModeVal.textContent = regimeMode + (debugDeposit ? ' [raw deposit — v to exit]' : '')
    if (!drawParticles && !measureParticles) {
      particleDrawReady = false
      regimeStatsFaint.textContent = ' (particle pass inactive)'
      return
    }

    const windowIn = windowOut * 0.1
    lastWindowIn = windowIn
    lastWindowOut = windowOut
    // math/lod.ts's cellSizeForBudget, not an ad-hoc altitude/200 formula -
    // that formula sized cells independently of PARTICLE_BUDGET, which (at
    // this session's ~10.7 km altitude) undersized cells enough that dim^3
    // ran ~1.93x past budget (see finalizeKernel's new compFactor - that
    // fixes the ALREADY-overflowing candidates' uncompensated energy loss,
    // this fixes the overflow itself being far larger than intended).
    // Volume is the windowOut sphere - the region taper>0 actually covers,
    // matching what cellSizeForBudget's doc means by "fits budget cells
    // into this volume". Clamped to the same [1,2000] sanity bound the old
    // formula used, as a safety net against a degenerate cube root.
    const candidateVolumeM3 = (4 / 3) * Math.PI * windowOut ** 3
    const cellSize = Math.min(Math.max(cellSizeForBudget(candidateVolumeM3, PARTICLE_BUDGET), 1), 2000)
    const particleStats = particles.update({
      camWorld: worldPos,
      altM: scaleCamera.altitude,
      windowIn,
      windowOut,
      Hs: state.Hs,
      Hmax: state.Hmax,
      sigma0PerTau: sigma0(1, state.Hs, state.Hmax),
      tauCap: saturationCapTau(state.Hs),
      w: regimeMode === 'particles' ? 1 : regimeMode === 'shell' ? 0 : partition.w,
      cellSize,
    })

    // math/lod.ts's sigma_pix schedule (spec 7.7): ~3px at the 10km end,
    // tightening toward 0.5-0.75px as the handoff completes - was
    // previously never applied (buildSplatPipeline's initial 3 was the
    // ONLY value ever used; setSigmaPix existed but nothing called it).
    // Rounded to 0.1px and only re-applied on change, since setSigmaPix
    // disposes and rebuilds both resolve materials (new taps, new WGSL) -
    // doing that unconditionally every frame would be a shader rebuild
    // per frame instead of only when the schedule actually moves.
    const sigmaPixTarget = Math.round(sigmaPixAt(scaleCamera.altitude) * 10) / 10
    if (sigmaPixTarget !== lastSigmaPix) {
      splat.setSigmaPix(sigmaPixTarget)
      lastSigmaPix = sigmaPixTarget
    }

    regimeStatsFaint.textContent = ` (synthetic particle test, ${particleStats.candidates.toLocaleString()} candidates, ${particleStats.cellSize.toFixed(1)} m cells, ${(REGIME2_FULL_ALT / 1000).toFixed(1)}–${(REGIME2_TOP_ALT / 1000).toFixed(1)} km altitude)`

    // On re-entry, discard the indirect draw left at the previous pose.
    // Submit fresh records before the first draw. Subsequent frames retain
    // the normal one-frame overlap between drawing and computing.
    const primeDraw = drawParticles && !particleDrawReady
    if (primeDraw) renderer.compute(particleKernels)
    particleDrawReady = drawParticles

    if (drawParticles) {
      renderer.setRenderTarget(splat.depositTarget)
      renderer.setClearColor(0x000000, 0)
      renderer.clear()
      renderer.render(splat.depositScene, camera)

      // Separable resolve: horizontal pass reads the deposit target, into
      // blurHTarget; vertical pass reads THAT, into resolveTarget. See
      // SplatBlur.ts's module doc for why this replaced a single 2D pass
      // (WGSL parser recursion limit on 441 unrolled taps).
      renderer.setRenderTarget(splat.blurHTarget)
      renderer.clear()
      renderer.render(splat.blurHScene, splat.resolveCamera)

      renderer.setRenderTarget(splat.resolveTarget)
      renderer.clear()
      renderer.render(splat.resolveScene, splat.resolveCamera)

      renderer.setRenderTarget(null)
      renderer.autoClear = false
      renderer.render(compositeScene, splat.resolveCamera)
      if (debugDeposit) {
        // Diagnostic: depositTarget.texture raw, unblurred, uncomposited -
        // see debugDepositMaterial's construction above. Opaque, drawn
        // last, so it fully overwrites the composite pass rather than
        // blending with it. Toggle with 'v'.
        renderer.render(debugDepositScene, splat.resolveCamera)
      }
      renderer.autoClear = true
    }

    if (!primeDraw) await renderer.computeAsync(particleKernels)
  }

  renderer.setAnimationLoop(async () => {
    const nowMs = performance.now()
    const dtSec = (nowMs - lastFrameMs) / 1000
    lastFrameMs = nowMs
    await renderFrame(dtSec)
  })
}
