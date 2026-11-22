import * as THREE from 'three/webgpu'
import {
  Fn, vec3, vec4, float, Loop, If, Break, positionWorld, cameraPosition,
  normalize, dot, sqrt, max, min, exp, smoothstep, length, mix, clamp,
} from 'three/tsl'
import { syntheticTauNode } from '../field/syntheticTSL'
import {
  smoothstepCubeDirectionNode, uncertaintyCoverageNode, uncertaintyOverlayColorNode,
  veilDitherScaleNode,
} from '../field/terrainNodes'
import { cubeFaceSize } from '../field/loadCube'
import { filteredCvNode, filteredSupportNode, filteredTauNode, supportedAreaMeanNode } from '../field/densityCoverage'
import { geographicCubeTexture as cubeTexture } from '../field/geographicCube'

export interface ShellFieldOpts {
  /** Runtime switch: true samples the analytic synthetic field per step, false falls back to the uniform `sigma0`. */
  useField: THREE.UniformNode<'bool', boolean>
  /**
   * `1 / (Hs * (1 - exp(-Hmax/Hs)))` — the CPU `sigma0(tau, Hs, Hmax)`
   * relationship with `tau` factored out, since that division is constant
   * per frame. Per-step sigma0 from the field is `tauRendered * sigma0PerTau`.
   * Equal to `sigma0(1, Hs, Hmax)` from `math/optical.ts`.
   */
  sigma0PerTau: THREE.UniformNode<'float', number>
  /**
   * Saturation cap on rendered nadir tau, `4/Q` (spec section 7.3). Constant
   * per frame — equal to `saturationCapTau(Hs)` from `math/optical.ts`.
   */
  tauCap: THREE.UniformNode<'float', number>
}

export interface ShellCubeOpts {
  /**
   * Plain geographic density cube. The texture's encoding selects linear
   * soil-v2 optical depth or the legacy/synthetic log-depth representation.
   */
  texture: THREE.CubeTexture
  /** Same meaning as `ShellFieldOpts.sigma0PerTau` — see its doc. */
  sigma0PerTau: THREE.UniformNode<'float', number>
  /** Same meaning as `ShellFieldOpts.tauCap` — see its doc. */
  tauCap: THREE.UniformNode<'float', number>
}

export interface ShellOpts {
  Hs: THREE.UniformNode<'float', number>
  Hmax: THREE.UniformNode<'float', number>
  /** Uniform-density fallback / scale, and the sole density source when neither `field` nor `cube` is set. */
  sigma0: THREE.UniformNode<'float', number>
  /**
   * Per-step analytic field wiring. Omitted entirely, no field code is
   * emitted at all, not even behind a runtime branch — which is what keeps
   * a `buildShellMaterial` call that sets neither `field` nor `cube`
   * (Session 1's measured baseline) reproducible.
   *
   * That used to say "reproduces Session 1's Exp-1 uniform-density shader
   * byte-for-byte", and no longer can: the output dither at the end of
   * `buildShellMaterial` is unconditional (see its comment for why), so
   * every config carries four extra ALU per FRAGMENT. Nothing inside the
   * march loop changed, so exp-01's per-step cost measurement is unaffected
   * at 128 steps a fragment, and exp-02's WGSL checks are structural
   * (explicit LOD, no unrolling) and see no new texture sampling or `exp()`
   * calls. The claim being narrowed is textual identity, not behaviour under
   * either experiment. Mutually exclusive with
   * `cube` — if both are set, `cube` takes precedence (see `stepSigma0`).
   */
  field?: ShellFieldOpts
  /**
   * Per-step cube-texture wiring: samples a pre-baked field instead of
   * evaluating it analytically. Like `field`, its absence emits no cube-
   * sampling code at all. Takes precedence over `field` if both are set,
   * though callers should treat them as mutually exclusive — each
   * `buildShellMaterial` call represents exactly one of the three configs
   * (uniform / analytic field / cube), never a runtime mix of the last two.
   */
  cube?: ShellCubeOpts
  /**
   * The camera's TRUE planet-relative world position. Only needed by callers
   * using floating-origin rendering - `camera.position` kept at the render
   * origin while the scene is translated by `-cameraWorldPosition` (spec
   * section 8.1's "the camera sits permanently at the render origin and the
   * world translates around it"). In that scheme TSL's `cameraPosition`
   * builtin reads `(0,0,0)`, which is meaningless as the sphere-space ray
   * origin `ro` below - it no longer carries the camera's position relative
   * to the PLANET's centre, only relative to itself.
   *
   * Omitted (as Session 1's offscreen probes do, where the camera sits
   * directly at its true world position and nothing is translated), `ro`
   * falls back to the `cameraPosition` builtin - byte-for-byte Session 1
   * behavior. `rd` needs no equivalent fallback: `positionWorld.sub(cameraPosition)`
   * is already correct in both schemes - under floating origin the builtin
   * `cameraPosition` is exactly 0, so the subtraction is a no-op and `rd`
   * is derived from the now-small, now-precise `positionWorld` directly;
   * without it, both operands are the same true-world-position pair Session 1
   * always used.
   */
  cameraWorldPosition?: THREE.UniformNode<'vec3', THREE.Vector3>
  /**
   * The Nelson Max LOD partition (regimes/Blend.ts), additive and optional:
   * absent, the shell renders exactly as it always has (Session 1's
   * measured baseline, `field`/`cube` unaffected either way). Present, only
   * with `field` or `cube` also set - a uniform-only shell has no per-step
   * tauSampled for the partition to act on.
   */
  partition?: ShellPartitionOpts
  /**
   * The far wall of the shell - the ray's SECOND crossing of the annular
   * density region, on the opposite side of the planet from the camera.
   * Absent entirely, the march stops at the near wall exactly as it always
   * has (byte-for-byte prior behaviour, not even a runtime branch emitted -
   * same discipline as `field`/`cube`/`partition`/`urban` above). Present,
   * the far wall is still gated PER-FRAME by `enabled` - see
   * `ShellFarWallOpts`'s doc for the two conditions that must both hold.
   */
  farWall?: ShellFarWallOpts
  /**
   * The Cobb inversion slider (spec 6.3), CORRECTED per
   * `docs/superpowers/results/2026-08-16-urban-claim-ruling.md`: the
   * original spec's `m_data(u) = 1 - 1.70u + 0.75u^2` ("cities as voids")
   * is unshippable, not merely uncalibrated — the source model is
   * structurally blind to urban land use and the training data runs the
   * OPPOSITE sign. The measured effect is null (a = -0.0085, CI
   * [-0.035, 0.019]), so the corrected data-end multiplier is exactly 1
   * (no modification), and only `m_cobb(u) = 1 + 3u` — explicitly labelled
   * a 1914 prediction, never presented as data — is real. Absent entirely,
   * density is unaffected (byte-for-byte prior behaviour); only `field` or
   * `cube` density (not the uniform fallback) has a per-direction `u` to
   * multiply.
   */
  urban?: ShellUrbanOpts
  /**
   * The model-uncertainty overlay (`u`). Absent entirely, not one line of
   * CV code is emitted — same build-time discipline as `field`/`cube`/
   * `partition`/`farWall`/`urban` above. Only the `cube` config has a B
   * channel to read; the analytic field has no uncertainty to report,
   * because it is not a measurement.
   */
  uncertainty?: ShellUncertaintyOpts
  /** Nyquist on 4.9 km texels wants ~261 at the limb; 128 relies on mip prefiltering. */
  nMax: number
}

export interface ShellUncertaintyOpts {
  /**
   * 0 = overlay off, 1 = full. A per-frame UNIFORM rather than a build-time
   * flag so `u` is a live toggle: at 0 the overlay term is an exact no-op
   * (`uncertaintyOverlayColor`'s doc — `mix(base, ..., 0)` returns `base`
   * identically), so nothing needs rebuilding to turn it off, and pressing
   * `u` never drops a frame or re-uploads an atlas.
   */
  strength: THREE.UniformNode<'float', number>
}

/**
 * Minimum rendered luminance for land the overlay covers, in the same units
 * as the march's accumulated `col`.
 *
 * The veil's own brightness cannot be the overlay's only light source. CV is
 * strongly ANTI-correlated with density in this dataset — the arid interiors
 * are simultaneously the sparsest and the least-certain places on the map
 * (Sahara: 9.1e5/m² and CV 0.259, against Europe's 4.0e6/m² and CV 0.130) —
 * so a purely multiplicative overlay would be dimmest exactly where it has
 * the most to say. This floor is gated by `paint` (strength * coverage), so
 * it is identically zero over ocean and identically zero whenever the
 * overlay is off. Chosen near the disc's own measured alpha at exposure 1
 * (~0.046, see viewer.ts) so covered land reads as a moderately dense
 * region rather than as a blown-out light source.
 */
export const UNCERTAINTY_MIN_LUMA = 0.06

/**
 * Per-step scratch for the uncertainty overlay: the CV and land fraction of
 * whichever texel the current march step landed in. Written by `stepSigma0`
 * (off the cube sample it already takes) and read by `marchSegment` one line
 * later, which is where the step's own contribution to the image is known
 * and can be used as the averaging weight.
 */
interface StepProbe {
  cv: FloatNode
  land: FloatNode
}

/**
 * Running, contribution-weighted totals for the overlay, accumulated across
 * BOTH shell walls and resolved once at the end of the fragment.
 *
 * Weighted by each step's own contribution to `col` rather than by a flat
 * step count: the march runs a fixed `nMax` steps through the whole shell,
 * most of which sit in near-vacuum far above the density, and a flat mean
 * would average those in equally — dragging every column toward whatever the
 * thin upper atmosphere happens to carry instead of reporting the CV of the
 * ground actually being looked at.
 */
interface UncertaintyAccum {
  cvWeighted: FloatNode
  landWeighted: FloatNode
  weight: FloatNode
}

export interface ShellUrbanOpts {
  /** GHSL built-up fraction (`.r`) — the same cube `sweep/Matter.ts`'s city lights sample. */
  texture: THREE.CubeTexture
  /** 0 = the measurement (flat, no urban effect), 1 = Cobb's 1914 prediction (towns bright) — see `ShellOpts.urban`'s doc. */
  cobbMix: THREE.UniformNode<'float', number>
}

export interface ShellPartitionOpts {
  /**
   * `1 - handoffWeight(altitude)` (regimes/Blend.ts's `handoffWeight`,
   * computed on the CPU from altitude alone and updated once per frame,
   * not per step). Multiplies the shell's own already-capped rendered
   * sigma0, nothing else.
   *
   * Blend.ts's `shellShare(w, tauRendered, tauData) = (1-w) * (tauRendered/tauData)`
   * looks like it needs `tauData` (the field's raw, uncapped value)
   * separately from `tauRendered` (`min(tauSampled, tauCap)`, already
   * computed below) - but `shellShare * tauData = (1-w) * tauRendered`
   * algebraically, `tauData` cancelling exactly. So the shell only ever
   * needs `(1-w)` itself: multiplying the EXISTING `tauRendered` by it is
   * the whole coupling, not a second field sample or a fresh ratio.
   *
   * `1` (i.e. w=0, at and above `REGIME2_TOP_ALT`) reproduces the
   * unpartitioned shell exactly. `0` (w=1, at and below `REGIME2_FULL_ALT`)
   * hands the shell nothing.
   */
  shellFraction: THREE.UniformNode<'float', number>
}

export interface ShellFarWallOpts {
  /** The solid globe has dissolved and the shell has its full share of the optical depth. */
  enabled: THREE.UniformNode<'bool', boolean>
  /** Display transition from zero to the full far-side column. Probes default to full visibility. */
  visibility?: THREE.UniformNode<'float', number>
}

type FloatNode = THREE.Node<'float'>
type Vec3Node = THREE.Node<'vec3'>

const A = 6378137.0
const B = 6356752.314245

/**
 * tau -> sigma0 for a config that already produced a sampled tau (either the
 * analytic field or the cube texture): cap per spec section 7.3 so the rim
 * never blows out during descent, then scale by the precomputed
 * `sigma0(tau,Hs,Hmax)` bracket. Shared by both `field` and `cube` branches
 * of `stepSigma0` below - the only difference between those two configs is
 * how `tauSampled` itself is obtained.
 */
function sigma0FromTau(
  tauSampled: FloatNode,
  sigma0PerTau: FloatNode,
  tauCap: FloatNode,
  shellFraction?: FloatNode,
): FloatNode {
  const sigma0 = min(tauSampled, tauCap).mul(sigma0PerTau)
  // See ShellPartitionOpts.shellFraction's doc for why this single
  // multiply - not a second field sample or tauData/tauRendered ratio - is
  // the entire Nelson Max coupling.
  return shellFraction ? sigma0.mul(shellFraction) : sigma0
}

/**
 * Cobb-inversion multiplier on a sampled tau, applied BEFORE the saturation
 * cap in `sigma0FromTau` so an artificially-brightened Cobb-mode city is
 * capped the same way any other high-density region is, not given a
 * separate exemption. `mix(1, 1+3u, cobbMix)` — see `ShellOpts.urban`'s doc
 * for why the data end is exactly 1 (no multiplier) rather than the
 * original spec's fabricated suppression curve.
 */
function applyCobbMix(tauSampled: FloatNode, worldDirection: Vec3Node, urban?: ShellUrbanOpts): FloatNode {
  if (!urban) return tauSampled
  // Warped on the URBAN atlas's own texel pitch, not the density cube's —
  // the two tiers differ, and a warp at the wrong spatial frequency is
  // worse than none. Same reconstruction fix as the density sample below;
  // see field/loadCube.ts::smoothstepCubeDirection for why.
  const urbanDir = smoothstepCubeDirectionNode(worldDirection, cubeFaceSize(urban.texture))
  const builtUp = cubeTexture(urban.texture, urbanDir).level(float(0)).r
  const cobbPrediction = float(1).add(builtUp.mul(3))
  const multiplier = mix(float(1), cobbPrediction, clamp(urban.cobbMix, 0, 1))
  return tauSampled.mul(multiplier)
}

/**
 * Base per-step sigma0 (before the altitude e-fold): cube-sampled if
 * `opts.cube` is set, else the analytic field if `opts.field` is set, else
 * the uniform fallback - decided at graph-construction time from which
 * options are present (plain JS values, not a shader-time branch), so the
 * Session 1 baseline shader is unaffected when neither is set.
 */
function stepSigma0(opts: ShellOpts, worldDirection: Vec3Node, probe?: StepProbe): FloatNode {
  if (opts.cube) {
    const { texture, sigma0PerTau, tauCap } = opts.cube
    // Cube textures have no textureLoad overload in WGSL and the march's
    // transmittance Break() makes control flow non-uniform, so this MUST be
    // explicit-LOD (.level(0)), never the implicit-LOD default - see
    // regimes/Shell.ts's module doc and exp02-wgsl.ts's WGSL-text check.
    // Geographic cubes use world/geocentric direction, like the terrain.
    // Sphere-space ps is ONLY for altitude/intersection: scaling Z by 1/B
    // changes latitude and would shift this field relative to the terrain.
    // Cube sampling ignores magnitude, so no normalization is needed. Both
    // channels come off the same sample so the compiler shares the fetch.
    //
    // It goes through smoothstepCubeDirectionNode first, and that is the
    // reconstruction fix measured on the August 2026 legacy atlas:
    // 78.2 km/texel drawn at ~31 screen pixels per texel. At that scale
    // the filter dominated: hardware bilinear is C0, its gradient jumps across
    // every texel-cell boundary, and 100.0% of the magnified field's
    // second-derivative mass therefore lands exactly on the texel grid —
    // which is the polygonal faceting the veil showed. Warping the texel
    // coordinate through smoothstep before sampling makes the
    // reconstruction C1 for a few ALU while leaving the filtering itself in
    // the texture unit. Rebaking that legacy ~78 km/cell extraction finer
    // could not recover detail absent from it. The full-local candidate
    // uses a conservative 2048x1024 grid and 512-face output (~19.55 km
    // mean pitch); the legacy result is not its resolution limit. This
    // warp uses the supplied texture's face size, independent of promotion.
    // See field/loadCube.ts::smoothstepCubeDirection
    // for the derivation, including why this cannot open a face seam.
    const dir = smoothstepCubeDirectionNode(worldDirection, cubeFaceSize(texture))
    const cubeSample = cubeTexture(texture, dir).level(float(0))
    // V2 R is the all-cell area mean, with unsupported taps set to zero
    // before filtering. Applying A again would count missing area twice.
    const tauSampled = applyCobbMix(filteredTauNode(texture, cubeSample.r), worldDirection, opts.urban)
    const landFraction = clamp(filteredSupportNode(texture, cubeSample.a), 0, 1)
    if (probe) {
      probe.cv.assign(filteredCvNode(texture, cubeSample.b, cubeSample.a))
      probe.land.assign(landFraction)
    }
    return sigma0FromTau(supportedAreaMeanNode(tauSampled, landFraction), sigma0PerTau, tauCap, opts.partition?.shellFraction)
  }
  if (opts.field) {
    const { useField, sigma0PerTau, tauCap } = opts.field
    // syntheticTauNode normalizes internally (mirroring syntheticTau's own
    // Math.hypot normalization), so world direction needs no pre-normalization.
    const tauSampled = applyCobbMix(syntheticTauNode(worldDirection), worldDirection, opts.urban)
    const fieldSigma0 = sigma0FromTau(tauSampled, sigma0PerTau, tauCap, opts.partition?.shellFraction)
    return useField.select(fieldSigma0, opts.sigma0)
  }
  return opts.sigma0
}

/**
 * Marches ONE shell-wall segment - `opts.nMax` steps of width `dt`, starting
 * at world-space ray parameter `tEnterX` - accumulating into the shared
 * `col`/`transmittance` vars. Factored out so the near wall (always) and the
 * far wall (opts.farWall only, see its doc) run byte-identical per-step
 * physics through two different entry points, rather than two hand-copied
 * Loop bodies that could drift apart.
 */
function marchSegment(
  opts: ShellOpts,
  ro: Vec3Node, rd: Vec3Node,
  invA: FloatNode, invB: FloatNode,
  tEnterX: FloatNode, dt: FloatNode,
  col: Vec3Node, transmittance: FloatNode,
  uncertainty?: UncertaintyAccum,
  visibility?: FloatNode,
): void {
  // Change only optical depth during the reveal. Sample the same geographic
  // locations at every frame, with the same midpoint spacing on both walls.
  const opticalStep = visibility ? dt.mul(visibility).toVar() : dt
  // One scratch pair per segment, reused every step — see StepProbe's doc.
  const probe: StepProbe | undefined = uncertainty
    ? { cv: float(0).toVar(), land: float(0).toVar() }
    : undefined
  Loop({ start: 0, end: opts.nMax, type: 'int' }, ({ i }) => {
    const t = tEnterX.add(dt.mul(float(i).add(0.5)))
    const p = ro.add(rd.mul(t))

    // Reduced altitude in sphere space; error O(f*H) ~ 215 m in a 64 km
    // shell, below the profile's own resolution.
    const ps = vec3(p.x.mul(invA), p.y.mul(invA), p.z.mul(invB))
    const z = length(ps).sub(1.0).mul(A)

    // dir into the field sweeps across lat/lon step by step for an
    // oblique ray - only an exactly-nadir ray would hold it fixed - so
    // this must be resampled every iteration, not hoisted out of the loop.
    // Keep geographic lookup in the bake's world/geocentric frame. GPU
    // regression: experiments/shellGeographyProbe.ts uses a latitude ramp.
    const density = stepSigma0(opts, p, probe).mul(exp(z.div(opts.Hs).negate()))
    // Smoothstep taper over the last e-fold so the outer bound is C1
    // continuous - a hard truncation leaves a visible alpha ring.
    // WGSL marks smoothstep's low >= high case indeterminate, so the
    // decreasing ramp is built as 1 - smoothstep(low, high, z) with
    // low < high, rather than swapping the edges.
    const taper = float(1.0).sub(smoothstep(opts.Hmax.sub(opts.Hs), opts.Hmax, z))
    const sigma = density.mul(taper)

    const stepT = exp(sigma.mul(opticalStep).negate())
    const contribution = transmittance.mul(float(1.0).sub(stepT))
    col.addAssign(contribution)
    if (uncertainty && probe) {
      // Weighted by THIS step's contribution to the image — see
      // UncertaintyAccum's doc for why a flat per-step mean is wrong.
      uncertainty.cvWeighted.addAssign(contribution.mul(probe.cv))
      uncertainty.landWeighted.addAssign(contribution.mul(probe.land))
      uncertainty.weight.addAssign(contribution)
    }
    transmittance.mulAssign(stepT)

    If(transmittance.lessThan(0.003), () => { Break() })
  })
}

/**
 * Regime 1 shell raymarch.
 *
 * Bounded by a sphere mesh rendered BackSide, so positionWorld gives the far
 * intersection and cameraPosition the ray origin - VolumeNodeMaterial is
 * geometry-bounded, not box-bounded, so the outer bound costs no custom code.
 *
 * Every texture sample inside the loop uses explicit LOD: WGSL's
 * derivative_uniformity rule has default severity ERROR, and the transmittance
 * Break() below makes the control flow non-uniform. Spec section 10.
 *
 * TWO WALLS, ONE COLUMN (opts.farWall). A ray that crosses the inner
 * (datum) ellipsoid exits the annular shell there, crosses empty interior
 * space where solid Earth used to be, and re-enters the shell on the far
 * side of the planet before exiting the outer boundary - `raySpheroid.ts`'s
 * `intersectShell` already reports this as two segments (proven in
 * `raySpheroid.test.ts`'s "produces two segments for a ray through the
 * whole planet"), and `shellFarWall.test.ts` checks the far-wall algebra
 * below reproduces that exact ground truth before it ever reaches TSL.
 * `opts.farWall` absent, only the near wall (segment one) marches - spec
 * 7.3's original text ("let the terrain depth-clip terminate the march")
 * describes exactly that, written when a solid Earth was always present to
 * do the clipping. Once matter can fully dissolve (spec 6.1), that
 * assumption stops holding at high altitude, which is what `opts.farWall`
 * (gated per-frame - see `ShellFarWallOpts`'s doc) restores.
 */
export function buildShellMaterial(opts: ShellOpts): THREE.NodeMaterial {
  const material = new THREE.NodeMaterial()
  material.transparent = true
  material.depthWrite = false
  material.side = THREE.BackSide
  // The march accumulates col += T*(1-stepT), i.e. PREMULTIPLIED radiance.
  // NormalBlending multiplies by alpha a second time, so without this the
  // disc renders at alpha^2 (~0.002 instead of the correct ~0.046).
  material.premultipliedAlpha = true

  const march = Fn(() => {
    // The veil's own near-monochrome bioluminescent tint (spec 6.2). Named
    // rather than repeated inline now that the uncertainty overlay mixes
    // AWAY from it — the two uses have to be the same colour.
    const VEIL_TINT = vec3(0.72, 0.86, 0.95)
    // See ShellOpts.cameraWorldPosition's doc for why `ro` needs this
    // fallback (graph-construction-time, a plain JS `??`) but `rd` does not.
    const ro = opts.cameraWorldPosition ?? cameraPosition
    const rd = normalize(positionWorld.sub(cameraPosition))

    // Sphere space: S = diag(1/a, 1/a, 1/b). Spec section 7.1.
    const invA = float(1.0 / A)
    const invB = float(1.0 / B)
    const o = vec3(ro.x.mul(invA), ro.y.mul(invA), ro.z.mul(invB))
    const dRaw = vec3(rd.x.mul(invA), rd.y.mul(invA), rd.z.mul(invB))
    const dLen = length(dRaw)
    const d = dRaw.div(dLen)

    // RTG method 6: L2 is radius-independent and shared across both surfaces.
    const b2 = dot(o, d)
    const fd = o.sub(d.mul(b2))
    const L2 = dot(fd, fd)

    const kOuter = float(1.0).add(opts.Hmax.div(A))
    const discOut = kOuter.mul(kOuter).sub(L2)
    const discIn = float(1.0).sub(L2)

    const col = vec3(0.0).toVar()
    const transmittance = float(1.0).toVar()
    const uncertainty: UncertaintyAccum | undefined = opts.uncertainty
      ? { cvWeighted: float(0).toVar(), landWeighted: float(0).toVar(), weight: float(0).toVar() }
      : undefined

    If(discOut.greaterThan(0.0), () => {
      const sOut = sqrt(discOut)
      // Segment length DIRECTLY as 2s, never by differencing two large t values.
      const chord = sOut.mul(2.0).div(dLen).toVar()

      const cQuad = dot(o, o).sub(kOuter.mul(kOuter))
      const q = b2.greaterThanEqual(0.0).select(sOut.negate().sub(b2), sOut.sub(b2))
      const tEnter = max(min(cQuad.div(q), q), 0.0).div(dLen).toVar()

      // If the inner surface is hit, stop the NEAR-wall segment there. Both
      // near roots share the same -b2, camera-outside-the-shell form (true
      // throughout Session 1's descent range): t_outer_near = -b2 - sOut and
      // t_inner_near = -b2 - sIn, so b2 cancels EXACTLY in their difference,
      // leaving sOut - sIn - two O(1) sphere-space values. Difference those
      // directly and scale once, rather than scaling each to world-space t
      // (order 1e6-1e7 m, where f32 ULP is 1-4 m) and differencing there.
      //
      // tEnter2 is the FAR wall's entry point (world-space t, same units as
      // tEnter) - see ShellFarWallOpts's doc for the two conditions gating
      // it, and this function's own doc for what it physically represents.
      // Declared here (not inside the opts.farWall-only branch below) so it
      // stays in scope for the second marchSegment call after this If block
      // closes; when opts.farWall is absent this whole variable, like every
      // line below that references it, is never emitted at all.
      const tEnter2 = opts.farWall ? float(0.0).toVar() : undefined
      If(discIn.greaterThan(0.0), () => {
        const sIn = sqrt(discIn)
        chord.assign(max(sOut.sub(sIn), 0.0).div(dLen))

        if (opts.farWall && tEnter2) {
          If(opts.farWall.enabled, () => {
            // Same c/q robust-root method as tEnter above (spec 7.2's
            // method 6), mirrored onto the INNER sphere (r=1) and taking
            // the FAR (larger) root instead of the near one - exactly
            // `math/raySpheroid.ts`'s `intersectShell`'s `roots(1.0)` far
            // branch, TSL-ported and checked against it numerically in
            // `shellFarWall.test.ts` before this code was written.
            const cQuadIn = dot(o, o).sub(1.0)
            const qIn = b2.greaterThanEqual(0.0).select(sIn.negate().sub(b2), sIn.sub(b2))
            tEnter2.assign(max(cQuadIn.div(qIn), qIn).div(dLen))
          })
        }
      })

      const dt = chord.div(float(opts.nMax))

      marchSegment(opts, ro, rd, invA, invB, tEnter, dt, col, transmittance, uncertainty)

      if (opts.farWall && tEnter2) {
        // The far wall's chord is EXACTLY as long as the near wall's - both
        // equal (sOut-sIn)/dLen by symmetry of the annular shell about its
        // centre (proven in shellFarWall.test.ts) - so it reuses `dt`
        // rather than a second division. Gated on discIn>0 again (tEnter2
        // is only ever assigned inside that branch above) AND on
        // opts.farWall.enabled again (tEnter2 stays its 0 default whenever
        // the earlier If didn't fire, and marching from t=0 with
        // whatever transmittance the near wall left behind would render a
        // segment that was never actually there).
        If(discIn.greaterThan(0.0).and(opts.farWall.enabled), () => {
          marchSegment(opts, ro, rd, invA, invB, tEnter2, dt, col, transmittance, uncertainty, opts.farWall?.visibility)
        })
      }

      // PLACEHOLDER antialias, not the finished thing: this smoothstep covers
      // only ~4e-5 of a pixel at typical framing, nowhere near a full pixel
      // footprint. Method 6 fixes the discriminant, NOT the tangent
      // singularity, so an edge-taper term of some kind is a requirement -
      // but proper pixel-footprint normalization (spec §7.2: step width
      // derived from screen-space derivatives, not this fixed 1e-7 constant)
      // is still owed and is Session 2 work, not done here. Applied once,
      // after BOTH walls have accumulated into col - it is a silhouette
      // antialias on the shared outer boundary, not a per-wall effect.
      const edge = smoothstep(0.0, float(1e-7), discOut)
      col.mulAssign(edge)
    })

    const alpha = float(1.0).sub(transmittance)

    // Output dither — the SECOND, independent defect behind "blocky", and
    // not one the reconstruction filter above touches. The veil's whole
    // on-screen range spans only about a dozen 8-bit codes (ACES filmic into
    // an 8-bit canvas, no dithering anywhere in the chain), so without this
    // those codes render as wide flat plateaus with hard contours between
    // them — terraced rather than faceted, and still there after the filter
    // fix. Fixing one and not the other leaves the complaint standing.
    //
    // ONE scalar for the whole vec4: `col` and `alpha` are algebraically the
    // same accumulated quantity here (`col += T*(1-stepT)` versus
    // `alpha = 1 - prod(stepT)`), the material is premultipliedAlpha, and a
    // single shared factor keeps the premultiplication exact while dithering
    // luminance only — per-channel noise would dither the HUE of a
    // deliberately near-monochrome veil. RELATIVE, so it contributes
    // identically nothing wherever the accumulated column is zero (no noise
    // sprayed across empty sky) and so it survives the 1x-64x exposure
    // control the shader cannot see. See
    // field/terrain.ts::VEIL_DITHER_AMPLITUDE for how 0.02 was derived
    // against a model of this exact tone-map/encode/quantise chain.
    //
    // UNCONDITIONAL, unlike every other option on ShellOpts: banding is a
    // property of the veil's output, not of which density source produced
    // it, so all three configs (uniform / analytic field / cube) need it.
    // That does cost the uniform config its byte-for-byte identity with
    // Session 1's baseline shader — see ShellOpts.field's doc, amended.
    const dither = veilDitherScaleNode()

    if (opts.uncertainty && uncertainty) {
      // Resolve the column's contribution-weighted averages. The guard
      // matters: a column with no density at all accumulates weight 0, and
      // an unguarded divide gives NaN, which propagates through the mix and
      // paints the fragment black or transparent depending on the vendor.
      const w = max(uncertainty.weight, float(1e-8))
      const cv = uncertainty.cvWeighted.div(w)
      const land = uncertainty.landWeighted.div(w)
      const strength = clamp(opts.uncertainty.strength, 0, 1)
      // Coverage (and therefore the ocean exclusion) lives in
      // uncertaintyOverlayColorNode, driven by LAND FRACTION — never by cv,
      // which reads 0 over ocean and would paint it as the most certain
      // surface on Earth. See field/terrain.ts's uncertaintyCoverage.
      const coverage = uncertaintyCoverageNode(land)
      const paint = strength.mul(coverage)
      const tint = uncertaintyOverlayColorNode(VEIL_TINT, cv, land, strength)
      // See UNCERTAINTY_MIN_LUMA: CV is anti-correlated with density here,
      // so covered land needs a floor or the Sahara — the clearest thing
      // this channel has to show — is the dimmest thing on screen.
      const lit = max(col.x, float(UNCERTAINTY_MIN_LUMA).mul(paint))
      return vec4(tint.mul(lit), max(alpha, float(UNCERTAINTY_MIN_LUMA).mul(paint))).mul(dither)
    }
    return vec4(col.mul(VEIL_TINT), alpha).mul(dither)
  })

  material.fragmentNode = march()
  return material
}

/**
 * Compiles `material` into a throwaway scene and pulls back the generated
 * WGSL fragment source for inspection.
 *
 * The brief's original approach (`renderer.backend.get(material)?.pipeline
 * ?.fragmentProgram?.code`) does not work: `Backend` (node_modules/three/src/
 * renderers/common/Backend.js) has no `get()`-keyed data for a `Material` at
 * all - its `DataMap` is keyed by GPU-pipeline/render-context objects, and
 * `@types/three`'s `Backend.d.ts` doesn't even declare a `get()` method, so
 * that line does not typecheck.
 *
 * The WGSL text actually lives on the renderer's Pipelines component:
 * `Renderer.compileAsync` (node_modules/three/src/renderers/common/
 * Renderer.js:3785) calls `this._pipelines.getForRender(renderObject, ...)`,
 * which (Pipelines.js:198-207) wraps the generated fragment-shader source in
 * a `ProgrammableStage` and stores it in `this.programs.fragment`, a
 * `Map<code, ProgrammableStage>`. `ProgrammableStage.code`
 * (ProgrammableStage.js:21,35) is exactly that WGSL string. `_pipelines` is
 * not part of the public `Renderer` type - no `.d.ts` declares it - so this
 * is the one narrow, isolated cast in this module needed to reach it.
 */
export async function dumpShellWGSL(
  renderer: THREE.WebGPURenderer,
  material: THREE.NodeMaterial,
): Promise<string> {
  const scene = new THREE.Scene()
  const geo = new THREE.SphereGeometry(A * 1.03, 64, 32)
  const mesh = new THREE.Mesh(geo, material)
  scene.add(mesh)
  const cam = new THREE.PerspectiveCamera(50, 1, 1, 1e9)
  cam.position.set(0, 0, A * 4)
  cam.lookAt(0, 0, 0)

  try {
    await renderer.compileAsync(scene, cam)

    const pipelines = (renderer as unknown as {
      _pipelines?: { programs?: { fragment?: Map<string, { code?: string }> } }
    })._pipelines

    const stages = pipelines?.programs?.fragment
    if (!stages || stages.size === 0) return ''

    const last = [...stages.values()].at(-1)
    return last?.code ?? ''
  } finally {
    scene.remove(mesh)
    geo.dispose()
  }
}
