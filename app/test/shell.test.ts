import { describe, it, expect } from 'vitest'
import { uniform } from 'three/tsl'
import { buildShellMaterial, type ShellOpts } from '../src/regimes/Shell'
import { bakeSyntheticCubeTexture } from '../src/field/bakeCube'
import { intersectShell, type Vec3 } from '../src/math/raySpheroid'
import { WGS84 } from '../src/math/wgs84'
import shellSource from '../src/regimes/Shell.ts?raw'

// TSL node functions cannot be numerically executed in this environment (no
// WebGPU - see webgpu.test.ts/isWebGPUAvailable), so - same as
// matter.test.ts's approach for the same reason - the veil's actual
// per-fragment arithmetic is pinned with structural regex checks against the
// generated source rather than numeric assertions. Each check below is
// written to fail against the PRE-fix source (verified by hand against the
// version at commit 5c4dd7c, which reads only `.r` off the density cube and
// never touches `.a` anywhere in the file).

function fakeCubeOpts(): ShellOpts {
  const fakeCube = bakeSyntheticCubeTexture({ size: 2 })
  return {
    Hs: uniform(32000),
    Hmax: uniform(192000),
    sigma0: uniform(1e-6),
    cube: { texture: fakeCube, sigma0PerTau: uniform(1e-6), tauCap: uniform(1) },
    nMax: 4,
  }
}

describe('buildShellMaterial, cube branch', () => {
  it('builds without throwing when the cube opts read both R and A channels', () => {
    // bakeSyntheticCubeTexture is single-channel (RedFormat) - graph
    // construction never touches real texel bytes, only builds a node
    // graph, so swizzling `.a` off it is exactly as valid at BUILD time as
    // swizzling `.r` (matter.test.ts's fakeOpts relies on the same fact for
    // its own `.a` reads of a synthetic cube). This is a weak pin on its
    // own - see the regex checks below for what actually distinguishes
    // fixed from unfixed - but catches a construction-time crash.
    const material = buildShellMaterial(fakeCubeOpts())
    expect(material.fragmentNode).toBeDefined()
  })
})

describe('the ocean no-data fix: the cube branch reads the A channel, not just R', () => {
  it('samples the A (land fraction) channel off the density cube', () => {
    // Old source has no `.a` anywhere in the file at all - grep confirmed
    // it on the pre-fix version. Any read of the density cube's alpha
    // channel is new. Both R and A come off the same `cubeTexture(...)`
    // sample rather than two separate fetches.
    // `dir`, not `ps`: the direction now goes through
    // smoothstepCubeDirectionNode first (the reconstruction-filter fix — see
    // test/cubeFilter.test.ts). That changes WHERE the sample is taken, not
    // that R and A come off one fetch, which is what this test is about.
    expect(shellSource).toMatch(/const cubeSample = cubeTexture\(texture, dir\)\.level\(float\(0\)\)/)
    expect(shellSource).toMatch(/filteredTauNode\(texture, cubeSample\.r\)/)
    expect(shellSource).toMatch(/cubeSample\.a\b/)
  })

  it('clamps the land-fraction read to [0,1], matching every other landFraction use in this codebase', () => {
    expect(shellSource).toMatch(/landFraction = clamp\(filteredSupportNode\(texture, cubeSample\.a\), 0, 1\)/)
  })

  it('does not add atlas coverage to the analytic-field branch', () => {
    // Geographic frame changed; an analytic field still has no A channel.
    expect(shellSource).toMatch(
      /const tauSampled = applyCobbMix\(syntheticTauNode\(worldDirection\), worldDirection, opts\.urban\)\n(?!.*landFraction)/,
    )
  })
})

// The old source-text assertion pinned a second coverage multiplication,
// which was itself wrong. densityCoverage.test.ts now tests the numeric
// contract; soilCoverageProbe.ts exercises the real shell with GPU readback.

// Seeing through to the far side: a ray through the whole planet crosses the
// annular shell TWICE (raySpheroid.ts's intersectShell, already proven in
// raySpheroid.test.ts) but the march used to stop at the near wall
// unconditionally - correct while matter (the opaque terrain) always
// existed to occlude the far wall, wrong once matter can fully dissolve.
// Same testing ceiling as above (no WebGPU here): structural regex against
// the generated source, each checked to fail on the pre-far-wall commit
// (06faab9) by stashing regimes/Shell.ts and rerunning.
function fakeFarWallCubeOpts(): ShellOpts {
  return { ...fakeCubeOpts(), farWall: { enabled: uniform(true) } }
}

describe('buildShellMaterial, far-wall branch', () => {
  it('builds without throwing with opts.farWall wired', () => {
    const material = buildShellMaterial(fakeFarWallCubeOpts())
    expect(material.fragmentNode).toBeDefined()
  })

  it('still builds without opts.farWall - the feature is additive, not required', () => {
    const material = buildShellMaterial(fakeCubeOpts())
    expect(material.fragmentNode).toBeDefined()
  })
})

describe('the far wall: marches a second shell segment through the planet interior', () => {
  it('factors the per-step march into a shared marchSegment, called for both walls', () => {
    // Old source has exactly one Loop inlined directly in buildShellMaterial
    // and no marchSegment function at all - this would not match against it.
    expect(shellSource).toMatch(/^function marchSegment\(/m)
    // The trailing `uncertainty` arg is the overlay's accumulator (added
    // with the `u` overlay); both walls must still route through the one
    // shared marchSegment, which is what this test is actually about.
    expect(shellSource).toMatch(/marchSegment\(opts, ro, rd, invA, invB, tEnter, dt, col, transmittance, uncertainty\)/)
    expect(shellSource).toMatch(/marchSegment\(opts, ro, rd, invA, invB, tEnter2, dt, col, transmittance, uncertainty, opts\.farWall\?\.visibility\)/)
  })

  it('declares ShellFarWallOpts with a per-frame enabled gate', () => {
    expect(shellSource).toMatch(/export interface ShellFarWallOpts \{/)
    expect(shellSource).toMatch(/enabled: THREE\.UniformNode<'bool', boolean>/)
    expect(shellSource).toMatch(/farWall\?: ShellFarWallOpts/)
  })

  it('computes the far-wall entry point via the inner sphere\'s FAR root, not the near one', () => {
    // tEnter (near, existing) takes the min of the two c/q roots, clamped to
    // 0. tEnter2 (far, new) must take the MAX of the analogous inner-sphere
    // pair, with no clamp-to-0 (the far root is never behind the camera
    // whenever the near one already wasn't).
    expect(shellSource).toMatch(/const cQuadIn = dot\(o, o\)\.sub\(1\.0\)/)
    expect(shellSource).toMatch(/const qIn = b2\.greaterThanEqual\(0\.0\)\.select\(sIn\.negate\(\)\.sub\(b2\), sIn\.sub\(b2\)\)/)
    expect(shellSource).toMatch(/tEnter2\.assign\(max\(cQuadIn\.div\(qIn\), qIn\)\.div\(dLen\)\)/)
  })

  it('reuses dt for the far wall rather than a second chord/nMax division', () => {
    // Only ONE `chord.div(float(opts.nMax))` should exist in the whole file -
    // a second, independently-computed far-wall dt would be the wrong
    // (needlessly duplicated, and easy to accidentally desync) approach.
    const dtDivisions = shellSource.match(/\.div\(float\(opts\.nMax\)\)/g) ?? []
    expect(dtDivisions).toHaveLength(1)
  })

  it('gates the far-wall march on both discIn>0 and opts.farWall.enabled, not either alone', () => {
    expect(shellSource).toMatch(/If\(discIn\.greaterThan\(0\.0\)\.and\(opts\.farWall\.enabled\)/)
  })

  it('applies the silhouette edge antialias once, after both walls have accumulated into col', () => {
    // The far-wall march call and its guard must appear BEFORE the `edge`
    // line in source order - otherwise the far wall's contribution would
    // miss the antialias entirely.
    const farWallCallIdx = shellSource.indexOf('marchSegment(opts, ro, rd, invA, invB, tEnter2,')
    const edgeIdx = shellSource.indexOf('const edge = smoothstep(0.0, float(1e-7), discOut)')
    expect(farWallCallIdx).toBeGreaterThan(0)
    expect(edgeIdx).toBeGreaterThan(farWallCallIdx)
  })
})

describe('the far-wall entry formula matches intersectShell exactly (not just "looks similar")', () => {
  // This does not execute Shell.ts's TSL (impossible without WebGPU) - it
  // re-derives the SAME formula the regex checks above pin, in plain JS, and
  // checks it against intersectShell's tested output. Together with
  // shellFarWall.test.ts (which established this formula BEFORE it was
  // transcribed into TSL), this is the strongest correctness evidence
  // available in this environment for a piece of math that cannot be run
  // where it actually lives.
  const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const toSphere = (v: Vec3): Vec3 => [v[0] / WGS84.a, v[1] / WGS84.a, v[2] / WGS84.b]
  const H = 192000
  const K_OUTER = 1 + H / WGS84.a

  it('reproduces intersectShell.segments[1][0] for a through-the-planet ray', () => {
    const origin: Vec3 = [WGS84.a + 1e6, 0, 0]
    const dir: Vec3 = [-1, 0, 0]

    const o = toSphere(origin)
    const dRaw = toSphere(dir)
    const dLen = Math.hypot(dRaw[0], dRaw[1], dRaw[2])
    const d: Vec3 = [dRaw[0] / dLen, dRaw[1] / dLen, dRaw[2] / dLen]
    const b2 = dot(o, d)
    const fd: Vec3 = [o[0] - b2 * d[0], o[1] - b2 * d[1], o[2] - b2 * d[2]]
    const L2 = dot(fd, fd)
    const sIn = Math.sqrt(1 - L2)
    const cQuadIn = dot(o, o) - 1.0
    const qIn = b2 >= 0.0 ? -sIn - b2 : sIn - b2
    const tEnter2 = Math.max(cQuadIn / qIn, qIn) / dLen

    const r = intersectShell(origin, dir, K_OUTER)
    expect(r.segments).toHaveLength(2)
    expect(tEnter2).toBeCloseTo(r.segments[1]![0], 6)
  })
})

// --- The uncertainty overlay (`u`) -----------------------------------------
//
// Channel B of the density cube is the per-pixel bootstrap coefficient of
// variation. It was baked, verified, and read by nothing until this option
// existed. Same testing ceiling as everything else in this file (no WebGPU
// here): structural pins against the generated source, each written to fail
// against the pre-overlay version, which contains no `.b` read anywhere.
function fakeUncertaintyCubeOpts(): ShellOpts {
  return { ...fakeCubeOpts(), uncertainty: { strength: uniform(1) } }
}

describe('buildShellMaterial, uncertainty branch', () => {
  it('builds without throwing with opts.uncertainty wired', () => {
    const material = buildShellMaterial(fakeUncertaintyCubeOpts())
    expect(material.fragmentNode).toBeDefined()
  })

  it('still builds without opts.uncertainty — the feature is additive, not required', () => {
    expect(buildShellMaterial(fakeCubeOpts()).fragmentNode).toBeDefined()
  })

  it('declares ShellUncertaintyOpts with a per-frame strength uniform, not a rebuild flag', () => {
    // `u` has to be a live toggle. A boolean that forced a material rebuild
    // would drop a frame and re-upload the atlas on every press.
    expect(shellSource).toMatch(/export interface ShellUncertaintyOpts \{/)
    expect(shellSource).toMatch(/strength: THREE\.UniformNode<'float', number>/)
    expect(shellSource).toMatch(/uncertainty\?: ShellUncertaintyOpts/)
  })
})

describe('the overlay reads the CV channel, and reads it off the existing fetch', () => {
  it('samples the B (bootstrap CV) channel of the density cube', () => {
    // The pre-overlay source has no `.b` read at all — the channel was
    // baked and unused. Any read of it is new.
    expect(shellSource).toMatch(/cubeSample\.b\b/)
  })

  it('takes CV off the SAME cubeSample as R and A, not a second texture fetch', () => {
    // A separate cubeTexture(...) call for B would double the per-step
    // sampling cost of the hottest loop in the piece for no reason — R, A
    // and B all arrive in one RGBA16Float fetch.
    // `dir` is the smoothstep-warped `ps` (see the R/A test above).
    const cubeFetches = shellSource.match(/cubeTexture\(texture, dir\)/g) ?? []
    expect(cubeFetches).toHaveLength(1)
  })

  it('emits no CV code at all when the option is absent', () => {
    // Same build-time discipline as field/cube/partition/farWall/urban: the
    // absent option must not leave a runtime branch behind in the baseline
    // shader. Every B-channel read sits behind a plain-JS `if`.
    expect(shellSource).toMatch(/if \(probe\) \{/)
    expect(shellSource).toMatch(/if \(opts\.uncertainty\b/)
    expect(shellSource).toMatch(/if \(uncertainty && probe\) \{/)
  })
})

describe('the overlay averages CV over the column that actually contributed light', () => {
  it('weights CV by each step\'s own contribution, not by a flat per-step count', () => {
    // A flat mean over marched steps would average in every empty step
    // above and below the density, dragging the whole planet toward
    // whatever CV the thin upper atmosphere happens to carry. The weight
    // must be the same `contribution` term that accumulates into col.
    expect(shellSource).toMatch(/const contribution = transmittance\.mul\(float\(1\.0\)\.sub\(stepT\)\)/)
    expect(shellSource).toMatch(/col\.addAssign\(contribution\)/)
    expect(shellSource).toMatch(/cvWeighted\.addAssign\(contribution\.mul\(probe\.cv\)\)/)
    expect(shellSource).toMatch(/weight\.addAssign\(contribution\)/)
  })

  it('normalises by the accumulated weight, guarded against an empty column', () => {
    // Over a column with no density at all the weight is 0, and an
    // unguarded divide yields NaN — which propagates through the mix and
    // paints the fragment black or transparent depending on the vendor.
    expect(shellSource).toMatch(/max\(uncertainty\.weight, float\(1e-8\)\)/)
    expect(shellSource).toMatch(/cvWeighted\.div\(w\)/)
    expect(shellSource).toMatch(/landWeighted\.div\(w\)/)
  })

  it('accumulates across BOTH walls, so the far side is not silently excluded', () => {
    const calls = shellSource.match(/marchSegment\(opts, ro, rd, invA, invB, \w+, dt, col, transmittance, uncertainty(?:, opts\.farWall\?\.visibility)?\)/g) ?? []
    expect(calls).toHaveLength(2)
  })
})

describe('the overlay delegates its colour to the tested terrain.ts/terrainNodes.ts pair', () => {
  it('uses uncertaintyOverlayColorNode rather than an inline ramp of its own', () => {
    // The colormap and the ocean exclusion are tested in terrain.test.ts
    // against CPU truth. A second, inline copy here would be untested.
    expect(shellSource).toMatch(/import \{[^}]*uncertaintyOverlayColorNode/s)
    expect(shellSource).toMatch(/uncertaintyOverlayColorNode\(/)
    expect(shellSource).not.toMatch(/UNCERTAINTY_STOPS/)
  })

  it('passes LAND FRACTION as the coverage argument, never the CV', () => {
    // THE bug the whole feature exists to avoid. Over ocean the CV channel
    // is 0 — indistinguishable from "extremely certain" — so coverage must
    // come from A. Pinned at the one call site where the choice is made.
    expect(shellSource).toMatch(/uncertaintyOverlayColorNode\(\s*VEIL_TINT,\s*cv,\s*land,\s*strength,?\s*\)/)
  })

  it('gives covered land a luminance floor, so the arid interiors do not vanish', () => {
    // CV is anti-correlated with density in this dataset: the Sahara is
    // both the sparsest and among the least certain places on the map. With
    // the veil's own brightness as the only light source, the single
    // clearest thing this channel has to show would be the dimmest thing on
    // screen. The floor is gated by `paint`, so it never touches ocean and
    // never touches anything while the overlay is off.
    expect(shellSource).toMatch(/const paint = strength\.mul\(coverage\)/)
    expect(shellSource).toMatch(/max\(col\.x, float\(UNCERTAINTY_MIN_LUMA\)\.mul\(paint\)\)/)
  })
})
