import { describe, it, expect } from 'vitest'
import {
  cubeFaceSize, directionToFaceUV, smoothstepCubeDirection, smoothstepTexelWarp,
} from '../src/field/loadCube'
import { bakeSyntheticCubeTexture, faceDirection } from '../src/field/bakeCube'
import { syntheticTau } from '../src/field/synthetic'
// Vite's `?raw` suffix — the test/dissolveNodes.test.ts precedent.
import shellSource from '../src/regimes/Shell.ts?raw'
import terrainNodesSource from '../src/field/terrainNodes.ts?raw'

/**
 * The instrument for the veil's "blocky" defect.
 *
 * The defect is NOT the atlas's resolution. `cube_plain_128` is 78.2
 * km/texel and is baked from a 512x256 equirect that is itself ~78 km/cell,
 * so rebaking recovers nothing (measured: 5.80% -> 5.58% -> 5.56% at
 * 256/512/1024, saturating, while the same instrument on the terrain
 * atlases grows monotonically). What the viewer actually sees at ~31 screen
 * pixels per texel is the RECONSTRUCTION FILTER, and hardware bilinear is
 * C0: linear inside each texel cell, with a gradient discontinuity across
 * every cell boundary. Every crease in the magnified field therefore falls
 * exactly on the texel grid, which is the polygonal faceting on screen.
 *
 * So the property measured below is that one: the share of the magnified
 * field's second-derivative mass that lands on texel-boundary lines. A
 * filter with nothing wrong with it spreads that mass roughly uniformly, so
 * the share sits near the fraction of pixels those lines occupy; bilinear
 * puts all of it there.
 *
 * Three guards keep this from being another instrument that returns the
 * same answer whether the system is healthy or broken:
 *
 *  - a CONTROL that renders the underlying continuous field with no texel
 *    structure at all, which must report the uniform baseline. If the metric
 *    were simply always high, this is where it would show.
 *  - the bilinear arm, which must report ~100%. That is the pre-fix state,
 *    and the test would fail if the metric could not see it.
 *  - both arms share every line of code except the one filter call.
 */

// ---------------------------------------------------------------------------
// A CPU model of the hardware's cube-face sampling.
//
// Texel `j` of an N-texel face axis is centred at face coordinate
// c = 2*(j+0.5)/N - 1 — the standard cube-map convention, and the same one
// `bakeCube.ts` writes with (`u = ((px + 0.5) / size) * 2 - 1`). Texel-space
// coordinate `p` places that centre at `p = j`.
// ---------------------------------------------------------------------------

const toTexelSpace = (c: number, n: number): number => ((c + 1) * 0.5) * n - 0.5
const fromTexelSpace = (p: number, n: number): number => ((p + 0.5) / n) * 2 - 1

function bilinear(grid: readonly number[], n: number, px: number, py: number): number {
  const at = (a: number, b: number): number => {
    const ca = Math.min(n - 1, Math.max(0, a))
    const cb = Math.min(n - 1, Math.max(0, b))
    return grid[cb * n + ca]!
  }
  const i = Math.floor(px)
  const j = Math.floor(py)
  const fx = px - i
  const fy = py - j
  return (
    at(i, j) * (1 - fx) * (1 - fy) + at(i + 1, j) * fx * (1 - fy)
    + at(i, j + 1) * (1 - fx) * fy + at(i + 1, j + 1) * fx * fy
  )
}

/** The field the grid below is a discretisation of — the project's own analytic stand-in for the baked atlas (`field/synthetic.ts`). */
const trueField = (face: number, u: number, v: number): number => {
  const [x, y, z] = faceDirection(face, u, v)
  return syntheticTau(x, y, z)
}

const FACE = 0
const N = 24 // texels along the modelled face patch
const PIXELS_PER_TEXEL = 31 // the veil's measured on-screen magnification

const grid: number[] = []
for (let j = 0; j < N; j++) {
  for (let i = 0; i < N; i++) {
    grid.push(trueField(FACE, ((i + 0.5) / N) * 2 - 1, ((j + 0.5) / N) * 2 - 1))
  }
}

// Render a patch well inside the face, so nothing here is measuring the
// edge clamp instead of the filter.
const FIRST_TEXEL = 2
const WIDTH = (N - 2 * FIRST_TEXEL) * PIXELS_PER_TEXEL
/** Texel-space coordinate of pixel `p`'s centre. */
const pixelToTexel = (p: number): number => FIRST_TEXEL + (p + 0.5) / PIXELS_PER_TEXEL

type Sampler = (px: number, py: number) => number

function render(sample: Sampler): Float64Array {
  const img = new Float64Array(WIDTH * WIDTH)
  for (let py = 0; py < WIDTH; py++) {
    for (let px = 0; px < WIDTH; px++) {
      img[py * WIDTH + px] = sample(pixelToTexel(px), pixelToTexel(py))
    }
  }
  return img
}

/**
 * Does the 3-pixel second-difference stencil centred on pixel `p` straddle a
 * texel-cell boundary? Cell boundaries — where a bilinear patch changes, and
 * so where its gradient may jump — sit at INTEGER texel-space coordinates.
 * Two pixels per boundary qualify, so the uniform baseline for this metric
 * is 2/31 = 6.45% rather than the 1/31 = 3.2% a one-pixel-wide line would
 * give; `reports the uniform baseline on a texel-free field` pins it.
 */
const onBoundary = (p: number): boolean =>
  Math.floor(pixelToTexel(p - 1)) !== Math.floor(pixelToTexel(p + 1))

interface CreaseMetric {
  /** Share of total |d2f| mass falling on texel-boundary lines. */
  share: number
  /** Share of the stencil population those lines are — what `share` means "uniform" against. */
  baseline: number
  peakOverMean: number
}

function creaseMass(img: Float64Array): CreaseMetric {
  const boundary: boolean[] = []
  for (let p = 0; p < WIDTH; p++) boundary.push(onBoundary(p))

  let total = 0
  let onLine = 0
  let peak = 0
  let stencils = 0
  let boundaryStencils = 0
  for (let py = 1; py < WIDTH - 1; py++) {
    for (let px = 1; px < WIDTH - 1; px++) {
      const c = img[py * WIDTH + px]!
      const d2x = Math.abs(img[py * WIDTH + px + 1]! - 2 * c + img[py * WIDTH + px - 1]!)
      const d2y = Math.abs(img[(py + 1) * WIDTH + px]! - 2 * c + img[(py - 1) * WIDTH + px]!)
      total += d2x + d2y
      stencils += 2
      if (boundary[px]) { onLine += d2x; boundaryStencils++ }
      if (boundary[py]) { onLine += d2y; boundaryStencils++ }
      if (d2x + d2y > peak) peak = d2x + d2y
    }
  }
  return {
    share: onLine / total,
    baseline: boundaryStencils / stencils,
    peakOverMean: peak / (total / (stencils / 2)),
  }
}

const bilinearArm = creaseMass(render((px, py) => bilinear(grid, N, px, py)))
const warp = (p: number): number => toTexelSpace(smoothstepTexelWarp(fromTexelSpace(p, N), N), N)
const warpedArm = creaseMass(render((px, py) => bilinear(grid, N, warp(px), warp(py))))
const controlArm = creaseMass(render((px, py) => trueField(
  FACE, fromTexelSpace(px, N), fromTexelSpace(py, N),
)))

describe('crease mass on texel boundaries — the defect the veil actually showed', () => {
  it('records what the three arms measure, so the numbers live with the code', () => {
    // bilinear 100.0% / smoothstep 10.8% / continuous control 6.1%, against
    // a 6.1% uniform baseline. The control landing on the baseline to within
    // 1.5% relative is the evidence that a high reading means something.
    expect(bilinearArm.share).toBeCloseTo(1.0, 6)
    expect(warpedArm.share).toBeCloseTo(0.108, 2)
    expect(controlArm.share).toBeCloseTo(0.061, 2)
    expect(bilinearArm.baseline).toBeCloseTo(0.0615, 3)
    expect(bilinearArm.peakOverMean).toBeCloseTo(74.5, 0)
    expect(warpedArm.peakOverMean).toBeCloseTo(9.0, 0)
  })

  it('reports the uniform baseline on a texel-free field (the instrument is not stuck high)', () => {
    // The continuous field, evaluated per PIXEL: no reconstruction filter is
    // involved at all, so there is nothing to prefer the texel grid and the
    // metric must land on its own uniform baseline.
    expect(controlArm.share).toBeGreaterThan(controlArm.baseline * 0.7)
    expect(controlArm.share).toBeLessThan(controlArm.baseline * 1.4)
  })

  it('puts essentially ALL of it on the texel grid under plain bilinear — the pre-fix state', () => {
    // Bilinear is linear in x for fixed y (and vice versa) throughout a
    // cell, so the axis-aligned second difference is exactly zero in every
    // cell interior and the entire mass collects on the cell boundaries.
    expect(bilinearArm.share).toBeGreaterThan(0.99)
    // The same fact seen as contrast: the crease pixels tower over the mean.
    expect(bilinearArm.peakOverMean).toBeGreaterThan(50)
  })

  it('spreads it off the grid once the texel coordinate is smoothstep-warped', () => {
    // This is the assertion that distinguishes the fix from its absence: it
    // fails outright (0.99+) against the bilinear arm above.
    expect(warpedArm.share).toBeLessThan(0.20)
    // and it is a real improvement, not merely "below a threshold".
    expect(warpedArm.share).toBeLessThan(bilinearArm.share / 5)
    expect(warpedArm.peakOverMean).toBeLessThan(bilinearArm.peakOverMean / 5)
  })

  it('leaves the baseline identical between arms — only the filter differs', () => {
    expect(warpedArm.baseline).toBe(bilinearArm.baseline)
    expect(controlArm.baseline).toBe(bilinearArm.baseline)
  })
})

describe('smoothstepTexelWarp', () => {
  it('is exactly the texel-space map floor(p) + smoothstep(frac(p))', () => {
    const n = 128
    for (const c of [-0.9, -0.31, -0.004, 0.0, 0.11, 0.5, 0.777, 0.99]) {
      const p = toTexelSpace(c, n)
      const f = p - Math.floor(p)
      const expected = Math.floor(p) + f * f * (3 - 2 * f)
      expect(toTexelSpace(smoothstepTexelWarp(c, n), n)).toBeCloseTo(expected, 10)
    }
  })

  it('fixes every texel centre, so the reconstruction still passes through the baked values', () => {
    const n = 16
    for (let j = 0; j < n; j++) {
      const centre = ((j + 0.5) / n) * 2 - 1
      expect(smoothstepTexelWarp(centre, n)).toBeCloseTo(centre, 12)
    }
  })

  it('is ODD — which is what lets a per-component warp stand in for a per-face one', () => {
    // W(-c) = -W(c) follows from smoothstep(1-f) = 1 - smoothstep(f) plus
    // the mirror symmetry of the texel grid. It is the property that makes
    // the warp commute with the sign flips and axis permutations different
    // cube-map conventions apply (three.js negates a component of its own
    // in CubeTextureNode.js), so no face table is needed anywhere.
    for (const n of [2, 8, 128, 1024]) {
      for (const c of [0.03, 0.25, 0.5, 0.61, 0.9, 1.0]) {
        expect(smoothstepTexelWarp(-c, n)).toBeCloseTo(-smoothstepTexelWarp(c, n), 12)
      }
    }
  })

  it('is monotone and never leaves [-1, 1]', () => {
    const n = 32
    let prev = -Infinity
    for (let k = 0; k <= 4000; k++) {
      const c = -1 + (2 * k) / 4000
      const w = smoothstepTexelWarp(c, n)
      expect(w).toBeGreaterThanOrEqual(-1 - 1e-12)
      expect(w).toBeLessThanOrEqual(1 + 1e-12)
      expect(w).toBeGreaterThanOrEqual(prev - 1e-12)
      prev = w
    }
  })
})

describe('face seams — the failure a cube-map warp had to avoid', () => {
  // A visible seam would be a worse defect than the faceting being fixed, so
  // these are the load-bearing tests of the whole change.

  it('fixes the face edge EXACTLY: W(+/-1) = +/-1', () => {
    // c = 1 gives p = N - 0.5, whose fractional part is 0.5 — and 0.5 is a
    // fixed point of smoothstep. Not approximately: the face edge is
    // untouched, so whatever the sampler does where faces meet, it does it
    // to the same coordinate as before this change.
    for (const n of [2, 4, 128, 1024, 2048]) {
      expect(smoothstepTexelWarp(1, n)).toBeCloseTo(1, 15)
      expect(smoothstepTexelWarp(-1, n)).toBeCloseTo(-1, 15)
    }
  })

  it('never moves a direction onto a different face', () => {
    // The major component of dir/max|dir| is exactly +/-1 and so is fixed,
    // while every other component stays within [-1, 1]; the major axis
    // cannot change. Checked against the pipeline's own face assignment.
    const dirs: [number, number, number][] = []
    for (let a = 0; a < 40; a++) {
      for (let b = 0; b < 40; b++) {
        const theta = (a + 0.5) / 40 * Math.PI
        const phi = (b + 0.5) / 40 * 2 * Math.PI
        dirs.push([
          Math.sin(theta) * Math.cos(phi),
          Math.sin(theta) * Math.sin(phi),
          Math.cos(theta),
        ])
      }
    }
    for (const [x, y, z] of dirs) {
      const before = directionToFaceUV(x, y, z).face
      const [wx, wy, wz] = smoothstepCubeDirection(x, y, z, 128)
      expect(directionToFaceUV(wx, wy, wz).face).toBe(before)
    }
  })

  it('maps a direction that lies ON a seam to a direction still exactly on that seam', () => {
    // Two components tied at magnitude 1 both being fixed points is what
    // makes the seam SET invariant — the in-seam coordinate is warped, but
    // identically as approached from either of the two faces sharing it.
    const t = 0.37
    const seamDirs: [number, number, number][] = [
      [1, t, 1], [1, t, -1], [-1, t, 1], [1, 1, t], [t, 1, 1], [t, -1, 1], [1, -1, -t],
    ]
    for (const [x, y, z] of seamDirs) {
      const w = smoothstepCubeDirection(x, y, z, 128)
      const m = Math.max(Math.abs(w[0]), Math.abs(w[1]), Math.abs(w[2]))
      // exactly as many components tied at the maximum as before, same signs
      const tiedBefore = [x, y, z].filter((c) => Math.abs(c) === 1)
      const tiedAfter = w.filter((c) => Math.abs(Math.abs(c) - m) < 1e-12)
      expect(tiedAfter.length).toBe(tiedBefore.length)
      expect(m).toBeCloseTo(1, 12)
    }
  })

  it('is continuous ACROSS a seam — no jump in the sampled face coordinate', () => {
    // Sweep through the +X/+Z seam (x = z, both positive) and check the
    // warped direction has no discontinuity. If the warp were derived
    // per-face without the odd-symmetry argument, this is where it would
    // tear.
    const eps = 1e-6
    let maxJump = 0
    for (let k = 0; k <= 200; k++) {
      const y = -0.9 + (1.8 * k) / 200
      const before = smoothstepCubeDirection(1, y, 1 - eps, 128)
      const after = smoothstepCubeDirection(1, y, 1 + eps, 128)
      // normalise both onto the same scale before comparing directions
      const unit = (v: readonly [number, number, number]): number[] => {
        const n = Math.hypot(v[0], v[1], v[2])
        return [v[0] / n, v[1] / n, v[2] / n]
      }
      const a = unit(before)
      const b = unit(after)
      maxJump = Math.max(maxJump, Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!))
    }
    // The seam crossing perturbs the direction by 2e-6; anything the warp
    // adds on top of that would be orders of magnitude larger.
    expect(maxJump).toBeLessThan(1e-4)
  })
})

describe('cubeFaceSize', () => {
  it('reads the face size off a real cube texture', () => {
    expect(cubeFaceSize(bakeSyntheticCubeTexture({ size: 8 }))).toBe(8)
    expect(cubeFaceSize(bakeSyntheticCubeTexture({ size: 2 }))).toBe(2)
  })

  it('THROWS rather than defaulting when it cannot determine one', () => {
    // A wrong face size does not look wrong, it looks like a slightly
    // different flavour of the same blockiness — so a silent default here
    // would be undebuggable. Matches loadCube.ts's discipline elsewhere.
    const empty = { images: [] } as unknown as Parameters<typeof cubeFaceSize>[0]
    expect(() => cubeFaceSize(empty)).toThrow(/no integer image width/)
  })
})

describe('the shell actually uses the warp', () => {
  it('warps the density cube direction on that atlas\'s own face size', () => {
    // Pre-fix source samples `cubeTexture(texture, ps)` directly — this
    // would not match against it.
    expect(shellSource).toMatch(/const dir = smoothstepCubeDirectionNode\(worldDirection, cubeFaceSize\(texture\)\)/)
    expect(shellSource).toMatch(/const cubeSample = cubeTexture\(texture, dir\)\.level\(float\(0\)\)/)
  })

  it('warps the urban cube on the URBAN atlas\'s face size, not the density one', () => {
    // The two tiers differ; reusing the density cube's pitch would warp the
    // urban sample at the wrong spatial frequency.
    expect(shellSource).toMatch(/smoothstepCubeDirectionNode\(worldDirection, cubeFaceSize\(urban\.texture\)\)/)
  })

  it('does not warp the analytic field — it has no texels to reconstruct', () => {
    expect(shellSource).toMatch(/const tauSampled = applyCobbMix\(syntheticTauNode\(worldDirection\), worldDirection, opts\.urban\)/)
  })

  it('mirrors the CPU warp term for term in TSL', () => {
    // Same three stages as smoothstepTexelWarp: to texel-centre space,
    // floor/smoothstep the fraction, back to a face coordinate.
    expect(terrainNodesSource).toMatch(/const p = c\.add\(1\)\.mul\(0\.5 \* faceSize\)\.sub\(0\.5\)/)
    expect(terrainNodesSource).toMatch(/const warped = i\.add\(f\.mul\(f\)\.mul\(f\.mul\(-2\)\.add\(3\)\)\)/)
    expect(terrainNodesSource).toMatch(/return warped\.add\(0\.5\)\.div\(faceSize\)\.mul\(2\)\.sub\(1\)/)
  })
})

/**
 * A TSL node graph is symbolic — building it never evaluates numbers, so a
 * flipped sign or a dropped term survives "builds without throwing" intact.
 * `mirror` below reproduces `terrainNodes.ts::smoothstepCubeDirectionNode`'s
 * expression SHAPE with ordinary arithmetic, so the port's algebra can be
 * property-tested against the CPU source of truth. The WGSL compile and the
 * GPU execution itself stay unverified in this environment (no browser); the
 * `cubeArrayNodes.test.ts` precedent takes the same position for the same
 * reason.
 */
function smoothstepCubeDirectionNodeMirror(
  x: number, y: number, z: number, faceSize: number,
): number[] {
  const m = Math.max(Math.max(Math.abs(x), Math.abs(y)), Math.abs(z))
  const denom = Math.max(m, 1e-20)
  const c = [x / denom, y / denom, z / denom]
  const p = c.map((v) => (v + 1) * (0.5 * faceSize) - 0.5)
  const i = p.map((v) => Math.floor(v))
  const f = p.map((v, k) => v - i[k]!)
  const warped = i.map((v, k) => v + f[k]! * f[k]! * (f[k]! * -2 + 3))
  return warped.map((v) => ((v + 0.5) / faceSize) * 2 - 1)
}

describe('the TSL warp agrees numerically with the CPU one', () => {
  it('matches smoothstepCubeDirection over a sphere of directions and every shipped face size', () => {
    for (const faceSize of [64, 128, 256, 512, 1024, 2048]) {
      for (let a = 0; a < 17; a++) {
        for (let b = 0; b < 23; b++) {
          const theta = ((a + 0.5) / 17) * Math.PI
          const phi = ((b + 0.5) / 23) * 2 * Math.PI
          const x = Math.sin(theta) * Math.cos(phi)
          const y = Math.sin(theta) * Math.sin(phi)
          const z = Math.cos(theta)
          const cpu = smoothstepCubeDirection(x, y, z, faceSize)
          const gpu = smoothstepCubeDirectionNodeMirror(x, y, z, faceSize)
          for (let k = 0; k < 3; k++) expect(gpu[k]!).toBeCloseTo(cpu[k]!, 12)
        }
      }
    }
  })

  it('agrees on unnormalized directions too — the shell passes sphere-space `ps`, not a unit vector', () => {
    for (const scale of [1.0, 1.0301, 6378137, 1e-4]) {
      const cpu = smoothstepCubeDirection(0.31 * scale, -0.87 * scale, 0.42 * scale, 128)
      const gpu = smoothstepCubeDirectionNodeMirror(0.31 * scale, -0.87 * scale, 0.42 * scale, 128)
      for (let k = 0; k < 3; k++) expect(gpu[k]!).toBeCloseTo(cpu[k]!, 12)
    }
  })

  it('differs from the CPU version ONLY on the zero direction, where the node clamps and the CPU throws', () => {
    // Deliberate: a JS caller passing (0,0,0) has a bug worth surfacing,
    // while a shader cannot throw and must not produce NaN inside the
    // march. Neither path is reachable from the shell (every marched point
    // sits on or above the datum, so |ps| >= 1).
    expect(() => smoothstepCubeDirection(0, 0, 0, 128)).toThrow(/zero direction/)
    expect(smoothstepCubeDirectionNodeMirror(0, 0, 0, 128).every(Number.isFinite)).toBe(true)
  })
})
