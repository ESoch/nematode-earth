import { describe, it, expect } from 'vitest'
import { float, vec3 } from 'three/tsl'
import * as terrainNodes from '../src/field/terrainNodes'
import { detailGradientEps, hypsometricColorNode, terrainNormalNode } from '../src/field/terrainNodes'
import {
  uncertaintyColorNode, uncertaintyCoverageNode, uncertaintyOverlayColorNode,
} from '../src/field/terrainNodes'
import { uncertaintyCoverage, UNCERTAINTY_LAND_MIN, UNCERTAINTY_LAND_FULL } from '../src/field/terrain'
import { bakeSyntheticCubeTexture } from '../src/field/bakeCube'
// Vite's `?raw` suffix — the test/dissolveNodes.test.ts precedent.
import terrainNodesSource from '../src/field/terrainNodes.ts?raw'

// detailBumpedNormalNode's finite-difference step lives in p-space (already
// scaled by DETAIL_FREQUENCY), where mx_fractal_noise_float's octave `i`
// (0-indexed) samples at p-space frequency `lacunarity^i` relative to the
// base octave - see node_modules/three's mx_fractal_noise_float: each loop
// iteration multiplies `p` by `lacunarity` BEFORE the next tap, so octave 0
// keeps the base period (1 p-unit, standard Perlin lattice spacing) and the
// finest octave (index `octaves-1`) has period `1 / lacunarity^(octaves-1)`.
describe('detailGradientEps', () => {
  it("for this app's 3-octave, 2.3-lacunarity noise, comes out far below the old fixed 0.5 step", () => {
    // The old bug: eps=0.5 was 2.6x the finest octave's own wavelength
    // (~0.189 p-units for octaves=3, lacunarity=2.3), so a two-tap
    // central difference spanning more than a full period of that octave
    // washed its contribution out of the gradient almost entirely.
    const eps = detailGradientEps(3, 2.3)
    expect(eps).toBeLessThan(0.5)
    expect(eps).toBeCloseTo(0.5 / 2.3 ** 2, 9)
  })

  it("is roughly half the finest octave's own p-space wavelength, not several times it", () => {
    const octaves = 3
    const lacunarity = 2.3
    const finestPeriod = 1 / lacunarity ** (octaves - 1)
    const eps = detailGradientEps(octaves, lacunarity)
    expect(eps / finestPeriod).toBeCloseTo(0.5, 9)
  })

  it('shrinks as octave count grows, keeping pace with ever-finer detail', () => {
    expect(detailGradientEps(4, 2.3)).toBeLessThan(detailGradientEps(3, 2.3))
  })

  it('shrinks as lacunarity grows', () => {
    expect(detailGradientEps(3, 3)).toBeLessThan(detailGradientEps(3, 2.3))
  })

  it('is exactly half the base period (1 p-unit) for a single octave', () => {
    expect(detailGradientEps(1, 2.3)).toBeCloseTo(0.5, 12)
  })
})

describe('sampleLandFractionNode', () => {
  it('has been deleted - the density cube land fraction it read has no consumer left', () => {
    // Matter migrated from this (density cube, ~78 km/texel) to
    // sampleLandMaskNode (elevation cube, ~9.77 km/texel) - see
    // matter.test.ts's assertion that Matter's source no longer calls this.
    // The doc comment above the old definition claimed the veil "still uses
    // it", which was already false; rather than leave a comment asserting a
    // consumer that does not exist, the dead function is removed outright.
    expect('sampleLandFractionNode' in terrainNodes).toBe(false)
  })
})

// Neither function below can be numerically evaluated in this test
// environment (no WebGPU execution - see webgpu.test.ts/isWebGPUAvailable).
// "builds a node graph without throwing" (the cubeArrayNodes.test.ts
// pattern) alone is too weak a regression pin here: JS does not enforce
// argument count, so calling the OLD 4-parameter signature with 5
// positional args silently accepts them (the 5th is dropped, or - worse,
// for terrainNormalNode - a Node object gets bound to the plain-JS-number
// `texelAngularWidthRad` parameter and never throws, it just silently
// produces NaN downstream). `.length` (a function's declared, non-default
// arity) is what actually distinguishes the old and new signatures, so
// that is the real regression pin; "doesn't throw" is kept alongside only
// as a basic construction sanity check once arity is confirmed. The actual
// FORMULAS these mirror (effectiveExaggeration, hypsometricColor's
// oceanProgress gating) are pinned with real numeric tests in
// terrain.test.ts.
describe('hypsometricColorNode / terrainNormalNode node construction', () => {
  it('hypsometricColorNode takes an oceanProgress parameter (arity 5, was 4)', () => {
    expect(hypsometricColorNode.length).toBe(5)
  })

  it('hypsometricColorNode builds with an oceanProgress node without throwing', () => {
    expect(() => hypsometricColorNode(
      float(-4000), float(0), float(0), float(0.5), float(0.5),
    )).not.toThrow()
  })

  it('terrainNormalNode takes a landFraction parameter (arity 5, was 4)', () => {
    expect(terrainNormalNode.length).toBe(5)
  })

  it('terrainNormalNode builds with a landFraction node without throwing', () => {
    const fakeCube = bakeSyntheticCubeTexture({ size: 2 })
    expect(() => terrainNormalNode(fakeCube, vec3(1, 0.2, 0.3), float(3.6), float(0.5), 0.01)).not.toThrow()
  })
})

// --- Uncertainty overlay: CPU/TSL agreement --------------------------------
//
// Numeric GPU execution is unavailable here (see the note above), so
// "agreement" is established the two ways this codebase already establishes
// it, and neither is "it did not throw":
//
//  1. SHARED CONSTANTS. `uncertaintyColorNode` feeds `terrain.ts`'s exported
//     `UNCERTAINTY_STOPS` into the same `rampColorNode` cascade that
//     LAND_STOPS/OCEAN_STOPS use, exactly as `uncertaintyColor` feeds them
//     into `rampColor`. The ramps cannot drift apart because there is only
//     one copy of the numbers — which is only true as long as nobody
//     re-types them, and that is what the source pins below check.
//  2. A PLAIN-JS MIRROR of the node's own algebra, numerically compared
//     against the CPU function across a grid — the cubeArrayNodes.test.ts
//     precedent, used here for the coverage smoothstep, where a swapped or
//     mistyped edge is the realistic failure and would otherwise be
//     invisible.
describe('uncertainty overlay nodes mirror terrain.ts', () => {
  it('builds the ramp and coverage nodes without throwing', () => {
    expect(() => uncertaintyColorNode(float(0.18))).not.toThrow()
    expect(() => uncertaintyCoverageNode(float(0.5))).not.toThrow()
    expect(() => uncertaintyOverlayColorNode(
      vec3(0.72, 0.86, 0.95), float(0.18), float(1), float(1),
    )).not.toThrow()
  })

  it('takes base, cv, landFraction and strength — arity pin against a silent signature change', () => {
    expect(uncertaintyOverlayColorNode.length).toBe(4)
    expect(uncertaintyColorNode.length).toBe(1)
    expect(uncertaintyCoverageNode.length).toBe(1)
  })

  it('builds its ramp from terrain.ts UNCERTAINTY_STOPS, never a second hand-copied table', () => {
    const importBlock = terrainNodesSource.slice(0, terrainNodesSource.indexOf("from './terrain'"))
    expect(importBlock).toMatch(/\bUNCERTAINTY_STOPS\b/)
    expect(terrainNodesSource).toMatch(/rampColorNode\(UNCERTAINTY_STOPS,/)
    // The stop colours must appear nowhere in this file as literals. If
    // someone inlines the table here, the CPU and GPU ramps become two
    // independently-editable copies and this fails.
    for (const literal of ['0.16, 0.05, 0.09', '0.42, 0.10, 0.16', '0.99, 0.80, 0.48']) {
      expect(terrainNodesSource).not.toContain(literal)
    }
  })

  it('drives coverage off land fraction, not off the CV channel', () => {
    // THE bug this whole feature has to avoid: gating on B (which is 0 over
    // ocean, i.e. indistinguishable from "extremely certain") instead of on
    // A. The mirror below cannot catch it — both are floats — so it is
    // pinned structurally, at the one place the choice is made.
    expect(terrainNodesSource).toMatch(
      /uncertaintyCoverageNode\(landFractionNode: FloatNode\)/,
    )
    expect(terrainNodesSource).toMatch(
      /smoothstep\(float\(UNCERTAINTY_LAND_MIN\), float\(UNCERTAINTY_LAND_FULL\), landFractionNode\)/,
    )
  })

  it('agrees numerically with uncertaintyCoverage across the full land-fraction range', () => {
    // Mirror of uncertaintyCoverageNode's algebra in plain JS, matching it
    // term for term (smoothstep with the same two shared edges).
    const mirror = (a: number): number => {
      const t = Math.min(1, Math.max(0, (a - UNCERTAINTY_LAND_MIN) / (UNCERTAINTY_LAND_FULL - UNCERTAINTY_LAND_MIN)))
      return t * t * (3 - 2 * t)
    }
    for (let a = -0.2; a <= 1.2001; a += 0.02) {
      expect(mirror(a)).toBeCloseTo(uncertaintyCoverage(a), 12)
    }
    // and the endpoints that actually matter
    expect(mirror(0)).toBe(0)
    expect(mirror(1)).toBeCloseTo(1, 12)
  })
})
