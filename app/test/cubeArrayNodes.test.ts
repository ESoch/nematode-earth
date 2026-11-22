import { describe, it, expect } from 'vitest'
import { vec3 } from 'three/tsl'
import { directionToFaceUvNode, sampleDensityArrayNode } from '../src/field/cubeArrayNodes'
import { directionToFaceUV } from '../src/field/loadCube'
import { bakeSyntheticDensityArrayTexture } from '../src/field/bakeCube'
// Vite's `?raw` suffix - see test/dissolveNodes.test.ts for the precedent.
import cubeArrayNodesSource from '../src/field/cubeArrayNodes.ts?raw'

/**
 * A TSL node graph is symbolic - building it (calling `.select()`,
 * `.greaterThanEqual()`, etc.) never evaluates numbers, so "does not throw"
 * alone cannot catch a wrong branch or a flipped sign the way it can for a
 * scene-graph API misuse. This plain-JS mirror reproduces
 * `directionToFaceUvNode`'s select-chain SHAPE with ordinary ternaries
 * instead of `.select()`, so it can be numerically property-tested against
 * `loadCube.ts`'s already-tested `directionToFaceUV` — giving real evidence
 * the TSL port's algebra is correct, even though the actual WGSL compile
 * and GPU execution remain unverified without a browser (no browser access
 * in this environment).
 *
 * Includes `safeDenom`'s sign-preserving zero-guard too (mirrored, not
 * re-derived), since `cubeArrayNodes.ts` now applies it to every division's
 * denominator — see that file's module doc for why (an axis-aligned
 * direction hits a literal-zero denominator in a DISCARDED branch; WGSL's
 * constant folder rejects that at parse time for a literal input, and even
 * for a real runtime input it is fragile to rely on IEEE754 infinity
 * falling out correctly).
 */
function safeDenomMirror(d: number): number {
  return d >= 0 ? Math.max(d, 1e-20) : Math.min(d, -1e-20)
}

function directionToFaceUvMirror(x: number, y: number, z: number): { face: number, u: number, v: number } {
  const ax = Math.abs(x)
  const ay = Math.abs(y)
  const az = Math.abs(z)
  const xMajor = ax >= ay && ax >= az
  const yMajor = ay >= ax && ay >= az
  const xPos = x > 0
  const yPos = y > 0
  const zPos = z > 0

  const xSafe = safeDenomMirror(x)
  const ySafe = safeDenomMirror(y)
  const zSafe = safeDenomMirror(z)

  const face = xMajor
    ? (xPos ? 0 : 1)
    : (yMajor ? (yPos ? 2 : 3) : (zPos ? 4 : 5))

  const uX = -z / xSafe
  const vX = xPos ? -y / xSafe : y / xSafe
  const uY = yPos ? x / ySafe : -x / ySafe
  const vY = z / ySafe
  const uZ = x / zSafe
  const vZ = zPos ? -y / zSafe : y / zSafe

  const u = xMajor ? uX : (yMajor ? uY : uZ)
  const v = xMajor ? vX : (yMajor ? vY : vZ)
  return { face, u, v }
}

// Deterministic direction set covering all 6 faces, off-axis points, and
// near-edge points (not a random sampler - reproducible without a seed).
function testDirections(): [number, number, number][] {
  const dirs: [number, number, number][] = []
  const steps = 9
  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < steps; j++) {
      const theta = (i / (steps - 1)) * Math.PI // 0..pi
      const phi = (j / (steps - 1)) * 2 * Math.PI // 0..2pi
      const x = Math.sin(theta) * Math.cos(phi)
      const y = Math.sin(theta) * Math.sin(phi)
      const z = Math.cos(theta)
      if (x === 0 && y === 0 && z === 0) continue
      dirs.push([x, y, z])
    }
  }
  return dirs
}

describe('directionToFaceUvMirror matches directionToFaceUV (ground truth)', () => {
  it('agrees on face, u, and v for a grid of directions covering all 6 faces', () => {
    let facesSeen = new Set<number>()
    for (const [x, y, z] of testDirections()) {
      const expected = directionToFaceUV(x, y, z)
      const actual = directionToFaceUvMirror(x, y, z)
      expect(actual.face).toBe(expected.face)
      expect(actual.u).toBeCloseTo(expected.u, 10)
      expect(actual.v).toBeCloseTo(expected.v, 10)
      facesSeen.add(actual.face)
    }
    expect(facesSeen.size).toBe(6) // the grid must actually exercise every face
  })

  it('agrees exactly at axis-aligned directions - the WGSL constant-fold hazard case', () => {
    // These are exactly the directions where a DISCARDED branch divides by
    // a literal zero (see cubeArrayNodes.ts's module doc) - confirming
    // safeDenomMirror does not perturb the SELECTED branch's result here
    // is confirming the guard is a no-op for real, non-degenerate inputs,
    // not just "doesn't throw".
    for (const [x, y, z] of [
      [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ] as const) {
      const expected = directionToFaceUV(x, y, z)
      const actual = directionToFaceUvMirror(x, y, z)
      expect(actual.face).toBe(expected.face)
      expect(actual.u).toBeCloseTo(expected.u, 10)
      expect(actual.v).toBeCloseTo(expected.v, 10)
    }
  })
})

describe('directionToFaceUvNode / sampleDensityArrayNode node construction', () => {
  it('builds a node graph without throwing for directions on every face', () => {
    for (const [x, y, z] of [
      [1, 0.2, 0.3], [-1, 0.2, 0.3], [0.2, 1, 0.3], [0.2, -1, 0.3], [0.2, 0.3, 1], [0.2, 0.3, -1],
    ] as const) {
      expect(() => directionToFaceUvNode(vec3(x, y, z))).not.toThrow()
    }
  })

  it('builds a full density-array sample without throwing', () => {
    const arrayTex = bakeSyntheticDensityArrayTexture({ size: 4 })
    expect(() => sampleDensityArrayNode(arrayTex, vec3(1, 0.2, 0.3))).not.toThrow()
  })
})

describe('the TSL algebra matches the mirror/CPU algebra (structural check)', () => {
  it('uses .select( for every branch, never If/Else', () => {
    expect(cubeArrayNodesSource).toMatch(/\.select\(/)
    expect(cubeArrayNodesSource).not.toMatch(/\bIf\(/)
  })
  it('uses explicit level(0) - required inside compute / non-uniform control flow', () => {
    expect(cubeArrayNodesSource).toMatch(/\.level\(float\(0\)\)/)
  })
  it('never imports cubeTexture from three/tsl - the whole point of this module', () => {
    // Checks the import, not just a call, since "cubeTexture" also appears
    // in this file's own doc comments explaining why it is avoided.
    const importLine = cubeArrayNodesSource.split('\n').find((l) => l.includes("from 'three/tsl'"))
    expect(importLine).toBeDefined()
    expect(importLine).not.toMatch(/\bcubeTexture\b/)
  })
  it('every division denominator goes through safeDenom, not a raw component', () => {
    // Regression guard for the WGSL constant-fold bug: every `.div(` call's
    // argument must be one of the *Safe locals, never a bare x/y/z.
    const divCalls = cubeArrayNodesSource.match(/\.div\(\w+\)/g) ?? []
    expect(divCalls.length).toBeGreaterThan(0)
    for (const call of divCalls) {
      expect(call).toMatch(/\.div\((xSafe|ySafe|zSafe)\)/)
    }
  })
})
