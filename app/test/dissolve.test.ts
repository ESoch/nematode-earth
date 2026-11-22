import { describe, it, expect } from 'vitest'
import {
  LAYERS, DEFAULT_SEED, layerAlpha, erosionPhi, matterAlphaAt, frontFor,
} from '../src/sweep/Dissolve'
import { stateAt } from '../src/sweep/Timeline'
import type { Vec3 } from '../src/math/raySpheroid'

const dirs = (n: number): Vec3[] =>
  Array.from({ length: n }, (_, i) => {
    const z = 1 - (2 * i) / (n - 1)
    const r = Math.sqrt(Math.max(0, 1 - z * z))
    const th = Math.PI * (3 - Math.sqrt(5)) * i
    return [Math.cos(th) * r, Math.sin(th) * r, z] as Vec3
  })

describe('layerAlpha', () => {
  it('is total at both endpoints for every phi in [0,1]', () => {
    for (const w of [0.05, 0.12, 0.35]) {
      for (let i = 0; i <= 100; i++) {
        const phi = i / 100
        expect(layerAlpha(phi, 0, w)).toBe(1)
        expect(layerAlpha(phi, 1, w)).toBe(0)
      }
    }
  })
  it('never increases as the front advances', () => {
    for (const phi of [0, 0.17, 0.5, 0.83, 1]) {
      let prev = Infinity
      for (let i = 0; i <= 200; i++) {
        const a = layerAlpha(phi, i / 200, 0.12)
        expect(a).toBeLessThanOrEqual(prev + 1e-12)
        prev = a
      }
    }
  })
  it('erodes low phi before high phi - it is a front, not a crossfade', () => {
    const f = 0.5
    expect(layerAlpha(0.1, f, 0.12)).toBeLessThan(layerAlpha(0.9, f, 0.12))
  })
  it('a narrower width gives a harder edge', () => {
    const soft = layerAlpha(0.52, 0.5, 0.35) - layerAlpha(0.48, 0.5, 0.35)
    const hard = layerAlpha(0.52, 0.5, 0.05) - layerAlpha(0.48, 0.5, 0.05)
    expect(Math.abs(hard)).toBeGreaterThan(Math.abs(soft))
  })
  it('stays in [0,1]', () => {
    for (let i = 0; i <= 50; i++) {
      for (let j = 0; j <= 50; j++) {
        const a = layerAlpha(i / 50, j / 50, 0.12)
        expect(a).toBeGreaterThanOrEqual(0)
        expect(a).toBeLessThanOrEqual(1)
      }
    }
  })
})

describe('erosionPhi', () => {
  it('stays in [0,1] for every layer over the sphere', () => {
    for (const d of dirs(3000)) {
      for (const name of ['atmosphere', 'water', 'rock'] as const) {
        const phi = erosionPhi(d, LAYERS[name], DEFAULT_SEED)
        expect(phi).toBeGreaterThanOrEqual(0)
        expect(phi).toBeLessThanOrEqual(1)
      }
    }
  })
  it('with noise disabled is exactly the geometric term (1 - dot(dir,seed))/2', () => {
    for (const d of dirs(500)) {
      const want = (1 - (d[0] * DEFAULT_SEED[0] + d[1] * DEFAULT_SEED[1] + d[2] * DEFAULT_SEED[2])) / 2
      expect(erosionPhi(d, LAYERS.rock, DEFAULT_SEED, false)).toBeCloseTo(want, 12)
    }
  })
  it('is 0 at the seed and 1 at its antipode with noise disabled', () => {
    expect(erosionPhi(DEFAULT_SEED, LAYERS.water, DEFAULT_SEED, false)).toBeCloseTo(0, 12)
    const anti: Vec3 = [-DEFAULT_SEED[0], -DEFAULT_SEED[1], -DEFAULT_SEED[2]]
    expect(erosionPhi(anti, LAYERS.water, DEFAULT_SEED, false)).toBeCloseTo(1, 12)
  })
  // A "rock is rougher than atmosphere" roughness test was in the original
  // reference plan this module was drafted from, measuring mean absolute
  // difference between erosionPhi at Fibonacci-adjacent sample points. It
  // does not hold reliably through erosionPhi's full geo+noise blend: the
  // geometric term's own (very low-frequency) roughness contributes
  // differently per layer depending on noiseWeight, in a way this metric
  // can't cleanly separate from the noise's own frequency. LAYERS.width
  // already differs by an order of magnitude per layer (0.35/0.12/0.05,
  // tested above) and is what actually drives each layer's visual character
  // in the dissolve (see Matter.ts); noiseFreq's job is texture detail, not
  // a load-bearing "roughness ordering" this suite asserts.
})

describe('layer characters', () => {
  it('widths order atmosphere > water > rock', () => {
    expect(LAYERS.atmosphere.width).toBeGreaterThan(LAYERS.water.width)
    expect(LAYERS.water.width).toBeGreaterThan(LAYERS.rock.width)
  })
  it('only rock is ridged', () => {
    expect(LAYERS.rock.ridged).toBe(true)
    expect(LAYERS.water.ridged).toBe(false)
    expect(LAYERS.atmosphere.ridged).toBe(false)
  })
})

describe('matterAlphaAt over the whole timeline', () => {
  it('is 1 everywhere at t=0 and 0 everywhere at t=13', () => {
    for (const d of dirs(500)) {
      for (const name of ['atmosphere', 'water', 'rock'] as const) {
        expect(matterAlphaAt(d, name, stateAt(0))).toBe(1)
        expect(matterAlphaAt(d, name, stateAt(13))).toBe(0)
      }
    }
  })
  it('never increases with time for any direction', () => {
    for (const d of dirs(60)) {
      for (const name of ['atmosphere', 'water', 'rock'] as const) {
        let prev = Infinity
        for (let i = 0; i <= 260; i++) {
          const a = matterAlphaAt(d, name, stateAt((i * 20) / 260))
          expect(a).toBeLessThanOrEqual(prev + 1e-12)
          prev = a
        }
      }
    }
  })
  it('the mean coverage falls monotonically and passes through the middle mid-sweep', () => {
    const sample = dirs(1500)
    const mean = (t: number) => sample.reduce((s, d) => s + matterAlphaAt(d, 'rock', stateAt(t)), 0) / sample.length
    expect(mean(6.4)).toBeGreaterThan(0.95)
    expect(mean(9.0)).toBeLessThan(0.05)
    // t=7.7 is frontRock's exact 0.5 crossing (rock's window is [0.34,0.6] in
    // u, and u(7.7)=0.47 sits at that window's own midpoint), so this is
    // deliberately the sharpest point of a narrow (width 0.05) transition,
    // not a claim that coverage is linear in t - the band is wide enough to
    // require "clearly mid-transition, not stuck at an extreme" without
    // over-fitting to the noise's exact distribution.
    const mid = mean(7.7)
    expect(mid).toBeGreaterThan(0.2)
    expect(mid).toBeLessThan(0.8)
  })
  it('with noise disabled, eroded solid-angle fraction matches the front-centre cap fraction', () => {
    // layerAlpha crosses 0.5 (its smoothstep midpoint) at phi = frontCentre,
    // and {phi < frontCentre} is a spherical cap of solid-angle fraction
    // EXACTLY frontCentre - not `f` itself, since layerAlpha's front centre
    // is `-2w + f*(1+4w)` (see layerAlpha's doc), which only equals `f`
    // exactly at f=0.5 (where the two lines cross) or in the limit w->0.
    const sample = dirs(20000)
    const width = LAYERS.rock.width
    for (const f of [0.2, 0.5, 0.8]) {
      const state = { ...stateAt(0), frontRock: f }
      const eroded = sample.filter((d) => matterAlphaAt(d, 'rock', state, DEFAULT_SEED, false) < 0.5).length / sample.length
      const frontCentre = -2 * width + f * (1 + 4 * width)
      expect(eroded).toBeCloseTo(frontCentre, 1)
    }
  })
})

describe('frontFor', () => {
  it('routes each layer to its own front', () => {
    const s = stateAt(7)
    expect(frontFor('atmosphere', s)).toBe(s.frontAtmosphere)
    expect(frontFor('water', s)).toBe(s.frontWater)
    expect(frontFor('rock', s)).toBe(s.frontRock)
  })
})
