import { describe, it, expect } from 'vitest'
import { ScaleCamera, ALT_MIN, ALT_MAX, GROUND_CLEARANCE_M } from '../src/core/ScaleCamera'
import { wheelDeltaToTicks, pinchDistanceToTicks } from '../src/core/orbitInput'
import { OpeningExperience } from '../src/core/openingExperience'
import { Timeline } from '../src/sweep/Timeline'
import { EVEREST_M } from '../src/math/descent'
import type { Vec3 } from '../src/math/raySpheroid'

describe('logarithmic descent', () => {
  it('gives a constant altitude RATIO per scroll tick', () => {
    const c = new ScaleCamera()
    c.setAltitude(1e6)
    const a0 = c.altitude
    c.scrollBy(1)
    const a1 = c.altitude
    c.scrollBy(1)
    const a2 = c.altitude
    expect(a1 / a0).toBeCloseTo(a2 / a1, 9)
  })

  it('takes a similar tick count across every decade', () => {
    const c = new ScaleCamera()
    const ticksFor = (from: number, to: number) => {
      c.setAltitude(from)
      let n = 0
      while (c.altitude > to && n < 10000) { c.scrollBy(1); n++ }
      return n
    }
    const highDecade = ticksFor(1e7, 1e6)
    const lowDecade = ticksFor(1e2, 1e1)
    expect(Math.abs(highDecade - lowDecade)).toBeLessThanOrEqual(1)
  })

  it('clamps to the altitude bounds', () => {
    const c = new ScaleCamera()
    c.setAltitude(1e12)
    expect(c.altitude).toBeCloseTo(ALT_MAX, 6)
    c.setAltitude(-5)
    expect(c.altitude).toBeCloseTo(ALT_MIN, 6)
  })

  it('ascends symmetrically', () => {
    const c = new ScaleCamera()
    c.setAltitude(1e5)
    const before = c.altitude
    c.scrollBy(3)
    c.scrollBy(-3)
    expect(c.altitude).toBeCloseTo(before, 6)
  })
})

describe('floating origin', () => {
  it('places the camera itself at the render origin', () => {
    const c = new ScaleCamera()
    c.setAltitude(1000)
    const [x, y, z] = c.worldToRender(c.worldPosition)
    expect(Math.hypot(x, y, z)).toBeLessThan(1e-6)
  })

  it('keeps nearby offsets exactly representable at millimetre scale', () => {
    const c = new ScaleCamera()
    c.setAltitude(0.01)
    const w = c.worldPosition
    const nearby: [number, number, number] = [w[0] + 0.001, w[1], w[2]]
    const [rx] = c.worldToRender(nearby)
    // In f32 world space anchored at the planet centre this would quantize to 0.
    expect(Math.abs(Math.fround(rx) - 0.001)).toBeLessThan(1e-9)
  })
})

describe('descent state coupling', () => {
  it('exposes a scale height that tracks altitude', () => {
    const c = new ScaleCamera()
    c.setAltitude(192000)
    expect(c.state.Hs).toBeCloseTo(32000, 6)
    c.setAltitude(60000)
    expect(c.state.Hs).toBeCloseTo(10000, 6)
  })

  it('never lets EXAGGERATED terrain exceed one shell e-folding', () => {
    // Vacuous below alt = 6*EVEREST_M = 53094 m, where k is floored at 1.0 and
    // no exaggeration is applied.
    const c = new ScaleCamera()
    for (const a of [1e7, 1e5, 192000, 32000, 5000, 100]) {
      c.setAltitude(a)
      if (c.state.k > 1.0) {
        expect(c.state.k * EVEREST_M).toBeLessThanOrEqual(c.state.Hs * 1.000001)
      }
    }
  })

  it('applies no terrain exaggeration below the crossover altitude', () => {
    const c = new ScaleCamera()
    for (const a of [100, 5000, 32000]) {
      c.setAltitude(a)
      expect(c.state.k).toBe(1.0)
    }
    c.setAltitude(192000)
    expect(c.state.k).toBeGreaterThan(1.0)
  })
})

// Altitude is a radius offset from WGS84.a, but the matter mesh in front of
// the camera is DISPLACED by elevation * k, so over high terrain the two
// disagree by kilometres and the camera descends straight through the mesh.
// Measured before this floor existed: 100 m over Tibet rendered exactly
// black (0.00% non-zero pixels); 100 m over the Indian Ocean rendered
// normally. See field/groundHeight.ts for why the floor is a bound over a
// tessellation cell rather than the terrain height at a point.
describe('the ground floor', () => {
  const at = (h: number) => () => h

  it('leaves altitude exactly as set when no ground sampler is attached', () => {
    const c = new ScaleCamera()
    c.setAltitude(100)
    expect(c.altitude).toBe(100)
  })

  it('never descends below the displaced mesh under the camera', () => {
    const c = new ScaleCamera()
    c.groundHeightAt = at(5000)
    c.setAltitude(100)
    expect(c.altitude).toBeCloseTo(5000 + GROUND_CLEARANCE_M, 6)
  })

  it('stands just clear of the ground, so the closest shot survives', () => {
    const c = new ScaleCamera()
    c.groundHeightAt = at(5000)
    c.setAltitude(ALT_MIN)
    expect(c.altitude - 5000).toBeCloseTo(GROUND_CLEARANCE_M, 9)
    // Must clear viewer.ts's 1 m camera near plane, or the ground under the
    // camera is clipped and the viewer sees straight through it — and stay
    // small enough that the close shot is still a close shot.
    expect(GROUND_CLEARANCE_M).toBeGreaterThan(1)
    expect(GROUND_CLEARANCE_M).toBeLessThan(10)
  })

  it('does not lift the camera over ground that sits below datum', () => {
    const c = new ScaleCamera()
    c.groundHeightAt = at(-4000) // open ocean before the bathymetry beat drops it
    c.setAltitude(ALT_MIN)
    expect(c.altitude).toBe(ALT_MIN)
  })

  it('imposes nothing at all when the sampler reports no ground', () => {
    // null is "no atlas has loaded" or "the sweep took matter away", NOT
    // "the ground is at sea level" — reading it as 0 would quietly make the
    // clearance a global minimum altitude and delete the closest shot.
    const c = new ScaleCamera()
    c.groundHeightAt = () => null
    c.setAltitude(ALT_MIN)
    expect(c.altitude).toBe(ALT_MIN)
    expect(c.groundFloor).toBe(0)
  })

  it('leaves ALT_MIN reachable over open water, where the bound is exactly zero', () => {
    // The bound viewer.ts supplies is the cell MAXIMUM, and over open water
    // every corner of a cell sits exactly at datum until the bathymetry beat
    // drops it. Adding the clearance to that zero would raise the minimum
    // altitude over every ocean on Earth, breaking a shot that has nothing
    // to do with the defect this floor fixes.
    const c = new ScaleCamera()
    c.groundHeightAt = at(0)
    c.setAltitude(ALT_MIN)
    expect(c.altitude).toBe(ALT_MIN)
    expect(c.groundFloor).toBe(0)
  })

  it('samples at the camera\'s own direction, not a fixed one', () => {
    const seen: Vec3[] = []
    const c = new ScaleCamera()
    c.groundHeightAt = (dir) => { seen.push(dir); return 0 }
    c.setAltitude(1000)
    c.orbitBy(1.0, 0.5)
    void c.altitude
    const last = seen[seen.length - 1]!
    const d = c.direction
    expect(last[0]).toBeCloseTo(d[0], 12)
    expect(last[1]).toBeCloseTo(d[1], 12)
    expect(last[2]).toBeCloseTo(d[2], 12)
  })

  it('lifts the camera when orbiting ONTO high ground, and lets it back down after', () => {
    // The floor is evaluated live, so a camera that never touched the scroll
    // wheel returns to the altitude it asked for once it is over the sea
    // again — being lifted over a mountain is not a state it gets stuck in.
    const c = new ScaleCamera()
    let overMountain = false
    c.groundHeightAt = () => (overMountain ? 5000 : 0)
    c.setAltitude(100)
    expect(c.altitude).toBe(100)
    overMountain = true
    expect(c.altitude).toBeCloseTo(5000 + GROUND_CLEARANCE_M, 6)
    overMountain = false
    expect(c.altitude).toBe(100)
  })

  it('makes scrolling further in a no-op at the floor, and scrolling out move at once', () => {
    const c = new ScaleCamera()
    c.groundHeightAt = at(5000)
    c.setAltitude(100)
    const floor = c.altitude
    c.scrollBy(5)
    expect(c.altitude).toBeCloseTo(floor, 6)
    c.scrollBy(-1)
    expect(c.altitude).toBeGreaterThan(floor * 1.05)
  })

  it('keeps worldPosition on the floored altitude, not the requested one', () => {
    const c = new ScaleCamera()
    c.groundHeightAt = at(5000)
    c.setAltitude(100)
    const r = Math.hypot(...c.worldPosition)
    expect(r - 6378137.0).toBeCloseTo(5000 + GROUND_CLEARANCE_M, 3)
  })
})

describe('pose', () => {
  it('round-trips the longitude and latitude it was set to', () => {
    const c = new ScaleCamera()
    c.setOrientation(1.2, -0.4)
    expect(c.longitude).toBeCloseTo(1.2, 12)
    expect(c.latitude).toBeCloseTo(-0.4, 12)
  })

  it('clamps latitude short of the poles, exactly as orbitBy does', () => {
    const c = new ScaleCamera()
    c.setOrientation(0, 10)
    expect(c.latitude).toBeLessThan(Math.PI / 2)
    expect(c.latitude).toBeGreaterThan(Math.PI / 2 - 1e-3)
  })

  it('reports a unit direction that agrees with worldPosition', () => {
    const c = new ScaleCamera()
    c.setOrientation(2.0, 0.3)
    c.setAltitude(1e6)
    const w = c.worldPosition
    const len = Math.hypot(...w)
    const d = c.direction
    expect(Math.hypot(...d)).toBeCloseTo(1, 12)
    for (let i = 0; i < 3; i++) expect(d[i]).toBeCloseTo(w[i]! / len, 12)
  })
})

describe('one public quality floor', () => {
  // These catch any altitude entry bypassing the quality floor, retaining
  // overscroll debt, or tying the quality limit to the sweep's ground sampler.
  it('bounds direct/scripted altitude requests and the resulting world position', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1_000_000
    c.setAltitude(0.01)
    expect(c.altitude).toBe(1_000_000)
    expect(c.requestedAltitude).toBe(1_000_000)
    expect(Math.hypot(...c.worldPosition) - 6378137).toBeCloseTo(1_000_000, 6)
    expect(c.atQualityFloor).toBe(true)
    c.setAltitude(2_000_000)
    expect(c.atQualityFloor).toBe(false)
  })

  it.each(['wheel', 'pinch'] as const)('%s stays bounded and reverses immediately after repeated overscroll', (input) => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1_000_000
    c.setAltitude(1_010_000)
    const inward = input === 'wheel' ? wheelDeltaToTicks(-100) : pinchDistanceToTicks(100, 109)
    for (let i = 0; i < 100; i++) c.scrollBy(inward)
    expect(c.altitude).toBe(1_000_000)
    expect(c.requestedAltitude).toBe(1_000_000)
    c.scrollBy(-inward)
    expect(c.altitude).toBeCloseTo(1_090_000, 6)
    expect(c.atQualityFloor).toBe(false)
  })

  it('keeps the same limit during playback and forward/backward scrubbing when matter disappears', () => {
    const c = new ScaleCamera()
    const timeline = new Timeline()
    const opening = new OpeningExperience(c, timeline)
    opening.setEssential('density', 'ready')
    opening.setEssential('elevation', 'ready')
    c.qualityFloorM = 1_000_000
    c.groundHeightAt = () => timeline.state.matter > 0 ? 5000 : null
    c.setAltitude(0.01)
    for (const t of [0, 3, 9, 13, 20, 9, 0]) {
      opening.scrubTo(t)
      expect(c.altitude).toBe(1_000_000)
      expect(c.atQualityFloor).toBe(true)
    }
    opening.togglePlayback()
    for (let i = 0; i < 401; i++) {
      opening.advance(0.1)
      expect(c.altitude).toBe(1_000_000)
    }
    expect(timeline.t).toBe(20)
  })

  it('raises the request on resize, but does not auto-descend when the floor later relaxes', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1_000_000
    c.setAltitude(0.01)
    c.qualityFloorM = 2_000_000
    expect(c.altitude).toBe(2_000_000)
    expect(c.requestedAltitude).toBe(2_000_000)
    c.qualityFloorM = 1_000_000
    expect(c.altitude).toBe(2_000_000)
    expect(c.atQualityFloor).toBe(false)
    c.scrollBy(1)
    expect(c.altitude).toBeCloseTo(1_834_862.3853211, 6)
  })

  it('still takes the physical ground maximum when it exceeds the quality floor', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1000
    c.groundHeightAt = () => 5000
    c.setAltitude(0.01)
    expect(c.altitude).toBe(5002)
    expect(c.atQualityFloor).toBe(false)
    c.scrollBy(100)
    c.scrollBy(-1)
    expect(c.altitude).toBeCloseTo(5452.18, 6)
  })

  it('clamps an oversized floor to the outer bound and leaves reverse input usable', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1e12
    c.setAltitude(-1)
    expect(c.altitude).toBe(ALT_MAX)
    c.qualityFloorM = 1_000_000
    c.scrollBy(1)
    expect(c.altitude).toBeCloseTo(36_697_247.706422, 5)
  })

  it('ignores non-finite updates instead of corrupting the current floor or pose', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1_000_000
    c.setAltitude(2_000_000)
    for (const invalid of [NaN, Infinity, -Infinity]) {
      c.qualityFloorM = invalid
      expect(c.qualityFloorM).toBe(1_000_000)
    }
    c.setAltitude(NaN)
    c.scrollBy(NaN)
    expect(c.altitude).toBe(2_000_000)
  })
})

describe('analytic approach to the quality floor', () => {
  // Fixtures integrate known fractions of the band: starting at x = band,
  // four inputs of band/2 yield x/band = 9/16, 1/4, 1/16, 0. This catches
  // a hard clamp, wrong taper, and numerical integration tied to event count.
  const bandTicks = Math.log(1.25) / Math.log(1.09)

  it('preserves ordinary logarithmic scrolling when the whole gesture stays above the band', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1_000_000
    c.setAltitude(2_000_000)
    c.scrollBy(1)
    expect(c.altitude).toBeCloseTo(1_834_862.3853211, 6)
  })

  it('reduces successive inward movements and reaches the exact floor in finite input', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1_000_000
    c.setAltitude(1_250_000)
    let previousAltitude = c.altitude
    let previousMotion = Infinity
    for (const expected of [1_133_735.856746342, 1_057_371.263440564, 1_014_044.177685514, 1_000_000]) {
      c.scrollBy(bandTicks / 2)
      expect(c.altitude).toBeCloseTo(expected, 5)
      const motion = previousAltitude - c.altitude
      expect(motion).toBeGreaterThan(0)
      expect(motion).toBeLessThan(previousMotion)
      expect(c.requestedAltitude).toBe(c.altitude)
      previousMotion = motion
      previousAltitude = c.altitude
    }
    expect(c.altitude).toBe(1_000_000)
    expect(c.atQualityFloor).toBe(true)
    c.scrollBy(100)
    expect(c.requestedAltitude).toBe(1_000_000)
  })

  it.each([
    [1_250_000, bandTicks, 1_057_371.263440564],
    [2_000_000, Math.log(1.6) / Math.log(1.09) + bandTicks, 1_057_371.263440564],
    [2_000_000, 100, 1_000_000],
  ])('one gesture and 100 fragments agree from %s m, including crossing the band', (start, ticks, expected) => {
    const whole = new ScaleCamera()
    const split = new ScaleCamera()
    for (const c of [whole, split]) {
      c.qualityFloorM = 1_000_000
      c.setAltitude(start)
    }
    whole.scrollBy(ticks)
    for (let i = 0; i < 100; i++) split.scrollBy(ticks / 100)
    expect(whole.altitude).toBeCloseTo(expected, 5)
    expect(split.altitude).toBeCloseTo(expected, 5)
    expect(split.altitude).toBeCloseTo(whole.altitude, 5)
  })

  it('reverses at full logarithmic speed while still inside the taper', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1_000_000
    c.setAltitude(1_250_000)
    c.scrollBy(bandTicks)
    expect(c.altitude).toBeCloseTo(1_057_371.263440564, 5)
    c.scrollBy(-1)
    expect(c.altitude).toBeCloseTo(1_152_534.677150215, 5)
  })

  it('does not taper diagnostic microscale scrolling when the quality floor is ALT_MIN', () => {
    const c = new ScaleCamera()
    c.setAltitude(0.0125)
    c.scrollBy(bandTicks)
    expect(c.altitude).toBe(ALT_MIN)
  })

  it('ignores NaN mid-approach without changing subsequent movement', () => {
    const c = new ScaleCamera()
    c.qualityFloorM = 1_000_000
    c.setAltitude(1_250_000)
    c.scrollBy(NaN)
    expect(c.altitude).toBe(1_250_000)
    c.scrollBy(bandTicks)
    expect(c.altitude).toBeCloseTo(1_057_371.263440564, 5)
  })
})
