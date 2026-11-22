import { describe, it, expect } from 'vitest'
import { pinchDistanceToTicks, dragToOrbit, wheelDeltaToTicks } from '../src/core/orbitInput'
import { PointerGesture } from '../src/core/pointerGesture'
import { ScaleCamera, ALT_RATIO_PER_TICK, GROUND_CLEARANCE_M } from '../src/core/ScaleCamera'
import type { Vec3 } from '../src/math/raySpheroid'

/**
 * Touch descent.
 *
 * `viewer.ts` bound descent to `wheel` alone and tracked exactly one pointer,
 * so on a phone the interaction the whole piece is built around — descending
 * through the scale journey — did not exist. Visitors could spin the globe
 * and nothing else.
 *
 * The assertions below are deliberately about the ALTITUDE, not about whether
 * a handler ran: this project has repeatedly shipped defects that were
 * invisible rather than broken, and "the pinch handler fired" is exactly the
 * kind of check that passes while the camera sits still. Every gesture here
 * is driven into a real `ScaleCamera` and the altitude is compared against a
 * figure predicted in advance.
 */

/** Drives a gesture's output into a camera exactly as viewer.ts does. */
function apply(cam: ScaleCamera, g: PointerGesture, out: ReturnType<PointerGesture['move']>): void {
  if (out.ticks !== 0) cam.scrollBy(out.ticks)
  if (out.dxPx !== 0 || out.dyPx !== 0) {
    const { dLon, dLat } = dragToOrbit(out.dxPx, out.dyPx, 800)
    cam.orbitBy(dLon, dLat)
  }
  void g
}

describe('pinchDistanceToTicks', () => {
  it('descends when the fingers spread apart', () => {
    expect(pinchDistanceToTicks(100, 200)).toBeGreaterThan(0)
  })

  it('ascends when the fingers close together', () => {
    expect(pinchDistanceToTicks(200, 100)).toBeLessThan(0)
  })

  it('is zero when the separation does not change', () => {
    expect(pinchDistanceToTicks(137, 137)).toBeCloseTo(0, 12)
  })

  it('maps a doubling of separation onto an exact halving of altitude', () => {
    // The gesture's contract, in one line: the world scales with the fingers.
    const ticks = pinchDistanceToTicks(100, 200)
    expect(Math.pow(ALT_RATIO_PER_TICK, ticks)).toBeCloseTo(2, 9)
  })

  it('depends only on the RATIO, so the same spread reads the same anywhere on screen', () => {
    expect(pinchDistanceToTicks(50, 100)).toBeCloseTo(pinchDistanceToTicks(400, 800), 12)
  })

  it('composes: two half-gestures equal one whole one', () => {
    const whole = pinchDistanceToTicks(100, 400)
    const halves = pinchDistanceToTicks(100, 200) + pinchDistanceToTicks(200, 400)
    expect(halves).toBeCloseTo(whole, 12)
  })

  it('is exactly invertible, so a pinch out and back leaves the altitude where it started', () => {
    expect(pinchDistanceToTicks(100, 250) + pinchDistanceToTicks(250, 100)).toBeCloseTo(0, 12)
  })

  it('yields no motion for degenerate separations rather than an infinity', () => {
    for (const [a, b] of [
      [0, 100],
      [100, 0],
      [0, 0],
      [-5, 100],
      [100, NaN],
      [NaN, 100],
    ]) {
      expect(pinchDistanceToTicks(a!, b!), `${a} -> ${b}`).toBe(0)
    }
  })

  it('shares the wheel path exactly: equal tick counts move the camera equally', () => {
    // The requirement is that touch and wheel cannot diverge. Both produce
    // ticks; ticks are the only thing either one hands the camera.
    const byWheel = new ScaleCamera()
    const byPinch = new ScaleCamera()
    byWheel.setAltitude(1e6)
    byPinch.setAltitude(1e6)
    const ticks = pinchDistanceToTicks(120, 340)
    byPinch.scrollBy(ticks)
    byWheel.scrollBy(wheelDeltaToTicks(-100 * ticks))
    expect(byPinch.altitude).toBeCloseTo(byWheel.altitude, 6)
  })
})

describe('one finger orbits, exactly as one mouse button always did', () => {
  it('reports the drag delta of a single pointer and no altitude change', () => {
    const g = new PointerGesture()
    g.down({ id: 1, x: 100, y: 100 })
    const out = g.move({ id: 1, x: 140, y: 90 })
    expect(out).toEqual({ dxPx: 40, dyPx: -10, ticks: 0 })
  })

  it('leaves the altitude untouched across a long single-finger drag', () => {
    const cam = new ScaleCamera()
    cam.setAltitude(5e6)
    const g = new PointerGesture()
    g.down({ id: 1, x: 100, y: 300 })
    for (let i = 1; i <= 40; i++) apply(cam, g, g.move({ id: 1, x: 100 + i * 7, y: 300 - i * 3 }))
    expect(cam.altitude).toBeCloseTo(5e6, 6)
    expect(cam.longitude).not.toBeCloseTo(0, 6)
  })

  it('ignores movement from a pointer that never went down (a hovering mouse)', () => {
    const g = new PointerGesture()
    expect(g.move({ id: 9, x: 10, y: 10 })).toEqual({
      dxPx: 0,
      dyPx: 0,
      ticks: 0,
    })
  })
})

describe('two fingers descend', () => {
  it('halves the altitude when the fingers spread to twice their separation', () => {
    const cam = new ScaleCamera()
    cam.setAltitude(1e7)
    const g = new PointerGesture()
    // Fingers 200 px apart, spreading to 400 px.
    g.down({ id: 1, x: 400, y: 500 })
    g.down({ id: 2, x: 600, y: 500 })
    apply(cam, g, g.move({ id: 2, x: 800, y: 500 }))
    expect(cam.altitude).toBeCloseTo(5e6, 3)
  })

  it('quadruples the altitude when the fingers close to a quarter of their separation', () => {
    const cam = new ScaleCamera()
    cam.setAltitude(1e6)
    const g = new PointerGesture()
    g.down({ id: 1, x: 100, y: 500 })
    g.down({ id: 2, x: 900, y: 500 })
    apply(cam, g, g.move({ id: 2, x: 300, y: 500 }))
    expect(cam.altitude).toBeCloseTo(4e6, 2)
  })

  it('reaches the same altitude whether the spread arrives in one step or fifty', () => {
    const target = (steps: number): number => {
      const cam = new ScaleCamera()
      cam.setAltitude(1e7)
      const g = new PointerGesture()
      g.down({ id: 1, x: 500, y: 500 })
      g.down({ id: 2, x: 700, y: 500 })
      for (let i = 1; i <= steps; i++) {
        apply(cam, g, g.move({ id: 2, x: 700 + (600 * i) / steps, y: 500 }))
      }
      return cam.altitude
    }
    expect(target(50)).toBeCloseTo(target(1), 3)
  })

  it('measures separation in both axes, not just horizontally', () => {
    const cam = new ScaleCamera()
    cam.setAltitude(1e7)
    const g = new PointerGesture()
    g.down({ id: 1, x: 500, y: 500 })
    g.down({ id: 2, x: 500, y: 700 }) // 200 px apart, vertically
    apply(cam, g, g.move({ id: 2, x: 500, y: 900 }))
    expect(cam.altitude).toBeCloseTo(5e6, 3)
  })

  it('emits no orbit while two fingers are down, so a pinch does not spin the globe', () => {
    const g = new PointerGesture()
    g.down({ id: 1, x: 400, y: 500 })
    g.down({ id: 2, x: 600, y: 500 })
    const out = g.move({ id: 1, x: 300, y: 560 })
    expect(out.dxPx).toBe(0)
    expect(out.dyPx).toBe(0)
    expect(out.ticks).not.toBe(0)
  })

  it('is driven by whichever of the two fingers moves', () => {
    const cam = new ScaleCamera()
    cam.setAltitude(1e7)
    const g = new PointerGesture()
    g.down({ id: 1, x: 400, y: 500 })
    g.down({ id: 2, x: 600, y: 500 })
    apply(cam, g, g.move({ id: 1, x: 200, y: 500 })) // the FIRST finger spreads
    expect(cam.altitude).toBeCloseTo(5e6, 3)
  })
})

describe('a pointer arriving or leaving must not jump the camera', () => {
  it('does not descend on the frame the second finger lands', () => {
    const cam = new ScaleCamera()
    cam.setAltitude(1e7)
    const g = new PointerGesture()
    g.down({ id: 1, x: 400, y: 500 })
    apply(cam, g, g.move({ id: 1, x: 420, y: 500 }))
    g.down({ id: 2, x: 900, y: 500 })
    expect(cam.altitude).toBeCloseTo(1e7, 6)
  })

  it('does not orbit on the frame a finger lifts, however far apart the two were', () => {
    // The defect this pins: keeping ONE `lastClientX/Y` and letting the
    // surviving pointer's first move measure from the lifted pointer's last
    // position. Fingers 600 px apart would snap the globe 600 px of orbit in
    // a single frame.
    const cam = new ScaleCamera()
    cam.setAltitude(1e6)
    const g = new PointerGesture()
    g.down({ id: 1, x: 200, y: 500 })
    g.down({ id: 2, x: 800, y: 500 })
    g.up(2)
    const lonBefore = cam.longitude
    const out = g.move({ id: 1, x: 203, y: 500 })
    apply(cam, g, out)
    expect(out.dxPx).toBe(3)
    expect(Math.abs(cam.longitude - lonBefore)).toBeLessThan(dragToOrbit(4, 0, 800).dLon)
  })

  it('does not descend when a THIRD finger lands or lifts mid-pinch', () => {
    const cam = new ScaleCamera()
    cam.setAltitude(1e7)
    const g = new PointerGesture()
    g.down({ id: 1, x: 400, y: 500 })
    g.down({ id: 2, x: 600, y: 500 })
    g.down({ id: 3, x: 100, y: 100 })
    expect(cam.altitude).toBeCloseTo(1e7, 6)
    g.up(3)
    expect(cam.altitude).toBeCloseTo(1e7, 6)
    // and the surviving pair still pinches from where it actually is
    apply(cam, g, g.move({ id: 2, x: 800, y: 500 }))
    expect(cam.altitude).toBeCloseTo(5e6, 3)
  })

  it('re-seats onto the surviving pair when one of the pinching fingers lifts', () => {
    const cam = new ScaleCamera()
    cam.setAltitude(1e7)
    const g = new PointerGesture()
    g.down({ id: 1, x: 500, y: 500 })
    g.down({ id: 2, x: 700, y: 500 }) // pinch pair: 1 and 2, 200 apart
    g.down({ id: 3, x: 1100, y: 500 })
    g.up(1) // pair becomes 2 and 3, 400 apart — no altitude change
    expect(cam.altitude).toBeCloseTo(1e7, 6)
    apply(cam, g, g.move({ id: 3, x: 1500, y: 500 })) // 400 -> 800
    expect(cam.altitude).toBeCloseTo(5e6, 3)
  })

  it('hands orbit back to the finger still on the glass, from where that finger is', () => {
    const g = new PointerGesture()
    g.down({ id: 1, x: 400, y: 500 })
    g.down({ id: 2, x: 600, y: 500 })
    g.move({ id: 1, x: 300, y: 520 }) // moves during the pinch; position must be tracked
    g.up(2)
    expect(g.move({ id: 1, x: 310, y: 530 })).toEqual({
      dxPx: 10,
      dyPx: 10,
      ticks: 0,
    })
  })

  it('forgets everything on cancel, and starts clean afterwards', () => {
    const g = new PointerGesture()
    g.down({ id: 1, x: 400, y: 500 })
    g.down({ id: 2, x: 600, y: 500 })
    g.clear()
    expect(g.pointerCount).toBe(0)
    expect(g.move({ id: 1, x: 999, y: 999 })).toEqual({
      dxPx: 0,
      dyPx: 0,
      ticks: 0,
    })
    g.down({ id: 7, x: 100, y: 100 })
    expect(g.move({ id: 7, x: 105, y: 100 })).toEqual({
      dxPx: 5,
      dyPx: 0,
      ticks: 0,
    })
  })

  it('tolerates an up for a pointer it never saw', () => {
    const g = new PointerGesture()
    g.down({ id: 1, x: 10, y: 10 })
    expect(() => g.up(99)).not.toThrow()
    expect(g.pointerCount).toBe(1)
  })
})

describe('pinch respects the ground floor the wheel already respects', () => {
  it('cannot pinch the camera inside the displaced mesh', () => {
    // ScaleCamera's floor lives on the altitude path, and pinch reaches the
    // camera only through that path — so this holds by construction, and it
    // is asserted because "by construction" is what stops being true first.
    const cam = new ScaleCamera()
    const terrainM = 8000
    cam.groundHeightAt = (_dir: Vec3) => terrainM
    cam.setAltitude(1e5)
    const g = new PointerGesture()
    g.down({ id: 1, x: 100, y: 500 })
    g.down({ id: 2, x: 110, y: 500 })
    // A ferocious spread: 10 px to 4000 px, far past the floor.
    apply(cam, g, g.move({ id: 2, x: 4100, y: 500 }))
    expect(cam.altitude).toBeGreaterThanOrEqual(terrainM + GROUND_CLEARANCE_M)
    expect(cam.altitude).toBeCloseTo(terrainM + GROUND_CLEARANCE_M, 6)
  })

  it('lets a pinch inward lift straight back off the floor', () => {
    const cam = new ScaleCamera()
    cam.groundHeightAt = (_dir: Vec3) => 8000
    cam.setAltitude(1)
    const g = new PointerGesture()
    g.down({ id: 1, x: 100, y: 500 })
    g.down({ id: 2, x: 900, y: 500 })
    apply(cam, g, g.move({ id: 2, x: 300, y: 500 })) // 800 -> 200, ascend 4x
    expect(cam.altitude).toBeCloseTo((8000 + GROUND_CLEARANCE_M) * 4, 3)
  })
})

describe('the whole descent is reachable by touch alone', () => {
  it("crosses the piece's scale journey in a plausible number of pinches", () => {
    // The opening pose is 11,500 km. A comfortable phone pinch spreads the
    // fingers about 3x. If the journey needed hundreds of them, touch descent
    // would exist and still be unusable.
    const cam = new ScaleCamera()
    cam.setAltitude(1.15e7)
    const g = new PointerGesture()
    let pinches = 0
    while (cam.altitude > 1000 && pinches < 500) {
      g.clear()
      g.down({ id: 1, x: 500, y: 500 })
      g.down({ id: 2, x: 560, y: 500 })
      apply(cam, g, g.move({ id: 2, x: 680, y: 500 })) // 60 -> 180 px, a 3x spread
      pinches++
    }
    expect(cam.altitude).toBeLessThanOrEqual(1000)
    expect(pinches).toBeLessThanOrEqual(12)
  })
})
