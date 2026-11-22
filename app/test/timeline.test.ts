import { describe, it, expect } from 'vitest'
import {
  BEATS, T_END, SWEEP_T0, SWEEP_T1, COBB_LINE, SWEEP_STATE_KEYS,
  beatAt, matterAt, timeForMatter, stateAt, smootherstep01, Timeline,
  clampTimelineDt, MAX_TIMELINE_DT_SEC,
} from '../src/sweep/Timeline'

const samples = (n = 2001): number[] => Array.from({ length: n }, (_, i) => (i * T_END) / (n - 1))

describe('beats', () => {
  it('matches spec 6.1 exactly', () => {
    expect(BEATS.map((b) => [b.name, b.t0, b.t1])).toEqual([
      ['cold-open', 0, 3],
      ['erosion', 3, 9],
      ['collapse', 9, 13],
      ['what-remains', 13, 18],
      ['release', 18, 20],
    ])
  })
  it('tiles [0,20] with no gap and no overlap', () => {
    for (let i = 1; i < BEATS.length; i++) expect(BEATS[i]!.t0).toBe(BEATS[i - 1]!.t1)
    expect(BEATS[0]!.t0).toBe(0)
    expect(BEATS.at(-1)!.t1).toBe(T_END)
  })
  it('resolves boundaries to the later beat and clamps outside', () => {
    expect(beatAt(0).beat).toBe('cold-open')
    expect(beatAt(3).beat).toBe('erosion')
    expect(beatAt(9).beat).toBe('collapse')
    expect(beatAt(13).beat).toBe('what-remains')
    expect(beatAt(18).beat).toBe('release')
    expect(beatAt(-5).beat).toBe('cold-open')
    expect(beatAt(99).beat).toBe('release')
    expect(beatAt(99).beatProgress).toBe(1)
  })
})

describe('matter', () => {
  it('is 1 through the cold open and 0 from the end of the collapse', () => {
    expect(matterAt(0)).toBe(1)
    expect(matterAt(SWEEP_T0)).toBe(1)
    expect(matterAt(SWEEP_T1)).toBe(0)
    expect(matterAt(T_END)).toBe(0)
  })
  it('is linear in t across the sweep span so the slider is uniform', () => {
    expect(matterAt(8)).toBeCloseTo(0.5, 12)
    expect(matterAt(5.5)).toBeCloseTo(0.75, 12)
  })
  it('never increases', () => {
    let prev = Infinity
    for (const t of samples()) {
      const m = matterAt(t)
      expect(m).toBeLessThanOrEqual(prev + 1e-12)
      prev = m
    }
  })
  it('round-trips through timeForMatter inside the span', () => {
    for (const t of [3, 4.25, 6, 8, 10.5, 13]) {
      expect(timeForMatter(matterAt(t))).toBeCloseTo(t, 9)
    }
  })
  it('clamps the round trip outside the span', () => {
    expect(timeForMatter(matterAt(1))).toBe(SWEEP_T0)
    expect(timeForMatter(matterAt(17))).toBe(SWEEP_T1)
    expect(timeForMatter(2)).toBe(SWEEP_T0)
    expect(timeForMatter(-1)).toBe(SWEEP_T1)
  })
})

describe('erosion fronts', () => {
  it('are all 0 before the erosion beat and all 1 after the collapse', () => {
    for (const t of [0, 1.5, 3]) {
      const s = stateAt(t)
      expect(s.frontAtmosphere).toBe(0)
      expect(s.frontWater).toBe(0)
      expect(s.frontRock).toBe(0)
      expect(s.bathymetry).toBe(0)
    }
    for (const t of [13, 15, 20]) {
      const s = stateAt(t)
      expect(s.frontAtmosphere).toBe(1)
      expect(s.frontWater).toBe(1)
      expect(s.frontRock).toBe(1)
      expect(s.bathymetry).toBe(1)
    }
  })
  it('stay ordered bathymetry >= atmosphere >= water >= rock at every t', () => {
    // Retimed (was atmosphere >= water >= rock >= bathymetry): bathymetry
    // now moves FIRST, not last - see Matter.ts's OCEAN COLLAPSE section
    // doc. The sea floor's drop is only visible while the ocean surface is
    // still opaque, and water's own discard front is exactly 0 (full
    // opacity everywhere, not asymptotically - see layerAlpha's
    // endpoint-totality doc in Dissolve.ts) only up to its own window's
    // start, so bathymetry has to finish its ramp at or before that point.
    for (const t of samples()) {
      const s = stateAt(t)
      expect(s.bathymetry).toBeGreaterThanOrEqual(s.frontAtmosphere - 1e-12)
      expect(s.frontAtmosphere).toBeGreaterThanOrEqual(s.frontWater - 1e-12)
      expect(s.frontWater).toBeGreaterThanOrEqual(s.frontRock - 1e-12)
    }
  })
  it('each front is monotone non-decreasing', () => {
    const keys = ['frontAtmosphere', 'frontWater', 'frontRock', 'bathymetry'] as const
    for (const k of keys) {
      let prev = -Infinity
      for (const t of samples()) {
        const v = stateAt(t)[k]
        expect(v).toBeGreaterThanOrEqual(prev - 1e-12)
        prev = v
      }
    }
  })
  it('hits the spec 6.1 stagger: bathymetry drops 3-4.6 s (before water moves at all), atmosphere done at 5.4 s, water spans 4.6-7.4, rock 6.4-9.0', () => {
    expect(stateAt(3.0).bathymetry).toBeCloseTo(0, 9)
    expect(stateAt(4.6).bathymetry).toBeCloseTo(1, 9)
    expect(stateAt(5.4).frontAtmosphere).toBeCloseTo(1, 9)
    expect(stateAt(5.39).frontAtmosphere).toBeLessThan(1)
    expect(stateAt(4.6).frontWater).toBeCloseTo(0, 9)
    expect(stateAt(7.4).frontWater).toBeCloseTo(1, 9)
    expect(stateAt(6.4).frontRock).toBeCloseTo(0, 9)
    expect(stateAt(9.0).frontRock).toBeCloseTo(1, 9)
  })
  it('the sea floor finishes dropping exactly as the water front starts to move, not nine seconds later', () => {
    // The bug this pins: bathymetry used to ramp 9-13s, entirely AFTER
    // water (4.6-7.4s) and rock (6.4-9.0s) had already finished eroding
    // network-wide - by t=9s nothing of Matter is even on screen any more
    // (both fronts are saturated, and layerAlpha is EXACTLY 0 everywhere
    // once its front reaches 1 - see Dissolve.ts's doc), so a drop timed
    // there could never be seen no matter how the shader read it. Now
    // bathymetry completes (reaches 1) at the exact instant frontWater
    // leaves ITS OWN start (0): frontNorm=0 means water's alpha is exactly
    // 1 everywhere, not asymptotically, so the ocean is still fully opaque
    // through this entire handoff.
    const t = 4.6
    expect(stateAt(t).bathymetry).toBeCloseTo(1, 9)
    expect(stateAt(t).frontWater).toBeCloseTo(0, 9)
  })
})

describe('the state carries no veil parameter', () => {
  it('has exactly the declared key set', () => {
    expect(Object.keys(stateAt(7)).sort()).toEqual([...SWEEP_STATE_KEYS].sort())
  })
  it('names nothing optical', () => {
    const forbidden = /sigma|tau|opacity_?veil|veil|Hs|Hmax|density|alpha/i
    for (const k of SWEEP_STATE_KEYS) {
      if (k === 'uiOpacity') continue
      expect(k).not.toMatch(forbidden)
    }
  })
})

describe('camera and titles', () => {
  it('never descends below 1e5 m - this session renders in the planet frame', () => {
    for (const t of samples()) expect(stateAt(t).cameraAltitudeM).toBeGreaterThan(1e5)
  })
  it('decelerates during the release beat', () => {
    expect(stateAt(20).cameraSpinRadPerSec).toBeLessThan(stateAt(18).cameraSpinRadPerSec)
  })
  it('types Cobb in during the cold open and completes before the erosion', () => {
    expect(stateAt(0).titleChars).toBe(0)
    expect(stateAt(1.5).titleChars).toBeGreaterThan(0)
    expect(stateAt(2.8).titleChars).toBe(COBB_LINE.length)
  })
  it('fades the controls in only at release', () => {
    expect(stateAt(17).uiOpacity).toBe(0)
    expect(stateAt(20).uiOpacity).toBe(1)
  })
})

describe('smootherstep01', () => {
  it('clamps and is C2 at the ends', () => {
    expect(smootherstep01(-1)).toBe(0)
    expect(smootherstep01(2)).toBe(1)
    expect(smootherstep01(0.5)).toBeCloseTo(0.5, 12)
  })
})

describe('Timeline', () => {
  it('advances only while playing and stops at the end', () => {
    const tl = new Timeline()
    tl.advance(1)
    expect(tl.t).toBe(0)
    tl.play()
    tl.advance(1.5)
    expect(tl.t).toBeCloseTo(1.5, 12)
    tl.advance(100)
    expect(tl.t).toBe(T_END)
    expect(tl.playing).toBe(false)
  })
  it('scrubbing pauses and clamps', () => {
    const tl = new Timeline()
    tl.play()
    tl.scrubTo(-4)
    expect(tl.t).toBe(0)
    expect(tl.playing).toBe(false)
    tl.scrubTo(999)
    expect(tl.t).toBe(T_END)
  })
  it('setMatter and the state agree - the slider drives the same variable', () => {
    const tl = new Timeline()
    tl.setMatter(0.25)
    expect(tl.state.matter).toBeCloseTo(0.25, 9)
    expect(tl.t).toBeCloseTo(timeForMatter(0.25), 9)
  })
  it('play() at the end restarts rather than sitting inert', () => {
    // Without a restart, play() at t===T_END sets playing=true and the very
    // next advance() immediately re-clamps to T_END and pauses again - the
    // Play button would then do nothing once the sweep has finished.
    const tl = new Timeline()
    tl.scrubTo(T_END)
    tl.play()
    expect(tl.t).toBe(0)
    expect(tl.playing).toBe(true)
    tl.advance(1)
    expect(tl.t).toBeCloseTo(1, 12)
    expect(tl.playing).toBe(true)
  })
  it('play() mid-sweep does NOT restart - only end-of-timeline does', () => {
    const tl = new Timeline()
    tl.scrubTo(5)
    tl.play()
    expect(tl.t).toBe(5)
  })
  it('the Play button (toggle) restarts at the end too, not just play() directly', () => {
    const tl = new Timeline()
    tl.scrubTo(T_END)
    tl.toggle() // this is what the Play button actually calls
    expect(tl.t).toBe(0)
    expect(tl.playing).toBe(true)
  })
})

describe('clampTimelineDt', () => {
  it('passes a normal frame dt through unchanged', () => {
    expect(clampTimelineDt(1 / 60)).toBeCloseTo(1 / 60, 12)
    expect(clampTimelineDt(0.1)).toBeCloseTo(0.1, 12)
  })
  it('caps a large dt (backgrounded tab, or a mid-sweep material rebuild stall)', () => {
    expect(clampTimelineDt(100)).toBe(MAX_TIMELINE_DT_SEC)
    expect(clampTimelineDt(5)).toBe(MAX_TIMELINE_DT_SEC)
  })
  it('is a no-op at exactly the cap', () => {
    expect(clampTimelineDt(MAX_TIMELINE_DT_SEC)).toBe(MAX_TIMELINE_DT_SEC)
  })
  it('prevents a single oversized dt from jumping the timeline straight to T_END', () => {
    // This is the actual bug: backgrounding the tab mid-play (or the
    // material rebuild when the 48 MiB elevation tier lands mid-sweep)
    // hands the render loop a multi-second wall-clock dt on the next
    // frame; feeding that straight into advance() jumps to T_END in one
    // step instead of resuming a beat late.
    const tl = new Timeline()
    tl.play()
    tl.advance(clampTimelineDt(100)) // e.g. a 100s tab-backgrounding stall
    expect(tl.t).toBeCloseTo(MAX_TIMELINE_DT_SEC, 12)
    expect(tl.playing).toBe(true) // still mid-sweep, not skipped to the end
  })
})
