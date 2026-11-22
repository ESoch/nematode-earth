import { describe, expect, it } from 'vitest'
import { Timeline, T_END } from '../src/sweep/Timeline'
import { playbackDelta, playbackSeconds } from '../src/sweep/pacing'

describe('a guided sweep that spends its time on the visible layers', () => {
  it.each([30, 60, 120])('gives the erosion room, then settles promptly at %i Hz', (hz) => {
    const timeline = new Timeline()
    timeline.play()
    for (let i = 0; i < 4 * hz; i++) timeline.advance(playbackDelta(1 / hz, timeline.t))
    expect(timeline.t).toBeCloseTo(3, 8)
    expect(timeline.playing).toBe(true)
    for (let i = 0; i < 24 * hz; i++) timeline.advance(playbackDelta(1 / hz, timeline.t))
    expect(timeline.t).toBeCloseTo(9, 8) // Last solid-Earth fragments are gone.
    expect(playbackSeconds(timeline.t)).toBeCloseTo(28, 8)
    for (let i = 0; i < 5.5 * hz; i++) timeline.advance(playbackDelta(1 / hz, timeline.t))
    expect(timeline.t).toBeCloseTo(T_END, 8)
    expect(timeline.playing).toBe(false)
  })

  it('keeps direct scrubbing immediate and maps the actual phase boundaries', () => {
    const timeline = new Timeline()
    expect(playbackSeconds(3)).toBeCloseTo(4)
    expect(playbackSeconds(9)).toBeCloseTo(28)
    timeline.scrubTo(T_END)
    expect(timeline.state.matter).toBe(0)
    expect(playbackSeconds(timeline.t)).toBeCloseTo(33.5)
  })

  it('does not skip the reveal when a background tab resumes', () => {
    expect(playbackDelta(100, 0)).toBeCloseTo(0.1125)
  })

  it('crosses the last-layer boundary without a one-frame pacing jump', () => {
    expect(playbackDelta(0.15, 8.99)).toBeCloseTo(0.23, 8)
  })
})
