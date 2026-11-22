import { describe, expect, it } from 'vitest'
import { stateAt, Timeline } from '../src/sweep/Timeline'
import { sweepGuideAt, sweepGuidePresentationAt } from '../src/sweep/guide'

describe('sweep guide follows the rendered layer fronts', () => {
  it.each([
    [0, 'ready', ['intact', 'intact', 'intact']],
    [3, 'ready', ['intact', 'intact', 'intact']],
    [4, 'atmosphere', ['leaving', 'leaving', 'intact']],
    [5, 'atmosphere', ['leaving', 'leaving', 'intact']],
    [5.5, 'water', ['gone', 'leaving', 'intact']],
    [7, 'water', ['gone', 'leaving', 'leaving']],
    [7.4, 'rock', ['gone', 'gone', 'leaving']],
    [9, 'remains', ['gone', 'gone', 'gone']],
    [20, 'remains', ['gone', 'gone', 'gone']],
  ] as const)('at authored time %s shows %s and all overlapping layers', (time, phase, statuses) => {
    const guide = sweepGuideAt(stateAt(time))
    expect(guide.phase).toBe(phase)
    expect(guide.layers.map((layer) => layer.status)).toEqual(statuses)
  })

  it('counts the visible ocean descent as receding before water starts dissolving', () => {
    const state = stateAt(4)
    expect(state.frontWater).toBe(0)
    expect(state.bathymetry).toBeGreaterThan(0)
    expect(sweepGuideAt(state).layers[1]?.status).toBe('leaving')
  })

  it('does not wait for the nominal matter slider or old chapter to finish', () => {
    const state = stateAt(9)
    expect(state.matter).toBeGreaterThan(0)
    expect(state.beat).toBe('collapse')
    expect(sweepGuideAt(state).phase).toBe('remains')
  })

  it('reconstructs status while scrubbing backward without sticky completed markers', () => {
    const timeline = new Timeline()
    const phases = [20, 8, 6, 4, 0].map((time) => {
      timeline.scrubTo(time)
      return sweepGuideAt(timeline.state).phase
    })
    expect(phases).toEqual(['remains', 'rock', 'water', 'atmosphere', 'ready'])
    expect(sweepGuideAt(timeline.state).layers.every((layer) => layer.status === 'intact')).toBe(true)
  })

  it('uses the shader inputs even if the authored clock is inconsistent', () => {
    const state = { ...stateAt(0), frontAtmosphere: 1, frontWater: 1, frontRock: 0.5 }
    expect(sweepGuideAt(state).phase).toBe('rock')
    expect(sweepGuideAt(state).layers.map((layer) => layer.status)).toEqual(['gone', 'gone', 'leaving'])
  })
})

describe('opening guide tells the premise before it names the layers', () => {
  it('shows a sourced Cobb excerpt over the intact Earth while waiting', () => {
    expect(sweepGuidePresentationAt(stateAt(0), true)).toMatchObject({
      mode: 'intro',
      announce: false,
      excerpt: 'our world would still be dimly recognizable',
      sourceHref: 'https://archive.org/details/yoa1914/page/472/mode/2up',
    })
  })

  it('reveals layer status when the visitor starts or explores', () => {
    expect(sweepGuidePresentationAt(stateAt(0), false)).toMatchObject({
      mode: 'layers', phase: 'ready', announce: true,
    })
  })

  it('never lets a stale opening flag cover a layer already being swept away', () => {
    expect(sweepGuidePresentationAt(stateAt(4), true)).toMatchObject({
      mode: 'layers', phase: 'atmosphere',
    })
  })
})
