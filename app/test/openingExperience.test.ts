import { describe, expect, it } from 'vitest'
import { ScaleCamera } from '../src/core/ScaleCamera'
import { OpeningExperience, openingControlLayout } from '../src/core/openingExperience'
import { Timeline } from '../src/sweep/Timeline'

function setup(ready = false) {
  const camera = new ScaleCamera()
  const timeline = new Timeline()
  const opening = new OpeningExperience(camera, timeline)
  if (ready) {
    opening.setEssential('density', 'ready')
    opening.setEssential('elevation', 'ready')
  }
  return { camera, timeline, opening }
}

const pose = (camera: ScaleCamera) => [camera.longitude, camera.latitude, camera.requestedAltitude]

describe('opening controls leave room for the globe', () => {
  it.each([[320, 568, true], [390, 844, false], [844, 390, false]])(
    'moves actions into the existing menu at %sx%s without a ready status row', (width, height, compactGuide) => {
      const { opening } = setup(true)
      expect(openingControlLayout(width, height, opening.readiness.canBegin)).toEqual({
        actions: 'menu', showStatus: false, compactGuide,
      })
      // Readiness copy stays available to aria-describedby and the live region.
      expect(opening.readiness.message).toMatch(/ready/i)
    },
  )

  it('keeps loading and failure explanations visible even on the shortest phone', () => {
    const { opening } = setup()
    for (const state of ['loading', 'error'] as const) {
      opening.setEssential('density', state)
      expect(openingControlLayout(320, 568, opening.readiness.canBegin)).toEqual({
        actions: 'menu', showStatus: true, compactGuide: true,
      })
      expect(opening.readiness.message).toMatch(state === 'error' ? /failed/i : /loading/i)
    }
  })

  it('restores dock actions on desktop while omitting redundant ready text', () => {
    expect(openingControlLayout(1440, 900, true)).toEqual({
      actions: 'dock', showStatus: false, compactGuide: false,
    })
  })
})

describe('stable opening and deliberate exploration', () => {
  it('gives the waiting Earth a perceptible 12-minute idle turn, then holds the view during playback', () => {
    const { camera, timeline, opening } = setup()
    expect(camera.longitude * 180 / Math.PI).toBeCloseTo(-10)
    expect(camera.latitude * 180 / Math.PI).toBeCloseTo(10)
    expect(camera.altitude).toBe(11_500_000)
    const initial = pose(camera)
    const initialLongitude = camera.longitude
    expect(opening.autoRotate).toBe(true)
    for (let i = 0; i < 150; i++) opening.advance(0.1)
    expect(camera.longitude - initialLongitude).toBeCloseTo(Math.PI / 24, 3)
    expect(camera.latitude).toBe(initial[1])
    expect(camera.requestedAltitude).toBe(initial[2])
    expect(timeline.t).toBe(0)
    opening.setEssential('density', 'ready')
    opening.setEssential('elevation', 'ready')
    opening.togglePlayback()
    expect(opening.autoRotate).toBe(false)
    const playbackPose = pose(camera)
    for (let i = 0; i < 200; i++) opening.advance(0.1)
    expect(timeline.t).toBeCloseTo(7)
    for (let i = 0; i < 201; i++) opening.advance(0.1)
    expect(timeline.t).toBe(20)
    expect(pose(camera)).toEqual(playbackPose)
  })

  it('does not jump around the globe after a backgrounded frame', () => {
    const { camera, opening } = setup()
    expect(opening.autoRotate).toBe(true)
    const start = camera.longitude
    opening.advance(60)
    expect(camera.longitude - start).toBeGreaterThan(0)
    expect(camera.longitude - start).toBeLessThan(0.002)
  })

  it('keeps the inspiration visible through loading, then yields on play or exploration', () => {
    const { opening } = setup()
    expect(opening.showIntro).toBe(true)
    opening.togglePlayback() // Still waiting for essential data.
    expect(opening.showIntro).toBe(true)
    opening.setEssential('density', 'ready')
    opening.setEssential('elevation', 'ready')
    expect(opening.showIntro).toBe(true)
    opening.togglePlayback()
    expect(opening.showIntro).toBe(false)
    const another = setup(true).opening
    another.explore()
    expect(another.showIntro).toBe(false)
  })

  it('yields immediately on interaction, retaining user orbit, zoom and time', () => {
    const { camera, timeline, opening } = setup(true)
    opening.togglePlayback()
    opening.advance(0.1)
    opening.setAutoRotate(true) // Even explicitly enabled diagnostics must yield.
    opening.explore()
    expect(timeline.playing).toBe(false)
    expect(opening.autoRotate).toBe(false)
    camera.orbitBy(0.4, -0.2)
    camera.scrollBy(8)
    const userPose = pose(camera)
    const t = timeline.t
    for (let i = 0; i < 100; i++) opening.advance(0.1)
    expect(pose(camera)).toEqual(userPose)
    expect(timeline.t).toBe(t)
    opening.togglePlayback()
    opening.advance(0.1)
    expect(pose(camera)).toEqual(userPose)
    expect(timeline.t).toBeGreaterThan(t)
  })

  it('returns only the pose, pausing at the current scrubbed time', () => {
    const { camera, timeline, opening } = setup(true)
    const initial = pose(camera)
    opening.scrubTo(7.4)
    camera.setOrientation(2, -0.5)
    camera.setAltitude(800)
    opening.togglePlayback()
    opening.returnToOpening()
    expect(pose(camera)).toEqual(initial)
    expect(timeline.t).toBe(7.4)
    expect(timeline.playing).toBe(false)
  })

  it('replays intentionally from the opening pose and beginning', () => {
    const { camera, timeline, opening } = setup(true)
    const initial = pose(camera)
    opening.scrubTo(11)
    camera.setAltitude(1000)
    opening.replay()
    expect(pose(camera)).toEqual(initial)
    expect(timeline.t).toBe(0)
    expect(timeline.playing).toBe(true)
    opening.scrubTo(20)
    camera.setAltitude(2000)
    opening.togglePlayback() // Watch again uses the same deliberate replay.
    expect(pose(camera)).toEqual(initial)
    expect(timeline.t).toBe(0)
  })

  it('keeps reduced motion stable, allows explicit playback, and clamps background stalls', () => {
    const { camera, timeline, opening } = setup(true)
    opening.setAutoRotate(true)
    opening.setReducedMotion(true)
    opening.setAutoRotate(true)
    const initial = pose(camera)
    opening.togglePlayback()
    opening.advance(60)
    expect(timeline.t).toBeCloseTo(0.1125)
    expect(pose(camera)).toEqual(initial)
    opening.setReducedMotion(false)
    opening.advance(0.1)
    expect(pose(camera)).toEqual(initial)
  })
})

describe('essential data gates every playback entry', () => {
  it.each(['density', 'elevation'] as const)('waits for %s with an actionable loading explanation', (missing) => {
    const { timeline, opening } = setup()
    opening.setEssential(missing === 'density' ? 'elevation' : 'density', 'ready')
    expect(opening.readiness.canBegin).toBe(false)
    expect(opening.readiness.message).toMatch(/loading/i)
    expect(opening.readiness.message).toMatch(missing === 'density' ? /soil/i : /elevation/i)
    for (const attempt of [
      () => opening.togglePlayback(), () => opening.replay(),
      () => opening.handleSpace({ key: ' ', repeat: false }),
    ]) {
      attempt()
      opening.advance(0.1)
      expect(timeline.playing).toBe(false)
      expect(timeline.t).toBe(0)
    }
  })

  it('starts with the first bound density and elevation, without urban or higher tiers', () => {
    const { timeline, opening } = setup(true)
    expect(opening.readiness.canBegin).toBe(true)
    opening.handleSpace({ key: ' ', repeat: false })
    expect(timeline.playing).toBe(true)
    // A failed progressive tier cannot invalidate a usable bound elevation.
    opening.setEssential('elevation', 'error')
    expect(opening.readiness.canBegin).toBe(true)
  })

  it('does not snap an explored pose when data arrives or Begin is pressed', () => {
    const { camera, timeline, opening } = setup()
    opening.explore()
    camera.orbitBy(0.2, -0.3)
    camera.scrollBy(10)
    const userPose = pose(camera)
    opening.setEssential('density', 'ready')
    opening.setEssential('elevation', 'ready')
    expect(timeline.playing).toBe(false)
    opening.togglePlayback()
    opening.advance(0.1)
    expect(timeline.t).toBeGreaterThan(0)
    expect(pose(camera)).toEqual(userPose)
  })

  it.each(['density', 'elevation'] as const)('explains failed %s and leaves exploration/scrubbing usable', (failed) => {
    const { camera, timeline, opening } = setup()
    opening.setEssential(failed, 'error')
    expect(opening.readiness.message).toMatch(/failed/i)
    expect(opening.readiness.message).toMatch(/explore/i)
    expect(opening.readiness.message).toMatch(/placeholder/i)
    expect(opening.readiness.canBegin).toBe(false)
    opening.explore()
    camera.scrollBy(1)
    opening.scrubTo(9)
    opening.replay()
    expect(timeline.playing).toBe(false)
    expect(timeline.t).toBe(9)
    opening.handleSpace({ key: ' ', repeat: false })
    expect(timeline.t).toBe(9)
    expect(timeline.playing).toBe(false)
    expect(camera.altitude).toBeLessThan(11_500_000)
    const userPose = pose(camera)
    opening.setEssential('density', 'ready')
    opening.setEssential('elevation', 'ready')
    expect(opening.readiness.canBegin).toBe(true)
    expect(timeline.playing).toBe(false) // Late success never starts a sweep.
    expect(pose(camera)).toEqual(userPose)
    expect(timeline.t).toBe(9)
  })

  it('leaves native Space activation to focused controls and ignores repeats/modifiers', () => {
    const { timeline, opening } = setup(true)
    for (const event of [
      { key: ' ', repeat: true }, { key: ' ', repeat: false, ctrlKey: true },
      { key: ' ', repeat: false, metaKey: true }, { key: ' ', repeat: false, altKey: true },
      { key: 'Enter', repeat: false },
    ]) expect(opening.handleSpace(event)).toBe(false)
    expect(opening.handleSpace({ key: ' ', repeat: false }, true)).toBe(false)
    expect(timeline.playing).toBe(false)
    expect(opening.handleSpace({ key: ' ', repeat: false })).toBe(true)
    expect(timeline.playing).toBe(true)
    opening.handleSpace({ key: ' ', repeat: false })
    expect(timeline.playing).toBe(false)
  })
})

describe('deliberate essential-data recovery', () => {
  it('keeps recovery unavailable until an essential load has failed', async () => {
    const { opening } = setup()
    expect(opening.readiness).toMatchObject({ canRetry: false, retrying: false })
    await opening.loadEssential('density', ['density'], async () => { throw new Error('offline') })
    expect(opening.readiness).toMatchObject({ canBegin: false, canRetry: true, retrying: false })
  })

  it('retries only failed assets, coalesces repeated clicks, and preserves explored pose and sweep time', async () => {
    const { camera, timeline, opening } = setup()
    const requests: string[] = []
    let release!: () => void
    const resumed = new Promise<void>((resolve) => { release = resolve })
    let offline = true
    await opening.loadEssential('density', ['density'], async (url) => {
      requests.push(url)
      if (offline) throw new Error('offline')
      await resumed
    })
    await opening.loadEssential('elevation', ['coarse', 'fine'], async (url) => { requests.push(url) })
    camera.setOrientation(2, -0.5)
    camera.setAltitude(800)
    opening.scrubTo(9)
    const userPose = pose(camera)
    offline = false
    const retry = opening.retryDataLoading()
    const duplicate = opening.retryDataLoading()
    expect(opening.readiness).toMatchObject({ canBegin: false, canRetry: false, retrying: true })
    release()
    await Promise.all([retry, duplicate])
    expect(requests).toEqual(['density', 'coarse', 'fine', 'density'])
    expect(opening.readiness).toMatchObject({ canBegin: true, canRetry: false, retrying: false })
    expect(pose(camera)).toEqual(userPose)
    expect(timeline.t).toBe(9)
    expect(timeline.playing).toBe(false)
    opening.togglePlayback()
    expect(timeline.playing).toBe(true)
  })

  it('bounds persistent failure to the original attempt and two deliberate retries', async () => {
    const { opening } = setup()
    let requests = 0
    await opening.loadEssential('density', ['density'], async () => {
      requests++
      throw new Error('still offline')
    })
    await opening.retryDataLoading()
    await opening.retryDataLoading()
    await opening.retryDataLoading()
    expect(requests).toBe(3)
    expect(opening.readiness).toMatchObject({
      canBegin: false, canRetry: false, retrying: false, retryExhausted: true,
    })
  })

  it('keeps the progressive ladder sequential and never retries or downgrades a healthy tier', async () => {
    const { opening } = setup()
    const requests: string[] = []
    let release!: () => void
    let reachedFine!: () => void
    const fineStarted = new Promise<void>((resolve) => { reachedFine = resolve })
    const fineTier = new Promise<void>((resolve) => { release = resolve })
    await opening.loadEssential('density', ['density'], async (url) => { requests.push(url) })
    const elevation = opening.loadEssential('elevation', ['coarse', 'fine', 'finest'], async (url) => {
      requests.push(url)
      if (url === 'fine') { reachedFine(); await fineTier }
      if (url === 'finest') throw new Error('upgrade unavailable')
    })
    await fineStarted
    expect(requests).toEqual(['density', 'coarse', 'fine'])
    expect(opening.readiness.canBegin).toBe(true)
    await opening.retryDataLoading()
    release()
    await elevation
    await opening.retryDataLoading()
    expect(requests).toEqual(['density', 'coarse', 'fine', 'finest'])
    expect(opening.readiness).toMatchObject({ canBegin: true, canRetry: false })
  })

  it('waits for remaining initial tiers before offering recovery and accepts a later-tier success', async () => {
    const { opening } = setup()
    const requests: string[] = []
    let release!: () => void
    let reachedFine!: () => void
    const fineStarted = new Promise<void>((resolve) => { reachedFine = resolve })
    const fineTier = new Promise<void>((resolve) => { release = resolve })
    const loading = opening.loadEssential('elevation', ['coarse', 'fine'], async (url) => {
      requests.push(url)
      if (url === 'coarse') throw new Error('coarse unavailable')
      reachedFine()
      await fineTier
    })
    await fineStarted
    expect(opening.readiness.canRetry).toBe(false)
    await opening.retryDataLoading()
    release()
    await loading
    await opening.loadEssential('density', ['density'], async () => {})
    expect(opening.readiness.canBegin).toBe(true)
    expect(requests).toEqual(['coarse', 'fine'])
  })
})
