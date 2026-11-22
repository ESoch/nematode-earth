import { describe, it, expect } from 'vitest'
import {
  ELEVATION_TIERS,
  URBAN_TIERS,
  SMALL_CLIENT_SHORT_SIDE_PX,
  CAPPED_MAX_FACE_SIZE,
  planAtlasLadder,
  type ClientBudgetHints,
} from '../src/platform/atlasLadder'

/**
 * The full atlas set is ~54 MB over the wire, and the elevation 2048 tier is
 * 31.7 MiB brotli of it — measured MINIFIED at the opening framing (the disc
 * spans 1,323 device px for 12,742 km, so a 1024 texel already lands at
 * 1.015 device px and a 2048 texel at 0.507). Its benefit only appears below
 * ~8,400 km of descent. A phone therefore pays the single largest download in
 * the piece for detail it cannot resolve until it is most of the way down.
 *
 * These assertions pin the RULING, not the plumbing: which ladders a given
 * client gets, that the decision is a pure function of its inputs (so a
 * rotation cannot change it mid-session — viewer.ts calls this once), and
 * that the urban ladder is deliberately NOT capped on size alone.
 */

const desktop: ClientBudgetHints = {
  viewportWidthPx: 1512,
  viewportHeightPx: 982,
}
const phonePortrait: ClientBudgetHints = {
  viewportWidthPx: 390,
  viewportHeightPx: 844,
}
const phoneLandscape: ClientBudgetHints = {
  viewportWidthPx: 956,
  viewportHeightPx: 440,
}
const tablet: ClientBudgetHints = {
  viewportWidthPx: 820,
  viewportHeightPx: 1180,
}

const faceSizes = (tiers: ReadonlyArray<{ faceSize: number }>): number[] =>
  tiers.map((t) => t.faceSize)

describe('the full ladders', () => {
  it('runs elevation 256 -> 1024 -> 2048, coarsest first', () => {
    expect(faceSizes(ELEVATION_TIERS)).toEqual([256, 1024, 2048])
  })

  it('runs urban 512 -> 1024 -> 2048, coarsest first', () => {
    expect(faceSizes(URBAN_TIERS)).toEqual([512, 1024, 2048])
  })

  it('names a real atlas directory for every tier', () => {
    for (const t of [...ELEVATION_TIERS, ...URBAN_TIERS]) {
      expect(t.url).toMatch(/^\/data\/(elevation|urban)_plain_\d+$/)
      expect(t.url.endsWith(`_${t.faceSize}`)).toBe(true)
    }
  })
})

describe('a client with room for the whole ladder', () => {
  it('gets every elevation tier, 2048 included', () => {
    expect(faceSizes(planAtlasLadder(desktop).elevation)).toEqual([256, 1024, 2048])
  })

  it('gets every urban tier', () => {
    expect(faceSizes(planAtlasLadder(desktop).urban)).toEqual([512, 1024, 2048])
  })

  it('reports no cap', () => {
    expect(planAtlasLadder(desktop).cap).toBe('none')
  })
})

describe('small clients: the elevation ladder stops at 1024', () => {
  it('caps a phone held in portrait', () => {
    const plan = planAtlasLadder(phonePortrait)
    expect(faceSizes(plan.elevation)).toEqual([256, 1024])
    expect(plan.cap).toBe('small-viewport')
  })

  it('caps the SAME phone held in landscape — the short side is what decides', () => {
    // The failure this pins: a 956 CSS px landscape phone is WIDER than plenty
    // of laptops, so a width-only rule would hand the largest phones on sale
    // the 31.7 MiB tier. The camera has a fixed vertical FOV, so the disc's
    // span in device px is set by the SHORT side either way.
    const plan = planAtlasLadder(phoneLandscape)
    expect(faceSizes(plan.elevation)).toEqual([256, 1024])
    expect(plan.cap).toBe('small-viewport')
  })

  it('gives a phone the identical plan in both orientations', () => {
    expect(planAtlasLadder(phonePortrait)).toEqual(planAtlasLadder(phoneLandscape))
  })

  it('caps a tablet', () => {
    expect(faceSizes(planAtlasLadder(tablet).elevation)).toEqual([256, 1024])
  })

  it('never drops a tier the piece needs to render at all', () => {
    // 256 is first paint. Capping must remove the TOP of the ladder, never
    // the bottom, or a capped client waits on 1024 with nothing on screen.
    for (const hints of [phonePortrait, phoneLandscape, tablet, desktop]) {
      expect(planAtlasLadder(hints).elevation[0]?.faceSize).toBe(256)
      expect(planAtlasLadder(hints).urban[0]?.faceSize).toBe(512)
    }
  })

  it('caps exactly at the documented short-side threshold, not near it', () => {
    const atThreshold = {
      viewportWidthPx: 1400,
      viewportHeightPx: SMALL_CLIENT_SHORT_SIDE_PX,
    }
    const justUnder = {
      viewportWidthPx: 1400,
      viewportHeightPx: SMALL_CLIENT_SHORT_SIDE_PX - 1,
    }
    expect(planAtlasLadder(atThreshold).cap).toBe('none')
    expect(planAtlasLadder(justUnder).cap).toBe('small-viewport')
  })

  it('treats a viewport it cannot measure as small', () => {
    // Conservative on purpose: an unreadable viewport must not buy 31.7 MiB.
    expect(planAtlasLadder({ viewportWidthPx: 0, viewportHeightPx: 0 }).cap).toBe('small-viewport')
    expect(planAtlasLadder({ viewportWidthPx: NaN, viewportHeightPx: NaN }).cap).toBe(
      'small-viewport',
    )
  })
})

describe('urban 2048 survives a small viewport, and only a small viewport', () => {
  /**
   * The ruling, stated so it can be argued with: urban 2048 is 6.65 MiB
   * brotli — 21% of what elevation 2048 costs, because built-up fraction is
   * mostly exact zeros and compresses ~29x. It is also the only atlas
   * carrying the piece's SUBJECT: the finding is about cities, and a viewer
   * who cannot see cities as cities has lost the thing being tested. Size
   * alone therefore does not cap it; an explicit request for less data does.
   */
  it('keeps every urban tier on a phone', () => {
    expect(faceSizes(planAtlasLadder(phonePortrait).urban)).toEqual([512, 1024, 2048])
  })

  it('drops urban 2048 when the visitor has asked for less data', () => {
    const plan = planAtlasLadder({ ...phonePortrait, saveData: true })
    expect(faceSizes(plan.urban)).toEqual([512, 1024])
    expect(faceSizes(plan.elevation)).toEqual([256, 1024])
    expect(plan.cap).toBe('save-data')
  })

  it('drops urban 2048 on a connection the browser calls slow', () => {
    for (const effectiveType of ['slow-2g', '2g', '3g']) {
      const plan = planAtlasLadder({ ...desktop, effectiveType })
      expect(faceSizes(plan.urban), effectiveType).toEqual([512, 1024])
      expect(faceSizes(plan.elevation), effectiveType).toEqual([256, 1024])
      expect(plan.cap, effectiveType).toBe('slow-network')
    }
  })
})

describe('navigator.connection is Chromium-only, so it is used and never depended on', () => {
  it('caps nothing on a roomy viewport when the hint is absent entirely', () => {
    expect(planAtlasLadder(desktop).cap).toBe('none')
    expect(
      planAtlasLadder({
        ...desktop,
        effectiveType: undefined,
        saveData: undefined,
      }).cap,
    ).toBe('none')
    expect(planAtlasLadder({ ...desktop, effectiveType: null }).cap).toBe('none')
  })

  it('caps nothing on a hint value it does not recognise', () => {
    expect(planAtlasLadder({ ...desktop, effectiveType: '4g' }).cap).toBe('none')
    expect(planAtlasLadder({ ...desktop, effectiveType: '5g' }).cap).toBe('none')
    expect(planAtlasLadder({ ...desktop, effectiveType: 'unknown' }).cap).toBe('none')
  })

  it('still caps a small viewport whatever the connection claims', () => {
    expect(planAtlasLadder({ ...phonePortrait, effectiveType: '4g' }).cap).toBe('small-viewport')
  })

  it('ranks save-data above every other reason, so the strongest signal names the cap', () => {
    expect(planAtlasLadder({ ...phonePortrait, saveData: true, effectiveType: '2g' }).cap).toBe(
      'save-data',
    )
    expect(planAtlasLadder({ ...phonePortrait, effectiveType: '2g' }).cap).toBe('slow-network')
  })
})

describe('the decision is a pure function of its inputs', () => {
  it('returns the same plan for the same hints, every time', () => {
    for (const hints of [desktop, phonePortrait, tablet, { ...desktop, saveData: true }]) {
      expect(planAtlasLadder(hints)).toEqual(planAtlasLadder(hints))
    }
  })

  it('reads nothing from the ambient environment', () => {
    // No globals: the whole point is that viewer.ts samples the environment
    // ONCE and this stays answerable without one. If this ever started
    // reading `window`, a rotation could change the answer mid-session.
    expect(planAtlasLadder.length).toBe(1)
    expect(String(planAtlasLadder)).not.toMatch(/\bwindow\b|\bnavigator\b/)
  })

  it('never hands back a tier above the cap size once capped', () => {
    const plan = planAtlasLadder({ ...phonePortrait, saveData: true })
    for (const t of [...plan.elevation, ...plan.urban]) {
      expect(t.faceSize).toBeLessThanOrEqual(CAPPED_MAX_FACE_SIZE)
    }
  })
})
