import { describe, it, expect } from 'vitest'
import { WGS84 } from '../src/math/wgs84'
import {
  capAreaVisibleM2, frustumFootprintAreaM2, visibleGroundAreaM2, densityMixFactor,
  estimatedCountInView, formatCount, humanComparison,
  COUNT_COMPARISONS, comparisonTextFor,
  GLOBAL_MEAN_DENSITY_PER_SQM, PUBLISHED_TOTAL_INDIVIDUALS, TOTAL_SURFACE_AREA_M2,
} from '../src/field/countEstimate'

const FOV_V = (50 * Math.PI) / 180 // matches viewer.ts's camera
const ASPECT = 16 / 9

describe('GLOBAL_MEAN_DENSITY_PER_SQM', () => {
  it('recovers close to the published 4.4e20 total when multiplied by the full sphere area', () => {
    const fullSphereArea = 4 * Math.PI * WGS84.R_MEAN ** 2
    const recovered = GLOBAL_MEAN_DENSITY_PER_SQM * fullSphereArea
    expect(Math.abs(recovered - PUBLISHED_TOTAL_INDIVIDUALS) / PUBLISHED_TOTAL_INDIVIDUALS).toBeLessThan(0.01)
  })

  it('is derived from the exact pipeline total-area figure, not a fresh sphere approximation', () => {
    expect(GLOBAL_MEAN_DENSITY_PER_SQM).toBeCloseTo(PUBLISHED_TOTAL_INDIVIDUALS / TOTAL_SURFACE_AREA_M2, 10)
  })
})

describe('capAreaVisibleM2', () => {
  it('is exactly 0 at altitude 0', () => {
    expect(capAreaVisibleM2(0)).toBe(0)
  })

  it('is monotonically increasing in altitude', () => {
    const alts = [0, 1000, 1e5, 1e7, 1e9]
    for (let i = 1; i < alts.length; i++) {
      expect(capAreaVisibleM2(alts[i]!)).toBeGreaterThan(capAreaVisibleM2(alts[i - 1]!))
    }
  })

  it('approaches exactly a hemisphere as altitude -> infinity', () => {
    const hemisphere = 2 * Math.PI * WGS84.R_MEAN ** 2
    const relErr = Math.abs(capAreaVisibleM2(1e13) - hemisphere) / hemisphere
    expect(relErr).toBeLessThan(1e-4)
  })

  it('never exceeds a hemisphere at any finite altitude', () => {
    const hemisphere = 2 * Math.PI * WGS84.R_MEAN ** 2
    for (const alt of [1000, 1e6, 1e9, 1e12]) {
      expect(capAreaVisibleM2(alt)).toBeLessThan(hemisphere)
    }
  })

  it('matches a hand-computed value at 11,500 km (this piece\'s default orbital altitude)', () => {
    // d = R + alt, cap = 2*pi*R^2*(1 - R/d)
    const alt = 1.15e7
    const d = WGS84.R_MEAN + alt
    const expected = 2 * Math.PI * WGS84.R_MEAN ** 2 * (1 - WGS84.R_MEAN / d)
    expect(capAreaVisibleM2(alt)).toBeCloseTo(expected, 0)
  })
})

describe('frustumFootprintAreaM2', () => {
  it('is exactly 0 at altitude 0', () => {
    expect(frustumFootprintAreaM2(0, FOV_V, ASPECT)).toBe(0)
  })

  it('scales quadratically with altitude (doubling altitude quadruples area)', () => {
    const a1 = frustumFootprintAreaM2(1000, FOV_V, ASPECT)
    const a2 = frustumFootprintAreaM2(2000, FOV_V, ASPECT)
    expect(a2 / a1).toBeCloseTo(4, 6)
  })

  it('scales linearly with aspect ratio', () => {
    const narrow = frustumFootprintAreaM2(1000, FOV_V, 1)
    const wide = frustumFootprintAreaM2(1000, FOV_V, 2)
    expect(wide / narrow).toBeCloseTo(2, 6)
  })

  it('matches a hand-computed value at 1 km', () => {
    const halfHeight = 1000 * Math.tan(FOV_V / 2)
    const height = 2 * halfHeight
    const width = height * ASPECT
    expect(frustumFootprintAreaM2(1000, FOV_V, ASPECT)).toBeCloseTo(width * height, 0)
  })
})

describe('visibleGroundAreaM2', () => {
  it('is frustum-bound (much smaller than the cap) near the surface', () => {
    const alt = 1000
    const area = visibleGroundAreaM2(alt, FOV_V, ASPECT)
    expect(area).toBeCloseTo(frustumFootprintAreaM2(alt, FOV_V, ASPECT), 0)
    expect(area).toBeLessThan(capAreaVisibleM2(alt) / 1000)
  })

  it('is cap-bound (a real fraction of a hemisphere) at this piece\'s orbital altitude', () => {
    const alt = 1.15e7
    const area = visibleGroundAreaM2(alt, FOV_V, ASPECT)
    expect(area).toBeCloseTo(capAreaVisibleM2(alt), 0)
    const hemisphere = 2 * Math.PI * WGS84.R_MEAN ** 2
    expect(area / hemisphere).toBeGreaterThan(0.1)
    expect(area / hemisphere).toBeLessThan(1)
  })

  it('is always the smaller of the two bounds, at any altitude', () => {
    for (const alt of [1, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8]) {
      const area = visibleGroundAreaM2(alt, FOV_V, ASPECT)
      expect(area).toBeCloseTo(Math.min(capAreaVisibleM2(alt), frustumFootprintAreaM2(alt, FOV_V, ASPECT)), 0)
    }
  })
})

describe('densityMixFactor', () => {
  it('is 0 at and below the low threshold', () => {
    expect(densityMixFactor(0)).toBe(0)
    expect(densityMixFactor(5000)).toBe(0)
    expect(densityMixFactor(10_000)).toBe(0)
  })

  it('is 1 at and above the high threshold', () => {
    expect(densityMixFactor(2_000_000)).toBe(1)
    expect(densityMixFactor(1e8)).toBe(1)
  })

  it('is 0.5 exactly at the midpoint (smoothstep is symmetric)', () => {
    expect(densityMixFactor(1_005_000, 10_000, 2_000_000)).toBeCloseTo(0.5, 10)
  })

  it('is monotonically non-decreasing across the transition band', () => {
    const samples = [10_000, 100_000, 500_000, 1_000_000, 1_500_000, 2_000_000].map((a) => densityMixFactor(a))
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]!).toBeGreaterThanOrEqual(samples[i - 1]!)
    }
  })
})

describe('estimatedCountInView', () => {
  it('reduces to area * density when local and mean densities are equal (mix is irrelevant)', () => {
    const density = 2e6
    for (const alt of [10, 1000, 1e5, 1e7]) {
      const area = visibleGroundAreaM2(alt, FOV_V, ASPECT)
      expect(estimatedCountInView(alt, density, density, FOV_V, ASPECT)).toBeCloseTo(area * density, 0)
    }
  })

  it('is dominated by the LOCAL density near the surface', () => {
    const local = 1e5
    const mean = 1e10 // wildly different, to make the dominant term obvious
    const count = estimatedCountInView(10, local, mean, FOV_V, ASPECT)
    const area = visibleGroundAreaM2(10, FOV_V, ASPECT)
    expect(count).toBeCloseTo(area * local, -2)
  })

  it('is dominated by the MEAN density from orbit', () => {
    const local = 1e5
    const mean = 1e10
    const count = estimatedCountInView(1e7, local, mean, FOV_V, ASPECT)
    const area = visibleGroundAreaM2(1e7, FOV_V, ASPECT)
    expect(count).toBeCloseTo(area * mean, -2)
  })

  it('is strictly positive for any positive density at a positive altitude', () => {
    for (const alt of [1, 1000, 1e6, 1e9]) {
      expect(estimatedCountInView(alt, 1e5, 1e6, FOV_V, ASPECT)).toBeGreaterThan(0)
    }
  })

  it('is exactly 0 at altitude 0 — a genuine edge case (zero visible ground area), not a bug', () => {
    expect(estimatedCountInView(0, 1e5, 1e6, FOV_V, ASPECT)).toBe(0)
  })

  describe('known altitudes (magnitude sanity, not exact figures — see field/countEstimate.ts\'s doc)', () => {
    // Using the global mean as a stand-in for "local" here too (no live
    // camera direction to sample outside the running app) - a real local
    // sample would move these within their biome's ~30x range but not
    // change the ORDER of magnitude, which is what these bounds check.
    const density = GLOBAL_MEAN_DENSITY_PER_SQM

    it('~1e20 at 11,500 km (this piece\'s default orbital altitude)', () => {
      const count = estimatedCountInView(1.15e7, density, density, FOV_V, ASPECT)
      expect(count).toBeGreaterThan(1e19)
      expect(count).toBeLessThan(1e21)
    })

    it('~1e16 at 100 km', () => {
      const count = estimatedCountInView(1e5, density, density, FOV_V, ASPECT)
      expect(count).toBeGreaterThan(1e15)
      expect(count).toBeLessThan(1e17)
    })

    it('~1e12 at 1 km', () => {
      const count = estimatedCountInView(1000, density, density, FOV_V, ASPECT)
      expect(count).toBeGreaterThan(1e11)
      expect(count).toBeLessThan(1e13)
    })

    it('~1e2 at 1 cm — "a handful", order of magnitude, at global-mean density', () => {
      const count = estimatedCountInView(0.01, density, density, FOV_V, ASPECT)
      expect(count).toBeGreaterThan(10)
      expect(count).toBeLessThan(1e4)
    })

    it('is monotonically decreasing as altitude drops across all four checkpoints', () => {
      const counts = [1.15e7, 1e5, 1000, 0.01].map((alt) => estimatedCountInView(alt, density, density, FOV_V, ASPECT))
      for (let i = 1; i < counts.length; i++) {
        expect(counts[i]!).toBeLessThan(counts[i - 1]!)
      }
    })
  })
})

describe('formatCount', () => {
  it('uses exponential notation with a superscript exponent above 1000', () => {
    expect(formatCount(1.2e19)).toBe('1.2 × 10¹⁹')
    expect(formatCount(4.4e20)).toBe('4.4 × 10²⁰')
  })

  it('uses a plain integer below 1000', () => {
    expect(formatCount(42)).toBe('42')
    expect(formatCount(999)).toBe('999')
  })

  it('uses one decimal place below 10', () => {
    expect(formatCount(3)).toBe('3.0')
    expect(formatCount(0.5)).toBe('0.5')
  })

  it('is exact at the 1000 boundary', () => {
    expect(formatCount(1000)).toBe('1.0 × 10³')
  })

  it('handles zero and negative input without producing NaN/Infinity text', () => {
    expect(formatCount(0)).toBe('0')
    expect(formatCount(-5)).toBe('0')
  })
})

/**
 * These tests exist because the version they replace could not fail for the
 * reason that mattered. It asserted `humanComparison(t).length > 0` at a list
 * of thresholds — which is true of ANY non-empty string, so it passed happily
 * while the table claimed 1e18 was "more than every grain of sand on every
 * beach on Earth" (the usual cited figure for that is ~7.5e18, so the claim
 * was false by nearly an order of magnitude at its own threshold) and that
 * 1e9 was "more than every human alive" (it is not; there are ~8.2e9).
 *
 * The load-bearing test is `is TRUE at its own threshold` below: it compares
 * each band's floor against the published figure the band's copy names, so a
 * threshold set below its own anchor fails. Flip any `atLeast` under its
 * `anchor.value` and watch it go red.
 */
describe('COUNT_COMPARISONS (the table behind humanComparison)', () => {
  it('is ordered strictly from largest threshold to smallest', () => {
    for (let i = 1; i < COUNT_COMPARISONS.length; i++) {
      expect(COUNT_COMPARISONS[i]!.atLeast).toBeLessThan(COUNT_COMPARISONS[i - 1]!.atLeast)
    }
  })

  it('covers every non-negative count (the last band has a floor of 0)', () => {
    expect(COUNT_COMPARISONS.at(-1)!.atLeast).toBe(0)
  })

  it('is TRUE at its own threshold: every "more than X" band sits at or above the published X', () => {
    // THE test. A band claiming "more than every tree on Earth" whose floor
    // is BELOW the published tree count is a false statement on screen at
    // every count between the two.
    for (const band of COUNT_COMPARISONS) {
      if (!band.anchor) continue
      expect(
        band.atLeast,
        `band "${comparisonTextFor(band, band.atLeast)}" claims more than ${band.anchor.label} `
        + `(${band.anchor.value.toExponential(2)}) but its floor is only ${band.atLeast.toExponential(2)}`,
      ).toBeGreaterThanOrEqual(band.anchor.value)
    }
  })

  it('names the thing it counts: every anchored band\'s text mentions its anchor label', () => {
    for (const band of COUNT_COMPARISONS) {
      if (!band.anchor) continue
      expect(comparisonTextFor(band, band.atLeast).toLowerCase()).toContain(band.anchor.label.toLowerCase())
    }
  })

  it('cites a real, lookup-able source for every anchored band (author or body, and a year)', () => {
    for (const band of COUNT_COMPARISONS) {
      if (!band.anchor) continue
      expect(band.anchor.source.length, `${band.anchor.label} has no source`).toBeGreaterThan(20)
      expect(band.anchor.source, `${band.anchor.label}'s source has no year`).toMatch(/\b(19|20)\d{2}\b/)
    }
  })

  it('never makes an unsourced comparative claim: unanchored bands say no "more than"', () => {
    // The escape hatch that let the old copy ship. A band with no anchor is
    // allowed to describe the count ("countable, one by one") but not to
    // compare it against an unnamed quantity.
    for (const band of COUNT_COMPARISONS) {
      if (band.anchor) continue
      const text = comparisonTextFor(band, Math.max(band.atLeast, 1))
      expect(text, `unanchored band "${text}" makes a comparative claim`).not.toMatch(/more than|every /i)
    }
  })
})

describe('humanComparison', () => {
  it('selects the band whose floor the count sits on, exactly at each floor', () => {
    for (const band of COUNT_COMPARISONS) {
      expect(humanComparison(band.atLeast)).toBe(comparisonTextFor(band, band.atLeast))
    }
  })

  it('drops to the NEXT band down just below each floor (no band swallows its neighbour)', () => {
    for (let i = 0; i < COUNT_COMPARISONS.length - 1; i++) {
      const band = COUNT_COMPARISONS[i]!
      const below = band.atLeast * (1 - 1e-9)
      expect(humanComparison(below)).not.toBe(comparisonTextFor(band, band.atLeast))
      expect(humanComparison(below)).toBe(comparisonTextFor(COUNT_COMPARISONS[i + 1]!, below))
    }
  })

  it('quantifies the top band as a share of the published global total', () => {
    // At this piece's default orbital altitude ~1.4e20 are in view against a
    // published 4.4e20 total, so the honest thing to say is "a third of them".
    expect(humanComparison(1.4e20)).toBe('About 32% of the global surface-soil nematode estimate')
    expect(humanComparison(PUBLISHED_TOTAL_INDIVIDUALS / 2)).toBe('About 50% of the global surface-soil nematode estimate')
  })

  it('does not clamp a share above 100% — an impossible reading must stay visible as one', () => {
    expect(humanComparison(PUBLISHED_TOTAL_INDIVIDUALS * 1.4)).toBe('About 140% of the global surface-soil nematode estimate')
  })

  it('is the emptiest comparison below 1', () => {
    expect(humanComparison(0.5)).toBe('The estimated count is below one')
  })

  it('carries each published figure at the exact value its source reports', () => {
    // Spot-check against the abstracts these came from, so a later edit that
    // "rounds" a figure has to change a test that names the paper.
    const byLabel = new Map(COUNT_COMPARISONS.flatMap((b) => (b.anchor ? [[b.anchor.label, b.anchor] as const] : [])))
    expect(byLabel.get('ant')?.value).toBe(2.0e16)        // Schultheiss 2022: "almost 20 x 10^15"
    expect(byLabel.get('cells')?.value).toBe(3.72e13)     // Bianconi 2013: "3.72 x 10^13"
    expect(byLabel.get('tree')?.value).toBe(3.04e12)      // Crowther 2015: "approximately 3.04 trillion"
    expect(byLabel.get('human')?.value).toBe(1.03e10)     // UN WPP 2024: peak "approximately 10.3 billion"
  })
})
