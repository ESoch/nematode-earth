import { describe, it, expect } from 'vitest'
import {
  HUD_COPY, KEY_BINDINGS, ALWAYS_VISIBLE_HINTS, maxParenDepth, allCopyStrings,
} from '../src/hud/copy'

/** Every parenthesised span, flattened, so a nesting depth can be asserted per string. */
describe('maxParenDepth', () => {
  it('is 0 for a string with no parentheses', () => {
    expect(maxParenDepth('plain text')).toBe(0)
  })

  it('is 1 for a single flat pair', () => {
    expect(maxParenDepth('a (b) c (d)')).toBe(1)
  })

  it('is 2 for a nested pair — the shape being banned', () => {
    expect(maxParenDepth('our own measurement: a = -0.0085 (95% CI -0.035, 0.019)')).toBe(1)
    expect(maxParenDepth('(our own measurement: a = -0.0085 (95% CI -0.035, 0.019))')).toBe(2)
  })
})

describe('HUD copy', () => {
  it('never nests parentheses anywhere on screen', () => {
    // The defect: "cobb inversion data (our own measurement: 243,294 pixels
    // ... a = -0.0085 (95% CI -0.035, 0.019) ...)" — a parenthetical inside a
    // parenthetical, in a line a lay reader was expected to parse.
    for (const [name, s] of allCopyStrings()) {
      expect(maxParenDepth(s), `${name}: "${s}"`).toBeLessThanOrEqual(1)
    }
  })

  it('balances its parentheses', () => {
    for (const [name, s] of allCopyStrings()) {
      const open = (s.match(/\(/g) ?? []).length
      const close = (s.match(/\)/g) ?? []).length
      expect(open, `${name}: "${s}"`).toBe(close)
    }
  })
})

describe('the finding (what replaced the run-on stat line)', () => {
  const { findingPrediction, findingResult, findingStatistics } = HUD_COPY

  it('states the prediction and the result as separate, plain sentences', () => {
    for (const s of [findingPrediction, findingResult]) {
      expect(s.length).toBeGreaterThan(20)
      expect(maxParenDepth(s)).toBe(0)
    }
    // The result tier must actually deliver a verdict, not restate the setup.
    expect(findingResult.toLowerCase()).toMatch(/\bnot\b|\bno\b/)
  })

  it('keeps every measured figure across the finding — a copy edit may not quietly drop the unflattering ones', () => {
    // R^2 is negative and the CI straddles zero. Both are the point. A future
    // tidy-up that removes them to make the line read better fails here.
    const whole = [findingPrediction, findingResult, findingStatistics].join(' ')
    for (const fragment of ['20', '243,294', '0.0085', '0.035', '0.019', '0.0003']) {
      expect(whole, `the finding lost ${fragment}`).toContain(fragment)
    }
  })

  it('says in words that the interval spans zero, rather than leaving it to be inferred', () => {
    expect(findingStatistics.toLowerCase()).toMatch(/span|straddl|cross|includ/)
  })

  it('marks the slope and interval as negative with a real minus sign, not a hyphen', () => {
    expect(findingStatistics).toContain('−0.0085')
    expect(findingStatistics).not.toMatch(/-0\.0085/)
  })
})

describe('Cobb slider labels (the control that read as broken)', () => {
  const { cobbAxisMeasured, cobbAxisPredicted } = HUD_COPY

  it('names both ends as pictures of the world, so neither end reads as off', () => {
    // The defect: the data end was labelled "data (measured: no effect)",
    // which on a slider end reads as "this end is switched off" rather than
    // as a finding. Any label carrying an on/off word brings that back.
    for (const label of [cobbAxisMeasured, cobbAxisPredicted]) {
      expect(label.toLowerCase(), `"${label}" reads as a dead control`)
        .not.toMatch(/no effect|nothing|none|off\b|disabled|inactive|null/)
    }
  })

  it('keeps the two ends distinct, and attributes the prediction to Cobb but not the measurement', () => {
    expect(cobbAxisMeasured).not.toBe(cobbAxisPredicted)
    expect(cobbAxisPredicted.toLowerCase()).toContain('cobb')
    expect(cobbAxisMeasured.toLowerCase()).not.toContain('cobb')
  })

  it('explains, at the prediction end, why full strength still barely shows', () => {
    // Otherwise a viewer drags to the far end, sees almost nothing, and
    // concludes the control is broken instead of learning that built-up
    // ground is a vanishing fraction of the planet.
    expect(HUD_COPY.cobbCaptionPrediction.toLowerCase()).toMatch(/built up|built-up/)
    expect(HUD_COPY.cobbCaptionPrediction).toContain('modelled density')
    expect(HUD_COPY.cobbCaptionPrediction).toContain('4×')
    expect(HUD_COPY.cobbCaptionPrediction).toContain('We chose')
  })

  it('identifies the two maps and the limits of a city comparison', () => {
    expect(HUD_COPY.cobbCaptionMeasured).toContain('two maps')
    expect(HUD_COPY.cobbCaptionMeasured).toContain('Field samples')
  })

  it('has a distinct explanation for each reason the control can be inert', () => {
    const inert = [HUD_COPY.cobbUnavailableSynthetic, HUD_COPY.cobbUnavailableUrbanLoading]
    expect(new Set(inert).size).toBe(inert.length)
    for (const s of inert) expect(s.length).toBeGreaterThan(15)
    // The synthetic case must point at the way out of it.
    expect(HUD_COPY.cobbUnavailableSynthetic).toContain('d')
  })
})

describe('key bindings (the largest ink block in the old HUD)', () => {
  it('keeps the always-visible hint list to three items or fewer', () => {
    // The defect: eight bindings on one run-on line, measured as the single
    // largest ink block in the HUD and the least important content in it.
    expect(ALWAYS_VISIBLE_HINTS.length).toBeLessThanOrEqual(3)
  })

  it('signposts the full list from the always-visible hints', () => {
    expect(ALWAYS_VISIBLE_HINTS.join(' ')).toContain('?')
  })

  it('gives every binding a single unique key and a description', () => {
    const keys = KEY_BINDINGS.map((b) => b.key)
    expect(new Set(keys).size, `duplicate key in ${keys.join(',')}`).toBe(keys.length)
    for (const b of KEY_BINDINGS) {
      expect(b.description.length, `${b.key} has no description`).toBeGreaterThan(3)
      expect(b.group.length).toBeGreaterThan(0)
    }
  })

  it('documents exactly the keys the viewer handles — no ghosts, no omissions', () => {
    // Kept in step with viewer.ts's keydown handler by hand; this list is the
    // thing to update when a binding is added or removed, and a mismatch here
    // is a HUD that lies about what the piece can do.
    const handled = [' ', 'd', 'u', 's', 'r', '[', ']', 'i', 'v', 'p', 't', '?']
    expect(KEY_BINDINGS.map((b) => b.key).sort()).toEqual([...handled].sort())
  })

  it('separates bindings that change what the piece CLAIMS from bindings that change how it LOOKS', () => {
    // `d` (real vs synthetic) and `u` (model uncertainty) are honesty
    // controls; burying them in a list of view toggles is what made them
    // undiscoverable. They get their own group, and are surfaced next to the
    // readouts they qualify rather than only in the key panel.
    const evidence = KEY_BINDINGS.filter((b) => b.group === 'evidence').map((b) => b.key)
    expect(evidence.sort()).toEqual(['d', 'u'])
  })
})
