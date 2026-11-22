import { describe, it, expect } from 'vitest'
import VIEWER_HTML from '../viewer.html?raw'
import { ALWAYS_VISIBLE_HINTS } from '../src/hud/copy'
import VIEWER_TS from '../src/viewer.ts?raw'

/**
 * The touch surface, checked against the shipped stylesheet.
 *
 * Two of these pin defects that render perfectly and cannot be used:
 *
 *  1. `touch-action`. The canvas tracks pointers itself, but a browser that
 *     has not been told otherwise claims two-finger gestures for its own page
 *     zoom and pans — it fires `pointercancel` and the pinch simply stops
 *     part-way, intermittently, on some devices and not others. The handler
 *     runs; the gesture dies. No screenshot shows it.
 *  2. Hit sizes. `.rail-row button` was 28 px tall and an `input[type=range]`
 *     defaults to about 16 px of hit box. Both draw correctly at any size.
 *
 * WCAG 2.2 SC 2.5.8 Target Size Minimum is 24x24 CSS px; 44 px is the figure
 * both Apple's and Google's touch guidance use, and it is what the coarse
 * pointer block below aims at. The desktop composition is untouched: every
 * rule here is inside `@media (pointer: coarse)`, so a mouse still gets the
 * compact rail this HUD was designed around.
 */

/** Body of the first `@media (pointer: coarse)` block, braces balanced. */
const coarseBlock = (() => {
  const at = VIEWER_HTML.indexOf('@media (pointer: coarse)')
  expect(at, 'no @media (pointer: coarse) block in viewer.html').toBeGreaterThan(-1)
  const open = VIEWER_HTML.indexOf('{', at)
  let depth = 0
  for (let i = open; i < VIEWER_HTML.length; i++) {
    if (VIEWER_HTML[i] === '{') depth++
    else if (VIEWER_HTML[i] === '}') {
      depth--
      if (depth === 0) return VIEWER_HTML.slice(open + 1, i)
    }
  }
  throw new Error('unbalanced @media (pointer: coarse) block')
})()

/** A `min-height`/`height` declaration in CSS px, given a selector's rule body. */
function ruleFor(css: string, selector: string): string {
  const at = css.indexOf(selector)
  expect(at, `no rule for ${selector}`).toBeGreaterThan(-1)
  const open = css.indexOf('{', at)
  const close = css.indexOf('}', open)
  return css.slice(open + 1, close)
}

function pxOf(body: string, prop: string): number {
  const m = new RegExp(`${prop}:\\s*([\\d.]+)(px|rem)`).exec(body)
  expect(m, `${prop} not declared, or not in px/rem`).not.toBeNull()
  const n = Number(m![1])
  return m![2] === 'rem' ? n * 16 : n
}

const MIN_TOUCH_PX = 44

describe('the browser must not steal the gesture', () => {
  it('declares touch-action: none on the canvas', () => {
    // Without this the page's own pinch-zoom wins and the canvas gets a
    // pointercancel mid-descent. The pinch is implemented and does nothing.
    expect(VIEWER_HTML).toMatch(/canvas\s*\{[^}]*touch-action:\s*none/)
  })

  it('leaves the rest of the page pinch-zoomable', () => {
    // The blunt fix is `user-scalable=no` on the viewport meta, which also
    // stops a visitor magnifying the HUD text -- a real accessibility cost
    // for a problem that is only the canvas's. The meta tag is read here
    // rather than the whole file, so the prose explaining that choice cannot
    // itself fail the check.
    const meta = /<meta\s+name="viewport"\s+content="([^"]*)"/.exec(VIEWER_HTML)?.[1] ?? ''
    expect(meta).toContain('width=device-width')
    expect(meta).not.toMatch(/user-scalable\s*=\s*no/)
    expect(meta).not.toMatch(/maximum-scale/)
    expect(VIEWER_HTML).not.toMatch(/(?:^|[^-])\bbody\s*\{[^}]*touch-action:\s*none/)
  })
})

describe('tap targets on a coarse pointer', () => {
  it('gives the Play and Replay buttons a full-size target', () => {
    expect(pxOf(ruleFor(coarseBlock, '.rail-row button'), 'min-height')).toBeGreaterThanOrEqual(
      MIN_TOUCH_PX,
    )
  })

  it('gives both sliders a full-size target', () => {
    // The Cobb axis and the sweep scrubber are the same element type and the
    // same rule; the piece's two primary controls are both `input[type=range]`.
    const body = ruleFor(coarseBlock, '.rail-row input[type="range"]')
    expect(pxOf(body, 'height')).toBeGreaterThanOrEqual(MIN_TOUCH_PX)
  })

  it('gives the range thumb a real size of its own, not just a tall track', () => {
    // A tall element with a 12 px thumb is a big target for STARTING a drag
    // and a small one for grabbing the handle. Both vendor pseudo-elements
    // are needed: -webkit- covers Safari and Chrome, and the standard
    // `::slider-thumb` does not exist in shipping browsers yet.
    expect(coarseBlock).toMatch(/-webkit-slider-thumb/)
    expect(coarseBlock).toMatch(/-moz-range-thumb/)
  })

  it('gives the methods link a target, since it is the only link on the page', () => {
    const body = ruleFor(coarseBlock, '.hud-about')
    expect(pxOf(body, 'min-height')).toBeGreaterThanOrEqual(24) // WCAG 2.2 AA
    // An inline link inherits `pointer-events: none` from #hud unless it opts
    // back in, which is how it once rendered and could not be clicked at all.
    expect(VIEWER_HTML).toMatch(/\.hud-about\s*\{[^}]*pointer-events:\s*auto/)
    // Padding only grows the hit box on an inline box if it is not inline.
    expect(body).toMatch(/display:\s*inline-block/)
  })

  it('changes nothing for a mouse', () => {
    // Every enlargement is inside the coarse block. If one leaked out, the
    // desktop rail would grow silently and the HUD's measured layout with it.
    const outside = VIEWER_HTML.slice(0, VIEWER_HTML.indexOf('@media (pointer: coarse)'))
    expect(outside).not.toMatch(/\.rail-row button\s*\{[^}]*min-height/)
    expect(outside).not.toMatch(/\.rail-row input\[type="range"\]\s*\{[^}]*height:/)
  })
})

describe('the one link out of the viewer survives a short viewport', () => {
  it('hangs off the finding, not off the hint line #hud clips first', () => {
    // #hud caps its own height and clips the overflow, and the ranking behind
    // that clip -- written out in this file's CSS -- puts the key hints at
    // the bottom of the list on purpose, as the utility tier. The about link
    // is not a hint: it is the only route out of the viewer, and ranking it
    // as utility deleted it on any short viewport, a phone held sideways
    // most obviously. It renders identically; it is only clipped later.
    const hints = /el\('div', 'hud-hints'[^)]*\)/.exec(VIEWER_TS)?.[0] ?? ''
    expect(hints, 'the about link is back in the clip-first tier').not.toContain('about')
    expect(VIEWER_TS).toMatch(/finding\.append\([\s\S]{0,400}hud-finding-link/)
  })
})

describe('the hints tell a phone what to do', () => {
  it('does not promise a gesture a phone does not have', () => {
    // "scroll to descend" is a mouse instruction, and descent is the whole
    // piece. A visitor on a phone reading it has been told to do the one
    // thing their device cannot do, which reads as the piece being broken.
    const hints = ALWAYS_VISIBLE_HINTS.join(' - ')
    expect(hints).toMatch(/pinch/i)
    expect(hints).toMatch(/descend/i)
  })

  it('still names the mouse gesture, since one string is shown to both', () => {
    expect(ALWAYS_VISIBLE_HINTS.join(' - ')).toMatch(/scroll/i)
  })
})
