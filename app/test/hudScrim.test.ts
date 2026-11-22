import { describe, it, expect } from 'vitest'
// `?raw` rather than node:fs — this project's tsconfig has no node types, and
// the point is to assert against the file that actually ships.
import CSS from '../viewer.html?raw'

/**
 * The HUD scrims are hand-written CSS gradients with nine stops each, whose
 * positions are `calc()` expressions. This test exists because a bad edit
 * shipped one of those stops OUT OF ORDER — `+ 16rem` where `+ 10rem` was
 * meant — and CSS responds to a decreasing stop position by clamping it to
 * its predecessor. That collapsed every stop after it onto one position and
 * turned the tail of a smooth ramp into a hard alpha step of 0.27, which is
 * a visible hard-edged band drawn across the planet.
 *
 * Nothing caught it: the pixel probe used to verify the scrim modelled the
 * INTENDED stop geometry rather than the geometry the browser resolved, so
 * it reported a smooth ramp while the screen showed a stripe. An instrument
 * that reproduces the author's intention cannot detect the author's typo.
 * These assertions read the shipped file.
 */

/** Reads a balanced parenthesised group starting at `open` (the index of "("). */
function balanced(src: string, open: number): { body: string, end: number } {
  let depth = 0
  for (let i = open; i < src.length; i++) {
    if (src[i] === '(') depth++
    else if (src[i] === ')') {
      depth--
      if (depth === 0) return { body: src.slice(open + 1, i), end: i + 1 }
    }
  }
  throw new Error('unbalanced parentheses in gradient')
}

/**
 * Every `<colour> <position>` pair of a named gradient, in source order.
 * Hand-parsed rather than regexed: the colours are `rgba(var(--x), calc(var(--y) * k))`
 * and the positions are `calc(var(--z) + 1.5rem + Nrem)`, both of which nest
 * parentheses deeply enough that a regex silently mis-parses them — which is
 * how the first version of this test managed to fail against correct CSS.
 */
function gradientStops(css: string, declaration: string): Array<{ alpha: number, offsetRem: number }> {
  const declAt = css.indexOf(declaration)
  expect(declAt, `declaration not found: ${declaration}`).toBeGreaterThan(-1)
  // The declaration string ends INSIDE `linear-gradient(`, so the paren to
  // balance from is the first one at or after declAt, not one past it.
  const { body } = balanced(css, css.indexOf('(', declAt))
  const out: Array<{ alpha: number, offsetRem: number }> = []
  let i = 0
  while (true) {
    const at = body.indexOf('rgba(', i)
    if (at === -1) break
    const colour = balanced(body, at + 4)
    // Alpha is the final top-level comma-separated term of the rgba().
    let depth = 0
    let lastComma = -1
    for (let k = 0; k < colour.body.length; k++) {
      const ch = colour.body[k]
      if (ch === '(') depth++
      else if (ch === ')') depth--
      else if (ch === ',' && depth === 0) lastComma = k
    }
    const alphaTerm = colour.body.slice(lastComma + 1).trim()
    const mult = /\*\s*([\d.]+)/.exec(alphaTerm)
    const alpha = mult ? Number(mult[1]) : alphaTerm.startsWith('var(') ? 1 : Number(alphaTerm)
    // The position follows the colour, up to the next top-level comma.
    const rest = body.slice(colour.end)
    const posEnd = (() => {
      let d = 0
      for (let k = 0; k < rest.length; k++) {
        if (rest[k] === '(') d++
        else if (rest[k] === ')') d--
        else if (rest[k] === ',' && d === 0) return k
      }
      return rest.length
    })()
    const pos = rest.slice(0, posEnd).trim()
    const rem = /\+\s*([\d.]+)rem\s*\)?\s*$/.exec(pos)
    out.push({ alpha, offsetRem: rem ? Number(rem[1]) : 0 })
    i = colour.end
  }
  return out
}

const GRADIENTS = [
  ['bottom scrim, vertical falloff', 'background: linear-gradient(to top,'],
  ['bottom scrim, horizontal mask', 'mask-image: linear-gradient(to right,'],
  ['top rail scrim', 'background: linear-gradient(to bottom,'],
] as const

describe.each(GRADIENTS)('%s', (_name, declaration) => {
  const stops = gradientStops(CSS, declaration)

  it('has the full nine-stop smoothstep ramp plus its origin stop', () => {
    expect(stops.length).toBeGreaterThanOrEqual(9)
  })

  it('never lets a stop position go backwards', () => {
    // THE test. A decreasing position is silently clamped by CSS, which
    // collapses the remaining stops and produces a hard edge.
    for (let i = 1; i < stops.length; i++) {
      expect(
        stops[i]!.offsetRem,
        `stop ${i} sits at ${stops[i]!.offsetRem}rem, behind stop ${i - 1} at ${stops[i - 1]!.offsetRem}rem`,
      ).toBeGreaterThanOrEqual(stops[i - 1]!.offsetRem)
    }
  })

  it('fades monotonically to fully transparent', () => {
    for (let i = 1; i < stops.length; i++) {
      expect(stops[i]!.alpha).toBeLessThanOrEqual(stops[i - 1]!.alpha)
    }
    expect(stops.at(-1)!.alpha).toBe(0)
  })

  it('never steps by more than 0.2 of the scrim alpha between adjacent stops', () => {
    // A smoothstep ramp's largest step is ~0.18 at its midpoint. Anything
    // materially above that is a ramp that will read as an edge.
    for (let i = 1; i < stops.length; i++) {
      const step = stops[i - 1]!.alpha - stops[i]!.alpha
      expect(step, `stop ${i - 1}->${i} drops ${step.toFixed(3)}`).toBeLessThanOrEqual(0.2)
    }
  })

  it('starts flat, so the ramp joins the plateau without a slope break', () => {
    // smoothstep begins with zero derivative; a linear ramp does not. This is
    // what stops a Mach band appearing where the fade meets the solid region.
    const firstDrop = stops[1]!.alpha - stops[2]!.alpha
    const midDrop = stops[4]!.alpha - stops[5]!.alpha
    expect(firstDrop).toBeLessThan(midDrop)
  })
})
