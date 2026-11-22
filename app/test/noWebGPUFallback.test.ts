import { describe, it, expect } from 'vitest'
// `?raw` rather than node:fs — the test/hudScrim.test.ts precedent. The point
// is to assert against the files that actually ship.
import VIEWER_HTML from '../viewer.html?raw'
import VIEWER_TS from '../src/viewer.ts?raw'
import { NO_WEBGPU_COPY, ABOUT_LINK_HREF } from '../src/hud/copy'
import { COBB_LINE, T_END } from '../src/sweep/Timeline'
import { playbackSeconds } from '../src/sweep/pacing'

/**
 * What a visitor without WebGPU is given.
 *
 * The whole of that visitor's page used to be:
 *
 *   hud.textContent = 'WebGPU required — Chrome 144+, Edge, or Safari 26+.'
 *
 * One sentence on a black page. `textContent =` replaces the HUD wholesale,
 * and every other element — including the link to the 10,800-word methods
 * page, which needs no GPU whatsoever — was constructed inside the `else`.
 * So the visitors least able to see the piece were also the only ones with no
 * route to the part of it they could have read.
 *
 * The fallback is STATIC MARKUP in `viewer.html`, revealed by CSS, not built
 * by the module. That is deliberate: `viewer.ts` is an ES module with
 * top-level `await` in it, and an old browser is exactly the browser that may
 * fail to run it at all. A fallback assembled by the script that cannot run
 * is not a fallback. These assertions therefore read the HTML.
 */

/** Text content of the fallback section, tags stripped, whitespace collapsed. */
const fallbackHtml = (() => {
  const at = VIEWER_HTML.indexOf('<main id="no-webgpu"')
  expect(at, 'no <main id="no-webgpu"> in viewer.html').toBeGreaterThan(-1)
  const end = VIEWER_HTML.indexOf('</main>', at)
  expect(end, 'unterminated <main id="no-webgpu">').toBeGreaterThan(at)
  return VIEWER_HTML.slice(at, end)
})()

const fallbackText = fallbackHtml
  .replace(/<[^>]*>/g, ' ')
  .replace(/\s+/g, ' ')
  .trim()

describe('the fallback is reachable without running the module', () => {
  it('lives in viewer.html as markup, not in viewer.ts as DOM calls', () => {
    expect(fallbackText.length).toBeGreaterThan(400)
  })

  it('is gated by a CLASSIC head script, so no module syntax stands between the visitor and it', () => {
    const head = VIEWER_HTML.slice(0, VIEWER_HTML.indexOf('</head>'))
    const gate = /<script>([\s\S]*?)<\/script>/g
    const bodies = [...head.matchAll(gate)].map((m) => m[1] ?? '')
    const gateScript = bodies.find((b) => b.includes('data-webgpu'))
    expect(gateScript, 'no classic head script sets data-webgpu').toBeDefined()
    // No module-only syntax in the one script an old browser must survive.
    expect(gateScript).not.toMatch(/\bawait\b|=>|\bconst\b|\blet\b|\?\./)
    expect(gateScript).toMatch(/navigator/)
  })

  it('shows itself when the gate never ran at all, and hides only on an explicit yes', () => {
    // A visitor with scripting off gets the fallback rather than a black page:
    // the module could not have run for them either. The hiding rule keys on
    // the POSITIVE value, so the absent attribute is the visible state.
    expect(VIEWER_HTML).toMatch(/html\[data-webgpu="yes"\]\s+#no-webgpu\s*\{[^}]*display:\s*none/)
    expect(VIEWER_HTML).not.toMatch(
      /html\[data-webgpu="no"\]\s+#no-webgpu\s*\{[^}]*display:\s*block/,
    )
  })

  it('scrolls, because it is a page of prose and not a canvas', () => {
    // `html, body { overflow: hidden }` is right for a full-bleed canvas and
    // wrong for this: it would clip the methods link off the bottom of a
    // phone. The hidden overflow is scoped to the case that wants it.
    expect(VIEWER_HTML).toMatch(/html\[data-webgpu="yes"\][^{]*\{[^}]*overflow:\s*hidden/)
  })

  it('hides the canvas and the HUD chrome, so nothing paints over it', () => {
    for (const selector of ['#hud', '#hud-scrim', 'canvas']) {
      const re = new RegExp(
        `html:not\\(\\[data-webgpu="yes"\\]\\)[^{]*${selector.replace('#', '#')}[^{]*\\{[^}]*display:\\s*none`,
      )
      expect(VIEWER_HTML, `${selector} still paints over the fallback`).toMatch(re)
    }
  })
})

describe('what the fallback actually says', () => {
  it("carries Cobb's 1914 line verbatim — the heart of the piece needs no GPU", () => {
    expect(fallbackText).toContain(COBB_LINE)
  })

  it('attributes it, as the live HUD does', () => {
    expect(fallbackText).toContain('Cobb')
    expect(fallbackText).toContain('1914')
  })

  it('explains the JavaScript and WebGPU requirements even when no detection ran', () => {
    expect(NO_WEBGPU_COPY.requirement).toContain('WebGPU')
    expect(NO_WEBGPU_COPY.requirement).toContain('JavaScript')
  })

  it('prints the complete fallback copy when JavaScript is disabled', () => {
    const { adapterFailure, remedyLinkHref, ...printed } = NO_WEBGPU_COPY
    for (const value of Object.values(printed)) expect(fallbackText).toContain(value)
    expect(fallbackHtml).toContain(remedyLinkHref)
    expect(fallbackText).not.toContain(adapterFailure)
  })

  it('asserts no browser version number anywhere', () => {
    // The old string claimed "Chrome 144+, Edge, or Safari 26+". Version
    // numbers rot, and none of those three had been checked. A claim the
    // page cannot support is worse than no claim.
    for (const [name, s] of Object.entries(NO_WEBGPU_COPY)) {
      expect(s, `${name} asserts a version`).not.toMatch(/\b\d{1,3}\s*\+/)
      expect(s, `${name} asserts a version`).not.toMatch(
        /\b(?:Chrome|Chromium|Edge|Safari|Firefox)\s+\d/i,
      )
    }
  })

  it('tells the visitor what they cannot see, in terms of the piece and not of the API', () => {
    expect(NO_WEBGPU_COPY.missing).toContain('Air, water and rock fade')
    expect(NO_WEBGPU_COPY.missing).toContain('soil model')
  })

  it('keeps the fallback description aligned with the live sweep duration', () => {
    const duration = `about ${Math.round(playbackSeconds(T_END))} seconds`
    expect(NO_WEBGPU_COPY.missing).toContain(duration)
    expect(fallbackText).toContain(duration)
  })

  it('tells the visitor what would let them see it', () => {
    expect(NO_WEBGPU_COPY.remedy.length).toBeGreaterThan(40)
  })

  it('says the finding itself survives without a GPU, which is the honest part', () => {
    expect(NO_WEBGPU_COPY.aboutLead.toLowerCase()).toMatch(/finding|method|read/)
  })
})

describe('the route to the methods page', () => {
  it('links to about.html, prominently and by the same href the HUD uses', () => {
    expect(ABOUT_LINK_HREF).toBe('/about.html')
    expect(fallbackHtml).toContain(`href="${ABOUT_LINK_HREF}"`)
  })

  it('uses the same href in the live HUD, so one edit cannot fix half of them', () => {
    expect(VIEWER_TS).toContain('ABOUT_LINK_HREF')
    expect(VIEWER_TS).not.toMatch(/about\.href\s*=\s*['"]/)
  })

  it('gives the link a target big enough to hit with a thumb', () => {
    expect(VIEWER_HTML).toMatch(/\.nw-about\s*\{[^}]*min-height:\s*(\d+(?:\.\d+)?)(px|rem)/)
  })
})

describe('the still image', () => {
  /**
   * A frame of the real thing — the last beat of the sweep at 11,500 km, with
   * Earth's matter gone and only the nematode veil left. Cropped to the globe
   * so no stale HUD copy travels with it.
   *
   * `import.meta.glob` enumerates the real filesystem, so this fails when the
   * asset is renamed, deleted, or never committed — the deploy-only 404 this
   * project has already been bitten by once, with about.html.
   */
  const inPublic = import.meta.glob('../public/*.{avif,jpg,jpeg,png,webp,gif}', {
    eager: true,
    query: '?url',
    import: 'default',
  })
  const names = Object.keys(inPublic)
    .map((p) => p.replace('../public/', ''))
    .sort()

  it('exists on disk, in both the modern and the universal format', () => {
    expect(names).toEqual(['poster_veil_11500km_v1.avif', 'poster_veil_11500km_v1.jpg'])
  })

  it('is referenced by the fallback, and every referenced file exists', () => {
    const referenced = [...fallbackHtml.matchAll(/(?:src|srcset)="\/([^"]+)"/g)].map((m) => m[1]!)
    expect(referenced.length).toBeGreaterThan(0)
    for (const r of referenced) {
      expect(names, `${r} is referenced but not in public/`).toContain(r)
    }
  })

  it('offers AVIF first and falls back to JPEG, since an old browser is the whole audience', () => {
    expect(fallbackHtml).toMatch(/<source[^>]+type="image\/avif"/)
    expect(fallbackHtml).toMatch(/<img[^>]+src="\/poster_veil_11500km_v1\.jpg"/)
  })

  it('is not a data URI — the bytes stay a cacheable file', () => {
    expect(fallbackHtml).not.toContain('data:image')
  })

  it('declares its intrinsic size, so the prose does not jump when it loads', () => {
    expect(fallbackHtml).toMatch(/width="1200"/)
    expect(fallbackHtml).toMatch(/height="800"/)
  })

  it('describes itself for a reader who cannot see it', () => {
    const alt = /<img[^>]+alt="([^"]*)"/.exec(fallbackHtml)?.[1] ?? ''
    expect(alt.length).toBeGreaterThan(40)
    expect(alt.toLowerCase()).not.toMatch(/^image of|^picture of|screenshot/)
  })

  it('is versioned in its filename, because public/ paths are not content-hashed', () => {
    // Same discipline as _headers' note on /data/*: a path cached immutably
    // must mean one thing forever, so a re-capture ships under a new name.
    for (const n of names) expect(n).toMatch(/_v\d+\.[a-z]+$/)
  })
})

describe('viewer.ts no longer owns the fallback', () => {
  it('has dropped the one-sentence dead end', () => {
    expect(VIEWER_TS).not.toContain('WebGPU required')
    expect(VIEWER_TS).not.toMatch(/hud\.textContent\s*=/)
  })

  it('routes an adapter that fails to the same fallback rather than a black page', () => {
    // navigator.gpu can be present while requestAdapter returns null, or
    // device creation throws. That visitor used to get an unhandled rejection
    // and an empty page — strictly worse than the one sentence.
    expect(VIEWER_TS).toContain('NO_WEBGPU_COPY.adapterFailure')
    expect(VIEWER_TS).toMatch(/catch/)
  })
})
