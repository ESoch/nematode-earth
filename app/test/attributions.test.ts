// @ts-ignore — tests run in Node; the browser tsconfig does not include Node types.
import { existsSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import ABOUT_HTML from '../about.html?raw'

// Public navigation/citation contracts, not a snapshot of the methods prose.
const credits = ABOUT_HTML.match(/<section\b[^>]*\bid="credits"[^>]*>([\s\S]*?)<\/section>/)?.[1] ?? ''

describe('public credits', () => {
  it('provides one static credits destination reachable from the methods page', () => {
    expect([...ABOUT_HTML.matchAll(/\bid="credits"/g)]).toHaveLength(1)
    expect(ABOUT_HTML).toMatch(/<a\b[^>]*href="#credits"/)
    expect(credits).toMatch(/<h2\b/)
  })

  it('links the actual research and data sources from the concise credits', () => {
    const links = [...credits.matchAll(/\bhref="([^"]+)"/g)].map((match) => match[1])
    for (const url of [
      'https://archive.org/details/yoa1914/page/472/mode/2up',
      'https://doi.org/10.1038/s41586-019-1418-6',
      'https://doi.org/10.1038/s41597-020-0437-3',
      'https://doi.org/10.3929/ethz-b-000354035',
      'https://doi.org/10.3929/ethz-b-000354394',
      'https://doi.org/10.6084/m9.figshare.11733897',
      'https://doi.org/10.25921/fd45-gt74',
      'https://doi.org/10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA',
      'https://rightsstatements.org/page/InC-NC/1.0/',
    ]) expect(links, `missing clickable source: ${url}`).toContain(url)
  })

  it('ships the installed three.js license verbatim with the static page', () => {
    const license = readFileSync(new URL('../node_modules/three/LICENSE', import.meta.url), 'utf8').trim()
    const shipped = ABOUT_HTML.match(/<pre\b[^>]*id="three-license"[^>]*>([\s\S]*?)<\/pre>/)?.[1] ?? ''
    expect(shipped.trim()).toBe(license)
  })

  it('ships the project software license with the static page', () => {
    const license = readFileSync(new URL('../../LICENSE', import.meta.url), 'utf8').trim()
    expect(credits).toContain(license)
  })

  it('preserves the same license in the repository notices', () => {
    const path = new URL('../../THIRD_PARTY_NOTICES.md', import.meta.url)
    const notices = existsSync(path) ? readFileSync(path, 'utf8') : ''
    const license = readFileSync(new URL('../node_modules/three/LICENSE', import.meta.url), 'utf8').trim()
    expect(notices).toContain(license)
  })

  it('carries the Vite browser helper notice in source and static distributions', () => {
    const viteLicense = readFileSync(new URL('../node_modules/vite/LICENSE.md', import.meta.url), 'utf8')
    const core = viteLicense.match(/\n(MIT License\n[\s\S]+?)\n# /)?.[1]
    expect(core).toContain('Permission is hereby granted')
    const notice = core!.trim()
    expect(credits).toContain(notice)
    expect(readFileSync(new URL('../../THIRD_PARTY_NOTICES.md', import.meta.url), 'utf8')).toContain(notice)
  })

  it('preserves the same upstream ACES shader notice in the page and the repository', () => {
    const notice = credits.match(/<pre id="ltc-license">([\s\S]+?)<\/pre>/)?.[1]?.trim()
    expect(notice).toContain('Copyright (c) 2017, Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt.')
    expect(notice).toContain('Real-Time Polygonal-Light Shading with Linearly Transformed Cosines.')
    expect(notice).toContain('OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.')
    expect(readFileSync(new URL('../../THIRD_PARTY_NOTICES.md', import.meta.url), 'utf8')).toContain(notice)
  })
})
