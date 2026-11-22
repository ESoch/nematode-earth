import { afterEach, test } from 'node:test'
import assert from 'node:assert/strict'
import { build, createServer } from 'vite'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { siteOrigin } from '../site-metadata.mjs'

const root = resolve(import.meta.dirname, '../..')
const sources = {
  'city_calibration.py.txt': '../pipeline/build/city_calibration.py',
  'optical.ts.txt': 'src/math/optical.ts',
  'descent.ts.txt': 'src/math/descent.ts',
  'veilDither.test.ts.txt': 'test/veilDither.test.ts',
  'rendering-checks.txt': '../docs/rendering-checks.md',
  'CITATION.cff': '../CITATION.cff',
}
const temporary = []
afterEach(async () => {
  await Promise.all(temporary.splice(0).map(path => rm(path, { recursive: true, force: true })))
})

test('production root is the exhibit, with a legacy alias and no GPU harness', async () => {
  const outDir = await mkdtemp(resolve(tmpdir(), 'nematode-release-entry-'))
  temporary.push(outDir)
  await build({ root, logLevel: 'silent', build: { outDir, emptyOutDir: true, copyPublicDir: false } })
  const html = await readFile(resolve(outDir, 'index.html'), 'utf8')
  assert.equal(html, await readFile(resolve(outDir, 'viewer.html'), 'utf8'))
  assert.ok(html.includes('id="no-webgpu"'))
  assert.ok((await readFile(resolve(outDir, 'about.html'), 'utf8')).includes('id="credits"'))
  const origin = siteOrigin(process.env.SITE_URL)
  for (const [file, path] of [['index.html', '/'], ['about.html', '/about']]) {
    const page = await readFile(resolve(outDir, file), 'utf8')
    assert.ok(page.includes(`rel="canonical" href="${origin}${path}"`))
    assert.ok(page.includes(`property="og:image" content="${origin}/poster_veil_11500km_v1.jpg"`))
    assert.ok(page.includes('name="twitter:card" content="summary_large_image"'))
    assert.ok(page.includes('href="/favicon.svg"'))
  }
  await assert.rejects(readFile(resolve(outDir, 'experiments.html')))
  assert.deepEqual((await readdir(resolve(outDir, 'sources'))).sort(), Object.keys(sources).sort())
  for (const [name, file] of Object.entries(sources)) {
    assert.deepEqual(await readFile(resolve(outDir, 'sources', name)), await readFile(resolve(root, file)), name)
  }
})

test('development serves the cited originals as plain text', async () => {
  const server = await createServer({
    root, logLevel: 'silent', server: { host: '127.0.0.1', port: 0, hmr: false },
  })
  try {
    await server.listen()
    const origin = server.resolvedUrls.local[0]
    for (const [name, file] of Object.entries(sources)) {
      const response = await fetch(new URL(`/sources/${name}`, origin))
      assert.equal(response.status, 200, name)
      assert.match(response.headers.get('content-type'), /^text\/plain\b/)
      assert.deepEqual(Buffer.from(await response.arrayBuffer()), await readFile(resolve(root, file)), name)
    }
    const missing = await fetch(new URL('/sources/package.json', origin))
    assert.equal(missing.status, 404)
  } finally {
    await server.close()
  }
})
