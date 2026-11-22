import { afterEach, test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile, readFile, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { brotliDecompressSync } from 'node:zlib'
import { precompressData } from '../precompress-data.mjs'
import { verifyReleaseData } from '../verify-release.mjs'
import { serveRequest, preparePreview } from '../preview.mjs'

const temporary = []
afterEach(async () => {
  await Promise.all(temporary.splice(0).map(path => rm(path, { recursive: true, force: true })))
})
async function fixture() {
  const root = await mkdtemp(resolve(tmpdir(), 'nematode-release-data-'))
  temporary.push(root)
  await mkdir(resolve(root, 'data/fixture'), { recursive: true })
  const data = Buffer.from('0123456789abcdef'.repeat(100))
  await writeFile(resolve(root, 'data/fixture/face_0.bin'), data)
  await writeFile(resolve(root, 'index.html'), '<!doctype html><title>Exhibit</title>')
  await writeFile(resolve(root, 'about.html'), '<!doctype html><title>Methods</title>')
  await writeFile(resolve(root, '404.html'), 'Not found')
  await writeFile(resolve(root, '_headers'), '/data/*.bin\n  Content-Encoding: br\n  Content-Type: application/octet-stream\n  Cache-Control: public, max-age=0, must-revalidate\n/*\n  X-Content-Type-Options: nosniff\n')
  return { root, data, path: resolve(root, 'data/fixture/face_0.bin') }
}

test('compression preserves bytes and refuses a second pass over compressed output', async () => {
  const f = await fixture()
  const report = await precompressData(f.root, { quality: 1 })
  assert.equal(report.files.length, 1)
  assert.deepEqual(brotliDecompressSync(await readFile(f.path)), f.data)
  await assert.rejects(precompressData(f.root, { quality: 1 }), /already|fresh/i)
  assert.deepEqual(brotliDecompressSync(await readFile(f.path)), f.data)
  const verified = await verifyReleaseData(f.root)
  assert.equal(verified.decodedBytes, f.data.length)
})

test('release verification rejects corruption, missing compression headers and oversize assets', async () => {
  const f = await fixture()
  await precompressData(f.root, { quality: 1 })
  await assert.rejects(verifyReleaseData(f.root, { maxAssetBytes: 1 }), /size|limit/i)
  const compressed = await readFile(f.path)
  await writeFile(f.path, f.data)
  await assert.rejects(verifyReleaseData(f.root), /hash|integrity|brotli/i)
  await writeFile(f.path, compressed)
  await writeFile(resolve(f.root, '_headers'), '/*\n  X-Content-Type-Options: nosniff\n')
  await assert.rejects(verifyReleaseData(f.root), /encoding|header/i)
})

test('preview serves compressed bytes with decoding headers; missing data is 404, not HTML', async () => {
  const f = await fixture()
  await precompressData(f.root, { quality: 1 })
  const response = await serveRequest(f.root, new Request('http://localhost/data/fixture/face_0.bin'))
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('content-encoding'), 'br')
  assert.equal(response.headers.get('x-content-type-options'), 'nosniff')
  assert.deepEqual(brotliDecompressSync(Buffer.from(await response.arrayBuffer())), f.data)
  const root = await serveRequest(f.root, new Request('http://localhost/'))
  assert.equal(root.status, 200)
  assert.match(await root.text(), /Exhibit/)
  const missing = await serveRequest(f.root, new Request('http://localhost/data/missing.bin'))
  assert.equal(missing.status, 404)
  assert.equal(missing.headers.get('content-encoding'), null)
  for (const path of ['/_headers', '/release-data.json', '/.git/config', '/%2e%2e%2foutside']) {
    const denied = await serveRequest(f.root, new Request(`http://localhost${path}`))
    assert.equal(denied.status, 404)
  }
})

test('preview does not follow links to files outside its release root', async () => {
  const f = await fixture()
  const outside = await mkdtemp(resolve(tmpdir(), 'nematode-preview-outside-'))
  temporary.push(outside)
  await writeFile(resolve(outside, 'outside.html'), 'Outside the release')
  await symlink(resolve(outside, 'outside.html'), resolve(f.root, 'linked.html'))
  assert.equal((await serveRequest(f.root, new Request('http://localhost/linked.html'))).status, 404)
})

test('preview refuses to start on an uncompressed or interrupted release', async () => {
  const f = await fixture()
  await assert.rejects(preparePreview(f.root), /release-data|ENOENT/)
  const html = '<!doctype html><link rel="canonical" href="https://nematode.earth/"><meta property="og:image" content="https://nematode.earth/poster_veil_11500km_v2.jpg"><main id="no-webgpu"></main><section id="credits"></section>'
  await writeFile(resolve(f.root, 'release.json'), JSON.stringify({ revision: process.env.SITE_REVISION ?? null }))
  for (const file of ['index.html', 'viewer.html', 'about.html']) await writeFile(resolve(f.root, file), html)
  for (const file of ['favicon.svg', 'poster_veil_11500km_v2.avif', 'poster_veil_11500km_v2.jpg']) await writeFile(resolve(f.root, file), 'fixture')
  await precompressData(f.root, { quality: 1 })
  const handler = await preparePreview(f.root)
  assert.equal((await handler(new Request('http://localhost/'))).status, 200)
  await writeFile(f.path, f.data)
  await assert.rejects(preparePreview(f.root), /integrity|hash/)
})
