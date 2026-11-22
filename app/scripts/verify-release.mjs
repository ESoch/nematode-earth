import assert from 'node:assert/strict'
import { readFile, stat } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { brotliDecompressSync } from 'node:zlib'
import { sha256, walkFiles, readHeaderRules, headersFor } from './release-files.mjs'
import { siteRevision } from './site-metadata.mjs'

export async function verifyReleaseData(root, { maxAssetBytes = 25 * 1024 * 1024 } = {}) {
  const manifest = JSON.parse(await readFile(resolve(root, 'release-data.json'), 'utf8'))
  assert.ok(manifest.schema === 1 && manifest.encoding === 'br' && manifest.files?.length, 'Invalid release manifest')
  const all = await walkFiles(root)
  const actual = all.filter(path => path.startsWith('data/') && path.endsWith('.bin'))
  assert.deepEqual(manifest.files.map(file => file.path).sort(), actual, 'Release data inventory mismatch')
  assert.ok(all.length <= 20000, 'Cloudflare Pages file-count limit exceeded')
  for (const path of all) {
    assert.ok((await stat(resolve(root, path))).size <= maxAssetBytes, `Asset size exceeds hosting limit: ${path}`)
  }
  const rules = await readHeaderRules(root)
  let decodedBytes = 0
  let encodedBytes = 0
  for (const file of manifest.files) {
    assert.equal(headersFor(rules, `/${file.path}`).get('content-encoding'), 'br', `Missing brotli encoding header: ${file.path}`)
    const encoded = await readFile(resolve(root, file.path))
    assert.ok(encoded.length === file.encodedBytes && sha256(encoded) === file.encodedSha256, `Encoded integrity/hash mismatch: ${file.path}`)
    const decoded = brotliDecompressSync(encoded)
    assert.ok(decoded.length === file.decodedBytes && sha256(decoded) === file.decodedSha256, `Decoded integrity/hash mismatch: ${file.path}`)
    decodedBytes += decoded.length
    encodedBytes += encoded.length
  }
  return { files: actual.length, decodedBytes, encodedBytes }
}

export async function verifyRelease(root) {
  const report = await verifyReleaseData(root)
  const { revision } = JSON.parse(await readFile(resolve(root, 'release.json'), 'utf8'))
  assert.equal(revision, siteRevision(revision), 'Invalid release revision')
  if (process.env.SITE_REVISION) assert.equal(revision, siteRevision(process.env.SITE_REVISION), 'Release revision differs from build')
  const html = await readFile(resolve(root, 'index.html'), 'utf8')
  assert.equal(html, await readFile(resolve(root, 'viewer.html'), 'utf8'), 'Root and legacy exhibit must match')
  assert.ok(html.includes('id="no-webgpu"'), 'Missing non-WebGPU fallback')
  const about = await readFile(resolve(root, 'about.html'), 'utf8')
  assert.ok(about.includes('id="credits"'), 'Missing credits')
  for (const page of [html, about]) {
    assert.match(page, /rel="canonical" href="https:\/\/[^"<>]+"/, 'Missing absolute canonical URL')
    assert.match(page, /property="og:image" content="https:\/\/[^"<>]+"/, 'Missing absolute sharing image')
  }
  assert.ok(!(await walkFiles(root)).includes('experiments.html'), 'Engineering harness must not ship')
  for (const file of ['favicon.svg', '404.html', 'poster_veil_11500km_v2.avif', 'poster_veil_11500km_v2.jpg']) await stat(resolve(root, file))
  return report
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const report = await verifyRelease(resolve(import.meta.dirname, '../dist'))
  console.log(`Release verified: ${report.files} lossless brotli files, ${report.encodedBytes.toLocaleString()} wire bytes / ${report.decodedBytes.toLocaleString()} decoded bytes; public routes, fallback, credits and share metadata present.`)
}
