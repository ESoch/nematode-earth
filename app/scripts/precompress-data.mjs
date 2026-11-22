#!/usr/bin/env node
/**
 * Precompress ONLY fresh dist/data binaries, never public/ source data.
 * The matching _headers is mandatory: undeclared brotli is corrupt to a browser.
 * An exclusive build marker prevents a second pass or resuming a partial pass.
 * Do not detect compression by trying to decode arbitrary raw float textures:
 * some raw textures happen to be syntactically valid brotli streams.
 * Run through pnpm build, which first recreates dist from source.
 */
import { brotliCompressSync, constants } from 'node:zlib'
import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { sha256, walkFiles } from './release-files.mjs'

export async function precompressData(root, { quality = 11, onProgress = () => {} } = {}) {
  const paths = (await walkFiles(root)).filter(path => path.startsWith('data/') && path.endsWith('.bin'))
  if (!paths.length) throw new Error('No data binaries found; run a fresh build first')
  try {
    await writeFile(resolve(root, '.precompress-started'), 'Rebuild from source before compressing again.\n', { flag: 'wx' })
  } catch (error) {
    if (error.code === 'EEXIST') throw new Error('Already compressed or interrupted; run a fresh build before retrying')
    throw error
  }
  const manifest = { schema: 1, encoding: 'br', quality, files: [] }
  for (const path of paths) {
    const original = await readFile(resolve(root, path))
    const compressed = brotliCompressSync(original, { params: {
      [constants.BROTLI_PARAM_QUALITY]: quality,
      [constants.BROTLI_PARAM_SIZE_HINT]: original.length,
    } })
    await writeFile(resolve(root, path), compressed)
    manifest.files.push({ path, decodedBytes: original.length, encodedBytes: compressed.length, decodedSha256: sha256(original), encodedSha256: sha256(compressed) })
    onProgress(path, manifest.files.length, paths.length)
  }
  await writeFile(resolve(root, 'release-data.json'), JSON.stringify(manifest, null, 2) + '\n')
  return manifest
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const report = await precompressData(resolve(import.meta.dirname, '../dist'), {
    onProgress: (path, done, total) => console.log(`Brotli ${done}/${total}: ${path}`),
  })
  console.log(`Precompressed ${report.files.length} files; recorded encoded and decoded SHA-256 checksums.`)
}
