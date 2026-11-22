import { createServer } from 'node:http'
import { readFile, realpath } from 'node:fs/promises'
import { extname, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { readHeaderRules, headersFor } from './release-files.mjs'
import { verifyRelease } from './verify-release.mjs'

const types = { '.html': 'text/html; charset=utf-8', '.js': 'application/javascript', '.css': 'text/css', '.json': 'application/json', '.bin': 'application/octet-stream', '.jpg': 'image/jpeg', '.avif': 'image/avif', '.svg': 'image/svg+xml', '.txt': 'text/plain' }
const notFound = () => new Response('Not found', { status: 404 })

// A loopback-only preview of our static release, including its encoding headers.
// Hosted redirects, edge caching and TLS still require post-deploy verification.
export async function serveRequest(root, request) {
  let path
  try { path = decodeURIComponent(new URL(request.url).pathname) } catch { return notFound() }
  if (path.includes('\\') || path.includes('\0') || path.split('/').some(part => part.startsWith('.') || part.startsWith('_')) || path === '/release-data.json') return notFound()
  if (!['GET', 'HEAD'].includes(request.method)) return new Response('Method not allowed', { status: 405 })
  const file = ({ '/': '/index.html', '/about': '/about.html', '/viewer': '/viewer.html' })[path] ?? path
  try {
    const resolvedRoot = await realpath(root)
    const resolvedFile = await realpath(resolve(root, `.${file}`))
    if (!resolvedFile.startsWith(resolvedRoot + sep)) return notFound()
    const bytes = await readFile(resolvedFile)
    const headers = headersFor(await readHeaderRules(root), path)
    if (!headers.has('content-type')) headers.set('content-type', types[extname(file)] ?? 'application/octet-stream')
    headers.set('content-length', String(bytes.length))
    return new Response(request.method === 'HEAD' ? null : bytes, { headers })
  } catch (error) {
    if (['ENOENT', 'EISDIR', 'ENOTDIR'].includes(error.code)) return notFound()
    throw error
  }
}

export async function preparePreview(root) {
  await verifyRelease(root)
  return request => serveRequest(root, request)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const root = resolve(import.meta.dirname, '../dist')
  const port = Number(process.env.PORT ?? 4173)
  const handle = await preparePreview(root)
  createServer(async (req, res) => {
    try {
      const response = await handle(new Request(`http://localhost${req.url}`, { method: req.method }))
      res.writeHead(response.status, Object.fromEntries(response.headers))
      res.end(Buffer.from(await response.arrayBuffer()))
    } catch {
      res.writeHead(500, { 'content-type': 'text/plain' })
      res.end('Preview failed; check the release build.')
    }
  }).listen(port, '127.0.0.1', () => console.log(`Release preview: http://127.0.0.1:${port}`))
}
