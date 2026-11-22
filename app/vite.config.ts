import { defineConfig, type Plugin } from 'vite'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { siteOrigin } from './scripts/site-metadata.mjs'

// The methods page cites these exact files. Only these names are public.
const citedSources = new Map([
  ['/city_calibration.py.txt', resolve(__dirname, '../pipeline/build/city_calibration.py')],
  ['/optical.ts.txt', resolve(__dirname, 'src/math/optical.ts')],
  ['/descent.ts.txt', resolve(__dirname, 'src/math/descent.ts')],
  ['/veilDither.test.ts.txt', resolve(__dirname, 'test/veilDither.test.ts')],
  ['/CITATION.cff', resolve(__dirname, '../CITATION.cff')],
])

// One source of truth for both the public root and old viewer links. The
// experiment harness remains available in development only.
function exhibitEntry(): Plugin {
  const origin = siteOrigin(process.env.SITE_URL)
  return {
    name: 'exhibit-entry',
    enforce: 'post',
    transformIndexHtml(_html, context) {
      if (!['/viewer.html', '/about.html'].includes(context.path)) return []
      const about = context.path === '/about.html'
      const image = `${origin}/poster_veil_11500km_v1.jpg`
      return [
        { tag: 'link', attrs: { rel: 'canonical', href: `${origin}${about ? '/about' : '/'}` }, injectTo: 'head' },
        { tag: 'link', attrs: { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }, injectTo: 'head' },
        { tag: 'meta', attrs: { property: 'og:url', content: `${origin}${about ? '/about' : '/'}` }, injectTo: 'head' },
        { tag: 'meta', attrs: { property: 'og:image', content: image }, injectTo: 'head' },
        { tag: 'meta', attrs: { property: 'og:image:width', content: '1200' }, injectTo: 'head' },
        { tag: 'meta', attrs: { property: 'og:image:height', content: '800' }, injectTo: 'head' },
        { tag: 'meta', attrs: { property: 'og:image:alt', content: 'The soil nematode model rendered as a translucent globe against a dark background.' }, injectTo: 'head' },
        { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' }, injectTo: 'head' },
      ]
    },
    configureServer(server) {
      server.middlewares.use('/sources', async (req, res, next) => {
        const file = citedSources.get(req.url?.split('?')[0] ?? '')
        if (!file) {
          res.writeHead(404).end()
          return
        }
        if (req.method !== 'GET' && req.method !== 'HEAD') {
          res.writeHead(405, { Allow: 'GET, HEAD' }).end()
          return
        }
        try {
          const source = await readFile(file)
          res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'Content-Length': source.length })
          res.end(req.method === 'HEAD' ? undefined : source)
        } catch (error) {
          next(error)
        }
      })
      server.middlewares.use((req, _res, next) => {
        const [path, query] = (req.url ?? '/').split('?')
        if (path === '/' || path === '/index.html') {
          req.url = `/viewer.html${query === undefined ? '' : `?${query}`}`
        }
        next()
      })
    },
    async generateBundle(_options, bundle) {
      const viewer = bundle['viewer.html']
      if (!viewer || viewer.type !== 'asset') throw new Error('Exhibit entry was not built')
      this.emitFile({ type: 'asset', fileName: 'index.html', source: viewer.source })
      for (const [name, file] of citedSources) {
        this.emitFile({ type: 'asset', fileName: `sources${name}`, source: await readFile(file) })
      }
    },
  }
}

export default defineConfig({
  plugins: [exhibitEntry()],
  server: { port: 5173 },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: {
        // The plugin also emits this exact exhibit at index.html.
        viewer: resolve(__dirname, 'viewer.html'),
        // The methods page. Static prose with an inline stylesheet and no
        // script of its own, but it MUST be listed here: Vite only emits the
        // HTML entry points it is given, so without this line about.html
        // simply never appears in dist/ and the link from the viewer 404s on
        // deploy while working fine under `pnpm dev` (which serves the source
        // tree). That asymmetry is exactly the kind of silent, deploy-only
        // failure the rest of this project is built to avoid.
        about: resolve(__dirname, 'about.html'),
      },
    },
  },
})
