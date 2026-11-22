# Release evidence — September 26, 2026

On September 26, 2026, the soil exhibit was deployed to Cloudflare Pages and tested on its custom domain. The checks and outstanding items below describe that release.

## Source and environment

- Branch: `design/nematode-earth-spec`; HEAD: `fde66ed32351d05716456bcc3cb3d8ae8d7fc645`.
- Working tree: 88 modified, deleted or untracked entries at the hosted-release check. The
  deployed artifact includes those local changes; HEAD alone cannot reproduce it.
- Runtime: Node.js 26.5.0, pnpm 10.30.3, uv 0.11.23, Python 3.14.6.

## Completed local checks

- `pnpm build` completed from a clean `app/dist/` and verified the output. The build reported 85 Brotli-compressed data binaries, 58,098,583 encoded bytes and 553,147,029 decoded bytes.
- `pnpm check:release` exited 0: 961 app tests in 53 files, 8 release-script tests, TypeScript checking, and release artifact verification.
- The Python pipeline suite completed with 324 tests passed, 0 failed, 0 errors, 0 skipped.
- All 85 decoded data-file hashes in the release manifest matched their source files under `app/public/data/`.
- Local release preview started on `http://127.0.0.1:4173`. The release-script suite also tested the preview's encoded responses, missing-data 404 behavior, and path containment.
- An independent scoped review of loading and retry changes found no important findings; its focused suite passed 37 tests. The root viewer and release packaging had an earlier review, and both reported issues were fixed and retested.

## Hosted deployment and checks

- Direct-upload project: `nematode-earth` on Cloudflare Pages. The public addresses are
  `https://nematode.earth/`, `https://www.nematode.earth/` and the provider address
  `https://nematode-earth.pages.dev/`. Cloudflare reported both custom domains Active with SSL.
- The accepted upload was a ZIP of only `app/dist/`, 111 files. SHA-256:
  `b7a29657e12a3e6b02646de762c454084ce0b79a26ea53141cc0b96d3e0abf70`.
  The bundled `release-data.json` SHA-256 is
  `05cb369a791f49f847ce90af8f4cd754738e9c8d762ed8c68521a29ad20954ff`.
- `pnpm build` and the full 961-test app suite passed again after correcting stale
  methods-page copy. The 8 release-script tests passed, and the ZIP archive passed its
  integrity check. The live methods page reflects that correction.
- Browser checks on the custom domain: the opening rendered the Earth after soil and
  elevation became ready; scrubbing to the end showed the nematode-only veil, the layer
  status and the marine-data limitation. The methods and credits page loaded over HTTPS.
- HTTP checks on the custom domain: `/`, `/about` and `www` returned 200; a deliberately
  missing data URL returned 404 rather than app HTML. Entry HTML was revalidatable.
  Soil, elevation and urban atlas faces returned `Content-Encoding: br` with immutable
  caching. Their transmitted bytes were individually identical to the validated release
  files. The live Pages URL was also checked with an explicit Brotli request and decoded
  successfully against the source data.

## Open at the time of deployment

- The deployment was a direct upload from a working tree with 88 changes. The ZIP and manifest checksums identify this artifact.
- Check a physical phone, a no-WebGPU browser and a cold/failed-network loading path. The
  browser check above used a desktop WebGPU session and a successful network.
- Check social-card previews and the canonical URL in an external sharing client, and
  exercise an actual rollback. These are not established by a successful upload.

## September 26 pacing update

- The idle globe turn was shortened from 18 to 12 minutes; it still stops on interaction,
  playback, or reduced-motion preference. The authored sweep timeline remains directly
  scrubbable, but playback now spends 4 seconds on the invitation, 24 seconds on visible
  air/water/rock removal, and 5.5 seconds settling on the nematode veil. It ends at
  33.5 seconds rather than leaving a 22-second static interval after solid Earth is gone.
- Local checks passed: 969 app tests in 54 files, 8 release-script tests, TypeScript
  checking, `git diff --check`, and the production build's 85-file lossless-Brotli
  verification. The data manifest SHA-256 remains
  `05cb369a791f49f847ce90af8f4cd754738e9c8d762ed8c68521a29ad20954ff`.
- Browser verification on the local preview observed water and rock leaving together,
  then the “What remains” state, with playback stopping at the new end. Cloudflare Pages
  accepted the production upload and reported Success. The uploaded archive contained
  111 files and passed ZIP integrity checks; its SHA-256 is
  `4482f5de7a02c2c0f2a8825ce671c0dd9c0671399970918d98d01ce394b10809`.
- The custom domain loaded the new `viewer-PcWVZHS0.js` asset, rendered the opening,
  and displayed `0:33` at the hosted scrub endpoint with all three layers marked gone.
  The browser was returned to the opening view afterward. This is a desktop browser
  check, not a physical-device or measured-frame-rate certification.
- The source checkout was at `fde66ed32351d05716456bcc3cb3d8ae8d7fc645` with uncommitted changes at the time of upload. The archive checksum identifies the exact hosted build.
