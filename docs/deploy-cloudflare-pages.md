# Deploying the exhibit on Cloudflare Pages

The site is a static build of the soil exhibit. Its default canonical origin is [nematode.earth](https://nematode.earth/). The [September 26 release record](release-evidence-2026-09-26.md) documents the original direct upload and checks on that deployment.

## Build and preview

From `app/`, with Node.js 22.12+ and pnpm 10, run

```sh
pnpm install --frozen-lockfile
pnpm build
pnpm check:release
pnpm preview
```

The build recreates `app/dist/`, checks TypeScript, builds the public pages, compresses the binary data, and verifies the result. For a fork, set `SITE_URL` to its public HTTPS origin at build time. The origin may contain a hostname only, without credentials, port, path, query or fragment.

The preview server binds to `127.0.0.1:4173` by default and supports a `PORT` override. It serves the release’s `_headers` rules so the browser decodes the textures correctly. The diagnostic `build:uncompressed` output requires different headers and includes files that exceed Pages’ upload limit.

| Route | Contents |
| --- | --- |
| `/` and `/viewer.html` | WebGPU exhibit and static fallback |
| `/about` and `/about.html` | Science, methods, sources, and credits |
| `/sources/*` | Plain-text copies of the cited calculations and software citation, drawn from the source revision at build time |
| `/release.json` | Git commit used for the hosted build. A local build with no declared revision reports `null`. |
| Missing path | HTTP 404 with the static error page |
| `/experiments.html` and `/soil-validation.html` | Development-server tools, excluded from the production build |

## Data integrity and caching

The build compresses `dist/data/**/*.bin` with Brotli because the default edge compression list omits `application/octet-stream`. An interrupted or repeated compression pass fails. Rebuild from source for a fresh artifact. The verifier reads `release-data.json`, checks both SHA-256 hashes and lengths, decodes each binary, and checks encoding headers, page outputs, and file-size limits. [Cloudflare compression documentation](https://developers.cloudflare.com/speed/optimization/content/compression/)

| Files | Cache policy |
| --- | --- |
| Content-hashed scripts and styles | One year, immutable |
| Data binaries and JSON sidecars | One year, immutable. Give replacement data a new versioned path and update its loader. |
| Versioned poster | One year, immutable. Give a replacement a new filename. |
| HTML, clean page URLs, citation files, and favicon | Revalidate |

Pages combines matching header rules. Keep cache declarations for these groups disjoint. The release verifier checks the compressed output against the 25 MiB per-asset limit and the 20,000-file limit for Wrangler uploads. The dashboard accepts ZIPs with up to 1,000 files. [Pages headers](https://developers.cloudflare.com/pages/configuration/headers/) · [Direct Upload limits](https://developers.cloudflare.com/pages/get-started/direct-upload/#limits)

## Hosting

Each push to `main` in `ESoch/nematode-earth` runs the exhibit and pipeline checks. Once both pass, **Deploy and verify production** downloads that run’s verified artifact, checks its embedded revision against the commit, and uploads it to the existing `nematode-earth` Pages project. Deployments run one at a time. An older run stops before upload if a newer commit has reached `main`. GitHub retains the build ZIP for 14 days. [Workflow runs and artifacts](https://github.com/ESoch/nematode-earth/actions/workflows/checks.yml)

After upload, the workflow polls the custom domain with fresh requests and compares `release.json` with the full commit SHA. It retries while the change propagates and fails the deployment check if the revision still differs. The verified commit appears in the workflow summary. To repeat the check from a checkout of the deployed commit, run `node app/scripts/verify-live-revision.mjs "$(git rev-parse HEAD)"` from the repository root. [Live revision](https://nematode.earth/release.json) · [GitHub main](https://github.com/ESoch/nematode-earth/commits/main/)

The GitHub `production` environment allows deployments from the `main` branch. It holds the account ID in the variable `CLOUDFLARE_ACCOUNT_ID` and an account-owned Pages Edit token in the secret `CLOUDFLARE_API_TOKEN`. Only the upload step receives the token. Cloudflare scopes this permission to the owning account. When rotating it, store the replacement in the same environment. [Cloudflare CI credentials](https://developers.cloudflare.com/pages/how-to/use-direct-upload-with-continuous-integration/) · [Account-owned tokens](https://developers.cloudflare.com/fundamentals/api/get-started/account-owned-tokens/)

For manual recovery, download the `nematode-earth-<commit>` ZIP from the successful run, choose **Create deployment → Production** in the Pages project, upload the ZIP, and run the same revision check against that commit. For a fresh manual build, set `SITE_REVISION` to the full output of `git rev-parse HEAD` in a clean checkout when running `pnpm build`. [Cloudflare Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/)

For a new Git-integrated Pages project, set the root directory to `app`, build command to `pnpm build`, and output directory to `dist`. Install the locked packages with the supported Node/pnpm versions. The repository includes the baked display data, so the host build needs no upstream scientific downloads. Pages project types have different deployment workflows. [Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/) · [Git integration](https://developers.cloudflare.com/pages/get-started/git-integration/)

For a custom hostname, configure it in Pages and follow the DNS instructions for that domain. Verify the actual hosted routes, binary decoding, caching, and device behavior for each release. [Pages custom domains](https://developers.cloudflare.com/pages/configuration/custom-domains/) · [Hosted release checks](release-checklist.md#verify-a-hosted-release)
