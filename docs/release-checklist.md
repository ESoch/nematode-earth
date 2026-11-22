# Release checklist

Use this checklist for each new version of the soil exhibit. Record the revision, runtimes, build date, and test results alongside the artifact. The September 26 record documents an earlier release. [Historical release evidence](release-evidence-2026-09-26.md)

## Check a clean checkout

- [ ] Install the locked dependencies and run the viewer and pipeline checks in the contributor guide. Record optional integration tests that skipped because their research inputs were unavailable. [Contributor guide](../CONTRIBUTING.md#run-and-check-a-change)
- [ ] From `app/`, run `pnpm build`, then `pnpm check:release`. The verifier checks the binary inventory, compressed and decoded hashes, response-header rules, public pages, and artifact limits. Compare each decoded data hash with its corresponding source under `app/public/data/`.
- [ ] Check the output inventory for unexpected files. Confirm the exhibit, compatibility entry, methods, credits, and 404 are present. The GPU harness and soil validation page belong in the development server only.
- [ ] Review the rights and attribution for each changed source or dependency. Check that the built methods page includes the notices and that the repository citation still identifies the software. [Third-party notices](../THIRD_PARTY_NOTICES.md) · [Software citation](../CITATION.cff)
- [ ] Give changed immutable data or poster assets new URLs and update their consumers in the same release.

## Exercise the production preview

Run `pnpm preview` from `app/` and open the printed URL. This server applies the release’s Brotli headers.

- [ ] Play and scrub the full opening. Orbit, zoom to the soil detail limit and back out, then try the sample, uncertainty, and city controls. Confirm the viewer loaded the soil data.
- [ ] Use the controls with a keyboard and at a narrow width. Check focus, dialog dismissal and return, reduced motion, and text readability.
- [ ] Test the poster and methods with JavaScript disabled and on a browser without a WebGPU adapter. Interrupt a data load, then exercise its recovery path.
- [ ] Record browser, OS, and device for desktop and physical-phone checks. Document any performance claim with a measurement on the device concerned.

## Verify a hosted release

- [ ] Record the host, release revision, archive checksum, and manifest. Confirm the canonical and social URLs point at the selected HTTPS origin. [Deployment guide](deploy-cloudflare-pages.md)
- [ ] Open the root viewer, its compatibility alias, the static methods, and a missing data URL. Confirm the last request returns HTTP 404. Check HTML revalidation and cache headers for data, scripts, and the poster.
- [ ] Fetch a density face, a composition-support face, an elevation face, and an urban face. Verify Brotli encoding on the response and decoded bytes against the release manifest.
- [ ] Exercise a first load and an update from the previous artifact. Repeat the browser and device checks against the hosted version. Inspect an external link preview and retain the previous artifact for rollback.
