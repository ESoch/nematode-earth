# Contributing to Nematode Earth

You can work on the viewer with Node.js 22.12+, pnpm 10.30.3, and the baked textures in the checkout. [Run the viewer locally](README.md#run-locally)

Open an issue to report a bug or suggest a scientific correction. For a rendering bug, include the route, steps to reproduce, browser, OS, and what you saw. A screenshot and GPU model help when the symptom depends on the device. For a scientific correction, identify the claim or layer and link to the relevant page, figure, or dataset record. [GitHub issues](https://github.com/ESoch/nematode-earth/issues)

## Where to work

| Area | Entry point |
| --- | --- |
| Exhibit | [viewer.ts](app/src/viewer.ts) connects the scene, controls, and data loading. [viewer.html](app/viewer.html) includes the static fallback. |
| Rendering | [field](app/src/field/) reads and samples the data. [regimes](app/src/regimes/) contains the public volumetric shell and the experimental particle renderer. [sweep](app/src/sweep/) controls the opening sequence. |
| Interface | [hud](app/src/hud/) holds the viewer’s labels, guide, and styles. [about.html](app/about.html) contains the static science, methods, and credits. |
| Data | [fetch](pipeline/fetch/) reads upstream data. [build](pipeline/build/) transforms it into cube faces and sample records. [public/data](app/public/data/) holds the browser-ready binaries and their metadata. |
| Research | [marine](pipeline/marine/) contains the offline marine experiment. Its normalized records include slices and subsamples. The experiment fits separate models for two regions. [Methods and held-out results](docs/superpowers/results/2026-09-19-marine-experiment.md) |
| Tests and packaging | [app/test](app/test/) exercises the viewer. [app/scripts](app/scripts/) builds and verifies release artifacts. Python tests live alongside their pipeline modules. |

## Run and check a change

For viewer work, start the development server using the README instructions. Both `/` and `/viewer.html` open the exhibit. The methods are at `/about.html`. The GPU experiment harness at `/experiments.html` and the soil validation page at `/soil-validation.html` are development tools. The production build includes the exhibit and methods.

From `app/`, run the relevant test file while editing, then run the full viewer suite. For example,

```sh
pnpm exec vitest run test/loadCube.test.ts
pnpm test
```

For pipeline work, install Python 3.11+ and uv. From the repository root, run

```sh
cd pipeline
uv sync --locked
uv run --offline --no-sync pytest
```

The suite uses generated fixtures and real soil-sample records. On the first run, the sample tests fetch about 2.7 MB of CSVs from Figshare and verify their checksums. Later runs use the files in `pipeline/cache/nematode_samples/`. The `--offline` option controls package resolution in uv. The marine integration test uses a separately cached publisher snapshot and skips when it is absent. Regenerating real research products requires separate source inputs. The soil extraction guide describes local-file and bounded remote reads. [Soil pipeline guide](pipeline/fetch/CONSERVATIVE.md) · [Marine inputs and reproduction commands](docs/superpowers/results/2026-09-19-marine-experiment.md)

For a production check, run these commands from `app/` after installing the locked dependencies

```sh
pnpm build
pnpm check:release
pnpm preview
```

The build recreates `app/dist/`, checks TypeScript, runs Vite, and compresses and verifies the data. `check:release` runs the viewer tests, Node release-script tests, type check, and verification of that build. `pnpm test:release` runs the release-script tests alone. For visual changes, open the printed preview URL (normally [127.0.0.1:4173](http://127.0.0.1:4173/)). The preview serves the Brotli encoding headers the browser needs to decode the binary textures. The deployment guide explains `SITE_URL` for builds targeting another HTTPS origin. [Release checklist](docs/release-checklist.md) · [Deployment guide](docs/deploy-cloudflare-pages.md)

## Changing data or scientific claims

Follow a value from the source record through the pipeline, the atlas metadata, and the label a visitor reads. When changing a dataset or an interpretation, describe its units, spatial coverage, handling of missing observations, transformations, and validation. Cite the publication for the finding and the specific dataset release for the values and rights. Update the methods and notices when those details change. [Science & methods source](app/about.html) · [Third-party notices](THIRD_PARTY_NOTICES.md)

The soil density product covers latitudes from 88° S to 88° N and uses six 512 × 512 faces. Density is already an area mean. Its coverage mask identifies cells with source support. Feeding-group masks track their coarser inputs separately. Summing the current GPU upload at atlas cell centres gives a total 0.009% above the input integral. The September extraction record documents the earlier encoding. Ecological accuracy requires other evidence. [Rendering checks](docs/rendering-checks.md) · [Atlas metadata](app/public/data/soil_plain_512_v2/cube_meta.json) · [Extraction and acceptance record](docs/superpowers/results/2026-09-22-local-soil-extraction.md)

Use a new versioned data directory when replacing published atlas bytes. Change the loader path and sidecars together because the existing URLs are cached as immutable. Keep downloaded rasters, local caches, and intermediate products out of proposed changes.

## Submitting a change

Keep a pull request focused on one change. Explain what a visitor or researcher will observe and give the test commands and results. Include a preview or screenshot for visible changes and the browser/device used. For a data correction, include the primary source, applicable reuse terms, and enough detail to reproduce the corrected value.

Original software and its associated software documentation are covered by the MIT License. Data, renders, excerpts, and dependencies retain their own terms. [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
