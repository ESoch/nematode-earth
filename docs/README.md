# Documentation

| If you want to… | Start here |
| --- | --- |
| Explore the exhibit and learn what its layers mean | [Visitor guide](../README.md#what-the-map-shows) · [Science & methods](https://nematode.earth/about) |
| Run the viewer, fix a bug, or propose a correction | [Contributor guide](../CONTRIBUTING.md) |
| Credit the project or reuse a figure or dataset | [Software citation](../CITATION.cff) · [Sources, transformations, and reuse terms](../THIRD_PARTY_NOTICES.md) |
| Understand a value in a baked texture | [Data and metadata by layer](../app/public/data/) · [Soil extraction guide](../pipeline/fetch/CONSERVATIVE.md) |
| Build, preview, or host the static site | [Release checklist](release-checklist.md) · [Cloudflare Pages deployment](deploy-cloudflare-pages.md) |

For the current coastline filter, sample markers, ocean lighting and sweep measurements, see the [rendering checks](rendering-checks.md).

## Research and design records

These dated records explain decisions made during development. The README and contributor guide describe the current exhibit and tooling.

- The [soil extraction and acceptance record](superpowers/results/2026-09-22-local-soil-extraction.md) identifies the source rasters by hash and reports the numerical checks for the shipped density and uncertainty atlas.
- The [city comparison](superpowers/results/2026-08-16-urban-claim-ruling.md) describes the exploratory regression behind the Cobb illustration.
- The [marine experiment](superpowers/results/2026-09-19-marine-experiment.md) gives source identifiers, units, exclusions, held-out results and commands for the two regional experiments.
- The [science and zoom audit](superpowers/results/2026-09-22-science-making-audit.md) explains the screen-space detail limit and the evidence behind the methods page.
- The [original design](superpowers/specs/2026-08-15-nematode-earth-design.md) includes early proposals for organism-scale rendering. The public exhibit stops at the resolution of the soil layer.
- The [September 26 release record](release-evidence-2026-09-26.md) identifies the historical hosted artifact by checksum and documents which devices and routes were exercised.
