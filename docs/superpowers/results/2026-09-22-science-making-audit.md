# Scientific interpretation and zoom limit

On September 22, 2026, we compared the viewer's claims with the original studies, the data files and the rendering code. This record explains the resulting corrections and the screen-space zoom checks. [Current methods](../../../app/about.html) · [Soil atlas checks](2026-09-22-local-soil-extraction.md)

## Consequential corrections

| Question | What the evidence supports |
| --- | --- |
| How much of the view would nematodes cover? | For one column with randomly placed opaque silhouettes and optical depth 0.04, expected coverage is `1 − exp(−0.04) ≈ 3.9%`. The assumed projected area of 1.49 × 10⁻⁸ m² per animal lacks an empirical calibration or confidence interval. We could not reproduce the earlier 4.06% global mean with a specified spatial weighting. The 27-fold biome contrast came from a synthetic illustration. Computing a mean of the coverage function requires applying it to each column before averaging. [Optical model](../../../app/src/math/optical.ts) |
| Do cities support more nematodes? | The twenty-city fit fixed its intercept at one and estimated `1 + a·u + b·u²`. Change from zero to full built-up cover depends on `a+b`. The nominal interval for `a` resampled individual pixels and omitted shared baselines, spatial dependence and uncertainty in the underlying predictions. The fitted curve performed slightly worse than predicting the pooled mean. The soil view keeps the published field. The fourfold Cobb illustration is a design choice. [Analysis and fit](2026-08-16-urban-claim-ruling.md) |
| What did urban field studies find? | Li and colleagues reported a higher point estimate for the urban total but no significant difference across its land-use comparisons. Gong and colleagues reported lower *proportions* of omnivores and predators in urban soils. Those proportions alone cannot establish a loss of individuals. The original global soil model used all 73 covariates, including human-related variables. [Li et al. (2022)](https://doi.org/10.1111/1365-2656.13744) · [Gong et al. (2024)](https://doi.org/10.1016/j.soilbio.2023.109297) · [2019 study and methods](https://nemaplex.ucdavis.edu/FerrisPublications/pdf%20files/213%20Hoogen_et_al-2019-Nature.pdf) |
| What does a sample marker mean? | Each marker represents a record in the 2020 release of 6,825 samples. Records may share a location. The 2019 study compiled 6,759 samples and fitted its model to 1,876 non-Antarctic map pixels. The updated database aggregates to 1,933 pixels. [2019 study](https://doi.org/10.1038/s41586-019-1418-6) · [2020 database paper](https://doi.org/10.1038/s41597-020-0437-3) |
| Where does the map show land? | A previous mask misclassified about 0.43% of the *source cells*. These cells cover different areas by latitude. The revised mask still uses an approximate density ratio (0.917), and surface height minus bed elevation includes the water cavity beneath a floating shelf. Named probes checked particular sites. A quantitative estimate of grounded-ice area would require an independent grounding-line dataset. Ice-free inland basins below sea level can still read as water. [ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model) · [Ice/seawater parameters](https://www.pism.io/docs/manual/parameters/index.html) |
| What does an empty cell mean? | GHSL values outside the product’s polar extent are filled with zero for display. The soil model has its own land and latitude limits. A gap in either layer describes available source coverage. [Built-up surface grid](https://doi.org/10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA) · [Atlas coverage](../../../app/public/data/soil_plain_512_v2/cube_meta.json) |

The rendering review also confirmed bounded front-to-back integration per segment, caps and tapers near the rim, and a synthetic particle prototype separate from the public soil field. The compressed-texture limitation applied to the installed KTX2 path. Agreements at cube seams do not imply pointwise identity under a projection warp. [Shell renderer](../../../app/src/regimes/Shell.ts) · [WGSL uniformity](https://www.w3.org/TR/WGSL/#uniformity)

## Approved zoom policy

The public camera stops when the widest central ground cell of the 512-face soil atlas would reach **eight CSS pixels**. Cell width is bounded by `2R/N`, or 24,886.71875 m for N = 512 and the project's globe radius. Camera projection and responsive framing determine the matching altitude. The policy uses CSS pixels and stays fixed throughout the sweep. It describes the view at its centre. It gives no per-fragment error bound for the finite-height, two-wall shell. [Zoom calculation](../../../app/src/core/zoomQuality.ts)

The same limit applies to scripted altitude changes. The terrain clearance floor takes precedence wherever higher. An invalid layout retains the last matching projection and viewport, and a larger permitted range after resize leaves the camera in place. Within 25% above the floor, inward gestures slow with a logarithmic taper. A finite input reaches the floor, and reversing direction moves outward immediately. The boundary hint is announced on entry and exit.

### Observed browser results

One desktop browser rendered the actual soil atlas at two emulated viewport sizes. Checks at playback times of 0, 20 and 40 seconds gave the same limit in both the Earth and veil states.

| CSS viewport | Quality floor | Projected widest central ground texel |
| --- | ---: | ---: |
| 1366 × 768 | 1,594,421.0001 m | 8 CSS px |
| 390 × 844 | 1,104,086.5084 m | 8 CSS px |

At the laptop size, repeated inward input moved the camera from 1,993,026 to 1,734,259 to 1,612,959 to 1,594,421 m. More inward input left it there. Reversing moved it to 1,650,341 m and cleared the hint. At the phone viewport, a 20-pixel outward wheel event moved the camera from the floor to 1,142,809.1321 m. The methods page fit widths of 390 and 1,280 CSS pixels without horizontal overflow. These observations describe browser geometry. Physical phone performance remained untested.

## Verification receipts

September 22 outputs included 942 passing app tests in 51 files, 324 passing Python tests in 8.05 s and a successful TypeScript check. The production HTML/JS/CSS check transformed 64 modules and emitted `index.html`, `viewer.html` and `about.html` in 2.24 s. Brotli and HTTP delivery were checked separately. [September 26 release evidence](../../release-evidence-2026-09-26.md)

```text
app/       pnpm test
app/       pnpm exec tsc --noEmit
pipeline/  uv run --no-sync pytest -q
app/       pnpm exec vite build --outDir <new temporary directory>
root/      git diff --check
```

## Remaining scientific work

Empirical optical coverage needs independent measurements of animal cross-section and orientation. Urban ecological effects need a design with spatially independent observations. Quantitative ice-area claims need an independently validated mask. Regional marine trials have limited geographic coverage and are documented separately. [Marine methods and results](2026-09-19-marine-experiment.md)
