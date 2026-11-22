# Third-party notices and data provenance

Ethan Soch created the viewer and data pipeline, with coding assistance from Claude Code and Codex. The MIT License covers the project’s original software and its associated software documentation. The research data, quotations, and rendered images have separate terms. [Project license](LICENSE) · [Software citation](CITATION.cff)

| Material | Where it appears | Terms |
| --- | --- | --- |
| ETH soil predictions | `app/public/data/soil_plain_512_v2/`, the historical `cube_plain_128/` atlas, and the two `poster_veil_11500km_v1` images | In Copyright – Non-Commercial Use Permitted (InC-NC) |
| Measured soil samples | `app/public/data/nematode_samples_v2/` | CC0 1.0 |
| NOAA elevation and derived land mask | `app/public/data/elevation_plain_*/` | CC0 1.0 |
| JRC built-up surface | `app/public/data/urban_plain_*/` | European Commission reuse notice. Acknowledge the source and identify the changes. |
| Browser software and adapted shader code | Production JavaScript and the CPU reference implementation | three.js and the Vite helper use MIT. The ACES fit and MaterialX noise routines carry additional notices below. |

## Research and exhibit data

### Historical quotation

Nathan A. Cobb, “Nematodes and their relationships,” *Yearbook of the United States Department of Agriculture 1914*. Government Printing Office, 1915, pp. 457–490. The exhibit quotes p. 472. The contributing library identifies the volume as believed to be out of copyright. [Page scan](https://archive.org/details/yoa1914/page/472/mode/2up) · [Archive record](https://archive.org/metadata/yoa1914)

### Soil abundance, uncertainty, and feeding groups

Johan van den Hoogen and coauthors published *Soil nematode abundance and functional group composition at a global scale* in *Nature* 572, 194–198 (2019). Johan van den Hoogen and Devin Routh deposited the abundance and uncertainty rasters and the additional functional-group rasters in the ETH Zurich Research Collection. [Paper](https://doi.org/10.1038/s41586-019-1418-6) · [Abundance and uncertainty](https://doi.org/10.3929/ethz-b-000354035) · [Functional groups](https://doi.org/10.3929/ethz-b-000354394)

Both deposits carry **In Copyright – Non-Commercial Use Permitted**. This statement permits non-commercial use. For other uses outside applicable copyright exceptions, seek permission from the rights holders. The soil atlases and project renders, including both posters, derive from these data. [Rights statement](https://rightsstatements.org/page/InC-NC/1.0/) · [Registered abundance record](https://api.datacite.org/dois/10.3929/ethz-b-000354035) · [Registered functional-group record](https://api.datacite.org/dois/10.3929/ethz-b-000354394)

The project resampled every row of the abundance and bootstrap uncertainty grids into six display faces. Feeding-group composition comes from a coarser extraction and has a separate coverage mask. The visual exposure, vertical profile, optical-area estimate, twenty-city comparison, and fourfold Cobb illustration are project choices or calculations. [Atlas metadata](app/public/data/soil_plain_512_v2/cube_meta.json) · [Conversion checks](docs/superpowers/results/2026-09-22-local-soil-extraction.md)

### Measured samples

Stefan Geisen, Johan van den Hoogen, Data Providers, and Thomas Crowther deposited *Nematode_abundance_dataset*, version 1 (2020). The Figshare item containing `nematode_full_dataset_wBiome.csv` is CC0 1.0. Van den Hoogen and coauthors describe it in *A global database of soil nematode abundance and functional group composition*, *Scientific Data* 7, 103 (2020). [Dataset and rights](https://doi.org/10.6084/m9.figshare.11733897.v1) · [Item metadata](https://api.figshare.com/v2/articles/11733897) · [Data paper](https://doi.org/10.1038/s41597-020-0437-3)

The overlay retains latitude, longitude, abundance per 100 g of dry soil, and a normalized biome code for 6,825 records. Several records can share a location. [Packed sample format](app/public/data/nematode_samples_v2/samples_meta.json)

### Terrain and grounded ice

NOAA National Centers for Environmental Information (2022), *ETOPO 2022 15 Arc-Second Global Relief Model*. The exhibit uses the 60 arc-second bedrock and ice-surface grids from the same release. NOAA waives potential worldwide rights under CC0 1.0. We resampled both grids and used their heights to estimate land and grounded ice. [Dataset DOI](https://doi.org/10.25921/fd45-gt74) · [NOAA product page](https://www.ncei.noaa.gov/products/etopo-global-relief-model) · [NOAA use constraints](https://data.noaa.gov/metaview/page?xml=NOAA/NESDIS/NGDC/MGG/DEM//iso/xml/etopo_2022.xml&view=getDataView&header=none) · [Terrain metadata](app/public/data/elevation_plain_2048/cube_meta.json)

### Built-up surface

Martino Pesaresi and Panagiotis Politis, European Commission Joint Research Centre, *GHS-BUILT-S R2023A – GHS built-up surface grid, derived from Sentinel2 composite and Landsat, multitemporal (1975–2030)*. We used epoch 2020, EPSG:4326, 30 arc-seconds, V1-0. The catalogue specifies the European Commission reuse notice and requires source acknowledgement, subject to individual notices and third-party rights. [Original dataset DOI](https://doi.org/10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA) · [JRC catalogue and current citation](https://data.jrc.ec.europa.eu/dataset/9f06f36f-4b11-47ec-abb0-4f8b7b1d72ea)

We converted square metres of built-up surface per cell to fractions of cell area and resampled them into cube faces. Where the source ends near the poles, the viewer uses zero for display. The optional upstream download includes *GHSL Data Package 2023* by Marcello Schiavina and coauthors, © European Union, 2024, licensed CC BY 4.0 subject to individual notices. [Atlas metadata](app/public/data/urban_plain_2048/cube_meta.json) · [Report and citation](https://publications.jrc.ec.europa.eu/repository/handle/JRC133256)

## Marine research inputs

The offline marine experiment uses four deposits. The public globe displays terrestrial soil.

| Dataset | Use in this project | Source and terms |
| --- | --- | --- |
| Freija Hauquier, Ellen Pape and Ann Vanreusel (2017), *Meiofauna abundances from multicorer samples during SONNE cruise SO239* | Convert Nematoda per 10 cm² to per m². Sum complete slices covering 0–5 cm. | [PANGAEA 873272](https://doi.org/10.1594/PANGAEA.873272) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) · [Methods](https://doi.org/10.5194/bg-16-3475-2019) |
| Jannik Schnier, Katarzyna Grzelak, Christiane Hasemann and Thomas Soltwedel (2025), *Data on nematode genus abundance and counts at LTER HAUSGARTEN* | Sum published taxon abundances for complete profiles. Later count-only records lack an unambiguous conversion to population density. | [PANGAEA 973147](https://doi.org/10.1594/PANGAEA.973147) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Study methods](https://doi.org/10.3389/fmars.2023.1271447) |
| Jeroen Ingels, Alexei Tchesunov and Ann Vanreusel (2011), *Nematode abundance in sediments obtained during the Belgica cruise BG06/13, Gollum Channels and Whittard Canyon* | Retain identification subsamples separately. Their counts cannot establish population density. | [PANGAEA 776716](https://doi.org/10.1594/PANGAEA.776716) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) · [Sampling methods](https://doi.org/10.1371/journal.pone.0020094) |
| Chih-Lin Wei and coauthors (2010), *Global Patterns and Predictions of Seafloor Biomass Using Random Forests*, File S1 | Retain 230 reported per-m² values. Unknown sediment intervals exclude them from the fixed-depth comparison. | [Supplement](https://doi.org/10.1371/journal.pone.0015323.s001) · [Article and CC Attribution notice](https://doi.org/10.1371/journal.pone.0015323) (version unspecified) |

The marine report gives the source checksums, unit conversions, inclusion rules, and results for each regional model. Running the pipeline also produces `out/marine/sources.v1.json` with the input URLs, licenses, byte counts, and hashes. [Reproduction guide and results](docs/superpowers/results/2026-09-19-marine-experiment.md)

## Software

The viewer imports three.js. The production build also embeds Vite's module-preload helper. The static [credits page](app/about.html#credits) carries the software notices with the distribution.

### three.js 0.185.1 — MIT

The CPU reference in `app/src/field/terrain.ts` follows the three.js filmic tone mapper. The particle code uses the same 32-bit PCG permutation as the three.js TSL hash. The PCG algorithm is due to M. E. O'Neill. [three.js](https://threejs.org/) · [TSL hash implementation](https://github.com/mrdoob/three.js/blob/r185/src/nodes/math/Hash.js) · [O'Neill's PCG paper](https://www.pcg-random.org/paper.html)

```text
The MIT License

Copyright © 2010-2026 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### Vite module-preload helper

Vite inserts this JavaScript into the browser bundle when building the HTML entry. The core notice travels with the emitted helper. [Vite core licence](https://github.com/vitejs/vite/blob/main/packages/vite/LICENSE.md)

```text
MIT License

Copyright (c) 2019-present, VoidZero Inc. and Vite contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MaterialX and Open Shading Language noise

The globe’s erosion and terrain shaders use noise functions that three.js 0.185.1 translated from MaterialX GLSL to its shader language. MaterialX adapted those routines from Open Shading Language. The site includes the MaterialX Apache 2.0 licence and the BSD-3-Clause notices for Sony Pictures Imageworks and the Open Shading Language contributors. [MaterialX licence](app/public/licenses/materialx-apache-2.0.txt) · [Open Shading Language notices](app/public/licenses/openshadinglanguage-noise.txt)

The three.js source identifies the upstream GLSL file. Its header credits Sony Pictures Imageworks, and MaterialX’s third-party notice credits the wider Open Shading Language project. [three.js port](https://github.com/mrdoob/three.js/blob/2431a09f46f34c560bc8e44b33be0e567723d5b9/src/nodes/materialx/lib/mx_noise.js) · [MaterialX noise source](https://github.com/AcademySoftwareFoundation/MaterialX/blob/f85fa6b59b77ff3b091158128e79eefb9d48498b/libraries/stdlib/genglsl/lib/mx_noise.glsl) · [MaterialX licence](https://github.com/AcademySoftwareFoundation/MaterialX/blob/f85fa6b59b77ff3b091158128e79eefb9d48498b/LICENSE) · [MaterialX third-party credits](https://github.com/AcademySoftwareFoundation/MaterialX/blob/f85fa6b59b77ff3b091158128e79eefb9d48498b/THIRD-PARTY.md#openshadinglanguage)

### ACES filmic fit

Stephen Hill's fitted colour transform is used in three.js and mirrored in the project's CPU reference. The upstream shader belongs to the example code accompanying the paper cited in its notice. [Shader](https://github.com/selfshadow/ltc_code/blob/31e5e96b54f98f33098f8503003119ba2231a1c6/webgl/shaders/ltc/ltc_blit.fs) · [Source licence](https://github.com/selfshadow/ltc_code/blob/31e5e96b54f98f33098f8503003119ba2231a1c6/LICENSE) · [Paper DOI](https://doi.org/10.1145/2897824.2925895)

```text
Copyright (c) 2017, Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* If you use (or adapt) the source code in your own work, please include a
  reference to the paper:

  Real-Time Polygonal-Light Shading with Linearly Transformed Cosines.
  Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt.
  ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH 2016) 35(4), 2016.
  Project page: https://eheitzresearch.wordpress.com/415-2/

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Development and offline-pipeline dependencies

These are the direct build, test, and pipeline dependencies. The lockfiles record resolved versions and platform variants. [Viewer lockfile](app/pnpm-lock.yaml) · [Pipeline lockfile](pipeline/uv.lock)

| Package | Inspected version | Declared licence |
| --- | --- | --- |
| `@types/three` | 0.185.4 | MIT |
| `@webgpu/types` | 0.1.71 | BSD-3-Clause |
| TypeScript | 5.9.3 | Apache-2.0 |
| Vite | 6.4.6 | MIT |
| Vitest | 4.1.12 | MIT |
| NumPy | 2.5.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| tifffile | 2026.8.16 | BSD-3-Clause |
| imagecodecs | 2026.8.16 | BSD-3-Clause |
| Requests | 2.34.2 | Apache-2.0 |
| pytest | 9.1.1 | MIT |

When redistributing a development environment or binary wheels, retain the notices shipped with those packages, including notices for bundled codecs.

For a distribution of the exhibit, keep the original software license, the browser software notices above, and the source credits applicable to the data and images you include. The static methods page carries the exhibit’s notices with the built site. [Credits page](app/about.html#credits) · [Release checks](docs/release-checklist.md)
