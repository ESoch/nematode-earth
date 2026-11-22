# Can water depth predict seafloor nematode abundance?

On September 19, 2026, we tested a depth-only model in two regions with complete observations of the top 5 cm of sediment. It performed worse than a constant in the Clarion–Clipperton Zone (CCZ) and better at three HAUSGARTEN stations. The small, region-specific samples and different abundance measures limit geographic transfer. The viewer uses terrestrial soil data.

## Reproduce

From the repository root, run the commands below with Python 3.11+ and uv. The first fetch needs access to the publishers. A cached fetch verifies sizes and SHA-256 hashes. The offline experiment verifies the downloaded data and their original landing pages before generating the reports.

```sh
cd pipeline
uv sync --locked
uv run --no-sync python -m marine fetch
uv run --no-sync python -m marine experiment
uv run --no-sync python -m pytest marine -q -p no:cacheprovider
shasum -a 256 out/marine/raw/873272.tsv out/marine/raw/973147.tsv out/marine/raw/776716.tsv out/marine/raw/wei2010.kml
```

The September 19 run passed 37 marine tests. The integration test uses archived publisher files and skips when they are absent. [Marine pipeline and tests](../../../pipeline/marine/)

## Files

The experiment writes six outputs under `pipeline/out/marine/`:

| Output | Contents |
| --- | --- |
| `raw/` | Four dataset downloads and their four source pages |
| `sources.v1.json` | Publisher citations, terms, URLs, retrieval times, sizes and file digests |
| `observations.v1.json` | 934 records from sediment slices, subsamples and compiled observations. Of these, 677 carry densities and 257 have unknown population density. |
| `cores-0-5cm.v1.json` | 161 CCZ and HAUSGARTEN core records, including 96 eligible targets and reasons for exclusion |
| `audit.v1.json` | Conversions, counts, source methods and the unresolved HAUSGARTEN area discrepancy |
| `regional-model.v1.json` | Training coefficients, fold membership, held-out predictions, error metrics, water-depth ranges and input/code hashes |

The regional report identifies the target, study, source and quality flags for each prediction. Its SHA-256 for the September code and runtime was `daaf7b056d6df1b59523a8a5fccae6a588bd4fd06a3754f072477a7aabbb104b`. Check your local report with `shasum -a 256 out/marine/regional-model.v1.json` from `pipeline/`. The publisher-file digests below identify the inputs independently of the report version.

## Target definitions in model output

Each region uses its own response variable, recorded alongside `observed_density_per_m2`. Within a core the pipeline sums complete, nonoverlapping sediment intervals. Across cores it keeps replicates together in their corresponding area or station.

| Source | `population_accounting_status` | Target |
|---|---|---|
| CCZ | `publisher_reported_total_nematoda_abundance` | Publisher-reported total Nematoda abundance per m² in 0–5 cm sediment |
| HAUSGARTEN | `reported_taxon_sum_whole_population_accounting_unverified` | Sum of publisher-reported taxon abundances per m² in 0–5 cm sediment; not a verified whole-population census |

## Downloaded evidence and audit

| Source | Licence | Download records → normalized records | Decision |
|---|---|---|---|
| [Hauquier, Pape & Vanreusel (2017), PANGAEA 873272](https://doi.pangaea.de/10.1594/PANGAEA.873272) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 44 rows → 44 abundance intervals → 32 complete cores | 31 undisturbed cores, 18 deployments, four geographic areas in experiment |
| [Schnier et al. (2025), PANGAEA 973147](https://doi.pangaea.de/10.1594/PANGAEA.973147) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 9,911 taxon rows → 601 core-slice/subsamples → 129 cores | 403 published abundance intervals; 198 count-only intervals. 65 complete abundance cores eligible |
| [Ingels, Tchesunov & Vanreusel (2011), PANGAEA 776716](https://doi.pangaea.de/10.1594/PANGAEA.776716) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 59 slices across 12 cores/four named stations | Identification counts only; all excluded from population-density training |
| [Wei et al. (2010), File S1](https://doi.org/10.1371/journal.pone.0015323.s001) | CC Attribution, version unspecified in the [article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0015323); [publisher licence policy](https://journals.plos.org/plosone/s/licenses-and-copyright) | 230 `CATEGORY=nematode` KML placemarks | All 230 reported per-m² values retained; all excluded from 0–5 cm training |

**Source-file SHA-256**

```text
873272   8fc0436791196368616b70e57b0a3d2e1620ed578ad06115e898b5ee833fcec1
973147   598c225f68647b995af2049543bb5f1c15003bb3b331d68cb8f777f4e392642f
776716   427709b30b5cf2b6841adff2dc5135d5bea247216675985b83b112a817b76c66
wei2010  99ecefdc031534f6e45dae825e6d3154cf6e529c0dbbaa14348f924051f74adc
```

**CCZ.** The archive reports Nematoda per 10 cm². Multiplying by 1,000 converts to individuals per m². For core `SO239_84_11`, the five intervals give `(267.300 + 47.408 + 50.146 + 19.453 + 25.217) × 1,000 = 409,524/m²`. Full 0–5 cm bulk samples need no interval aggregation. The researchers used 94 mm cores and a 32 µm sieve. Only the higher-taxon abundance count enters this analysis. The later genus-identification subsample serves a different purpose. The disturbed IOM core is retained in the observations and excluded from the undisturbed comparison. [Cruise sampling methods](https://doi.org/10.5194/bg-16-3475-2019)

**HAUSGARTEN.** For 2000–2009, `Abund` gives each reported taxon’s abundance per 10 cm². We sum those fields for each event, core, subsample and sediment interval, then multiply once by 1,000. Missing taxon rows and sediment intervals remain unknown. Of 87 historical abundance cores, 65 had unambiguous 0–5 cm coverage. Twenty had incomplete intervals and two had overlapping 1–2 cm intervals (`PS62/161-2:161-2_B`, `PS62/183-2:183-2_C`). Station aliases were reconciled with the original identifiers retained. The 65 eligible cores represent 25 deployments at three stations.

Later records report specimen counts. The archive’s stated area of 0.62832 m² times its subsample factor of 0.00604947 gives 38.0100 cm². The stated 2.2 cm syringe diameter gives 3.80133 cm², and eight 10 cm cores would cover 0.0628319 m². The 2010 methods describe handpicking all nematodes, but the area discrepancy and population conversion for later years remain unresolved. We excluded all 42 count-only cores. The completeness of the historical taxon sums also remains uncertain. [HAUSGARTEN archive](https://doi.org/10.1594/PANGAEA.973147) · [2010 sampling methods](https://doi.org/10.3389/fmars.2023.1271447)

**Canyon samples.** The archive lists counts per 25.51759 cm² core, but the paper describes selecting 100–150 individuals per slice for identification. Observed taxon sums range from 25 to 150. Without a population total or fraction selected, converting these to density would give a misleading result. The output retains identified-specimen counts with population density set to `null`. [Sampling and identification methods](https://doi.org/10.1371/journal.pone.0020094)

**Wei compilation.** We selected the 230 KML records labelled `CATEGORY=nematode` and used `ABUNDANCE.per.m2` directly. The compilation has 158 distinct station-name strings. Some records may share sites, and protocols vary. Without sediment-depth intervals these records cannot be used for the 0–5 cm target. Original identifiers, sampling details and reference fields are retained when available. [Supplementary File S1](https://doi.org/10.1371/journal.pone.0015323.s001)

## Actual held-out results

For each region, least squares fitted `ln(1 + density_per_m²) = intercept + slope × water_depth_km`. The baseline predicted the mean transformed response in the training fold. We held out each entire area or station, including all its years and replicates. Each complete core contributed one target. Both fits used only the remaining groups, with no hyperparameter selection.

| Experiment | Cores / groups | Depth RMSE / baseline RMSE, natural-log1p | Depth MAE / baseline MAE | Geometric absolute fold error, depth / baseline |
|---|---:|---:|---:|---:|
| CCZ area holdout | 31 / 4 | **1.646149 / 1.220665** | 1.345072 / 0.945975 | **3.838× / 2.575×** |
| HAUSGARTEN station holdout | 65 / 3 | **0.394341 / 1.031128** | 0.313828 / 0.886444 | **1.369× / 2.426×** |

RMSE and MAE weight each core equally. Giving each area or station equal weight instead yields depth/baseline RMSE of **1.629575 / 1.317366** for CCZ and **0.432342 / 1.214194** for HAUSGARTEN. The final column is `exp(mean(abs(log1p(prediction) − log1p(observation))))`. It expresses the typical multiplicative error in density plus one. Neither region had a zero-valued eligible target.

| Held-out group | Test cores | Depth RMSE | Baseline RMSE | Outside training depth range |
|---|---:|---:|---:|---:|
| CCZ APEI#3 | 8 | 1.226968 | 1.258023 | 0 |
| CCZ GSR | 9 | 1.658162 | 1.531753 | 0 |
| CCZ IFREMER | 11 | 1.874618 | 0.649477 | 11 |
| CCZ IOM | 3 | 1.689057 | 1.609691 | 3 |
| HAUSGARTEN HG-I | 29 | 0.281025 | 1.108350 | 29 |
| HAUSGARTEN HG-IV | 27 | 0.439790 | 0.512921 | 0 |
| HAUSGARTEN HG-VII | 9 | 0.536998 | 1.712095 | 9 |

The CCZ records span 4,431–5,012 m water depth, 11.0753–18.7910° N and 130.1387–119.6580° W (April 2–23, 2015). Densities range from 1,585 to 775,097/m². HAUSGARTEN spans 1,246–4,065 m, 79.0586–79.1410° N and 3.475–6.103° E (2000–2009), with reported taxon sums of 168,805–3,870,134/m². These envelopes bound the samples. They do not indicate continuous coverage.

## Concerns and deployment decision

Fourteen of 31 CCZ and 38 of 65 HAUSGARTEN predictions fall outside the training fold’s water-depth range. Across four CCZ areas, depth can vary alongside food supply and geography. At HAUSGARTEN, years overlap across station folds and incomplete profiles disproportionately remove deeper-station cores. Neither evaluation tests future years or transfer to other ocean regions.

Sediment properties, oxygen, food flux and extraction efficiency are outside the model. All densities refer to the sampled sediment interval and size class. Both regional reports therefore retain `public_layer_allowed=false`. The experiment produces regional checks only.
