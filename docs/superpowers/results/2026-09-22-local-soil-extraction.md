# Full-row soil extraction and viewer atlas

On September 22, 2026, we converted the complete ETH abundance and bootstrap uncertainty rasters into the six 512 × 512 faces used by the viewer. The September viewer’s texture encoding changed the area integral by **0.056%**. Feeding-group composition still comes from a coarser extraction and has its own coverage mask. The September 26 release included this atlas. [Numerical checks](#numerical-checks) · [Shipped atlas metadata](../../../app/public/data/soil_plain_512_v2/cube_meta.json) · [Hosted release record](../../release-evidence-2026-09-26.md)

## Download and extraction evidence

The reducer read each local GeoTIFF in full. Both headers specify geographic EPSG:4326, PixelIsArea, a 1/120° pixel scale and a tiepoint at (−180°, 88°). The source grids have different dimensions, so each was georeferenced independently. Both output grids have 2,048 × 1,024 cells spanning −180° to 180° longitude and 88° N to 88° S. [ETH abundance and bootstrap rasters](https://doi.org/10.3929/ethz-b-000354035) · [Conservative extraction guide](../../../pipeline/fetch/CONSERVATIVE.md)

| Measurement | Density | Bootstrap CV |
| --- | --- | --- |
| Filename | `TotalNumber_perSquareMeter.tif` | `TotalNumber_Bootstrap_CoefVar.tif` |
| Actual bytes | 7,299,925,008 | 7,299,410,424 |
| Actual native grid | 43,201 × 21,121 | 43,200 × 21,120 |
| Native rows processed | 21,121 / 21,121 | 21,120 / 21,120 |
| Extraction time, including local hash | 25.114 s | 24.601 s |
| HTTP requests | 0 | 0 |
| Native area integral | 4.614562821134864e20 | 2.5152855856497637e13 |
| Reduced area integral | 4.614562821134919e20 | 2.5152855856497652e13 |
| Relative conservation error | 1.1796339677382364e-14 | 5.374763914272761e-16 |

The abundance integral uses spherical cell areas on this domain. The coefficient of variation (CV) is dimensionless. Its integral checks numerical conservation during resampling and has no interpretation as a population count or uncertainty interval.

**SHA-256 of the complete downloaded files**

- Density: `bd2d93df261aad715a2108591cf86184e7db4a7f618796ae8894e546cab8d389`
- Bootstrap CV: `af956b73b56bf858b5715534d62515c1878c56dfad94b3441b92a0fd45664887`

These fingerprints identify the inputs used here. The archive sizes matched, but there was no independent publisher checksum for comparison.

## Products and independent readback

The extraction produced `density-2048-local_grid.npz` and `bootstrap-cv-2048-local_grid.npz`, each with a `_meta.json` file under `pipeline/out/soil-local-20260922/`. For each input, `complete=true`, `publishable=true` and `row_coverage_actual=1` describe extraction completeness. The cube and GPU checks below determine suitability for the viewer.

Readback verified file hashes, finite values, geographic bounds, units and the original and reduced integrals. Both products had 520,204 supported cells, a valid fraction in [0, 1] and a processed fraction in [0.9999999999999996, 1]. The area with valid source data was 1.3120121712033505 × 10¹⁴ m². The full-row output is four times finer along each axis than the [August 512 × 256 extraction](2026-08-16-session-2-data-pipeline.md). Finer encoding adds no field observations.

## Initial candidate

The strict first build required composition wherever density was available. In 3,631 display cells with a valid density, at least one of the older feeding-group inputs was missing. CV was available in every density-supported cell. The builder rejected the combined product before writing faces.

## Candidate gate

To reproduce that decision, prepare the two full local products using the [local TIFF instructions](../../../pipeline/fetch/CONSERVATIVE.md#already-downloaded-tiffs). Use the filenames above and the five original 512 × 256 feeding-group products in `pipeline/out/`. Run from `pipeline/`.

```sh
uv run --offline --no-sync build/soil_product.py \
  --density-meta out/soil-local-20260922/density-2048-local_meta.json \
  --cv-meta out/soil-local-20260922/bootstrap-cv-2048-local_meta.json \
  --groups-dir out --legacy-extent -180 180 88 -88 \
  --face-size 512 --out-dir out/soil-cube-local-20260922-512
```

The default command exits with status 2 and records the missing coverage:

```json
{
  "status": "rejected",
  "support": {
    "cv_missing_texels": 0,
    "groups_missing_texels": 3631,
    "missing_area_m2": 1345822094884.5874,
    "missing_density_integral": 5.816415045062177e17
  }
}
```

These cells account for about 0.126% of the input density integral. The reported area comes from the candidate texture grid. For the accepted product, repeat the same command with `--groups-policy separate-coverage` and a fresh `--out-dir out/soil-cube-local-20260922-v2-512`. This writes the missing composition as a separate mask while retaining the available density. [Builder source](../../../pipeline/build/soil_product.py) · [ETH feeding-group rasters](https://doi.org/10.3929/ethz-b-000354394)

## Accepted v2 derivative

| Channel | What it stores | How to use it |
| --- | --- | --- |
| R | Log of the all-cell mean density from all source rows | Decode once. Its area mean already includes partial coverage. |
| A | Fraction of the area with a valid density prediction | Treat zero support as unavailable. Multiplying density by A again would dilute coasts twice. |
| B | Mean of native pixel bootstrap CVs over supported area | Interprets variation among source predictions. The CV of an aggregate count requires a separate calculation. |
| G | Coarse omnivore-plus-predator fraction times its own support | Filter G and its separate support mask identically, then divide where support is positive. |

Six R16Float mask faces add 3 MiB to the 12 MiB RGBA texture. All 3,631 composition gaps remain explicit. The viewer checks twelve file digests and uses density and CV. The CPU probe can also read supported feeding-group values. [Published cube and support masks](../../../app/public/data/soil_plain_512_v2/)

### Numerical checks

| Stage | Integral | Relative difference from full-row source |
| --- | --- | --- |
| Geographic source | 4.6145628211349186e20 | — |
| Cube before half encoding | 4.614849717947267e20 | +0.006217% |
| Decoded stored cube | 4.614853157661627e20 | +0.006292% |
| Renderer-equivalent texel encoding | 4.617147289390193e20 | +0.056007% |

All values passed the 1% numerical tolerance. Latitude-band differences were at most 0.0266%. Four regional integrals differed by at most 0.425%, including boundary quadrature. Ten regional, coastal and no-data samples and six composition gaps were compared directly with their source arrays. Half-float encoding preserved all positive density and composition support in these checks.

The source integral remains about 4.9% above the paper’s central estimate of 4.4 × 10²⁰ after processing every row. The comparison has no established explanation. The pipeline preserves its input integral. [Published population estimate](https://doi.org/10.1038/s41586-019-1418-6)

### GPU checks and the instrument correction

The [development probe](../../../app/soil-validation.html) exercised the viewer’s loader, cube orientation, filtered values and missing-data rule. A full and a half-covered test cell each returned an alpha byte of 10. An unsupported cell returned zero. All twelve centre/off-axis orientation probes passed across the six faces. Setting G to 100 left the R, B and A outputs unchanged across the 24 sampled directions.

An ideal float64 reference matched **21 of 24** samples within 0.5% in optical depth. Three thinly supported coastal cells differed by up to 2.1789%. A separate alternating texture showed that this GPU quantized filter weights in 1/256 increments. For example, (0.134366, 0.662338) became (0.1328125, 0.6640625). Half-float rounding affected filtered logarithms and truncated positive optical-depth readback by up to one unit in the last place. A Float32 target checked exponentiation separately.

Using those measured stages, the reference matched **24 of 24** probes. The current viewer decodes optical depth before filtering near coasts. Its measurements are in the [November rendering checks](../../rendering-checks.md). These sampled points do not measure the integral of the full raymarched image, continuity across all cube seams or performance on other devices.

### Viewer and tests

On September 22, the 1280 × 720 browser check loaded `soil-cube-candidate-v2` with `densityFaceSize=512` and exercised the opening, full sweep, uncertainty overlay and regional view. The local-input change passed 321 pipeline tests. After v2 integration, 324 pipeline tests, 902 app tests and TypeScript checking passed. The twelve atlas binaries round-tripped through Brotli with matching hashes, from 15,728,640 raw bytes to 2,578,696 compressed bytes (about 2.46 MiB), excluding metadata.

### Final production build

The September 22 `pnpm build` compressed 85 binary files from 553,147,029 to 58,098,583 bytes (89.5% smaller). All 85 decompressed to their original bytes. The metadata’s `accepted-local-viewer-not-deployed` field dates this local acceptance. Hosted binary and browser checks followed on September 26. [Release evidence](../../release-evidence-2026-09-26.md)
