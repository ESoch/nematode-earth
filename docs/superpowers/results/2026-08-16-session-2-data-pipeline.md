# First extraction of the ETH soil abundance raster

On August 16, 2026, the pipeline read approximately 2.42% of the native rows and all columns to make a 512 × 256 grid. Its area integral was 4.617 × 10²⁰, about 4.9% above the published central estimate. On September 22, an extraction of every row yielded 4.615 × 10²⁰. The discrepancy persisted. Its cause remains unresolved. The viewer now uses the later extraction for density and bootstrap uncertainty. [Full-row results](2026-09-22-local-soil-extraction.md#numerical-checks) · [2019 study](https://doi.org/10.1038/s41586-019-1418-6)

## Input layout

The August run inspected the headers of the density raster and a companion per-100-g raster. Each density row occupied one uncompressed TIFF strip, which allowed the reader to fetch rows by byte offset. [ETH abundance dataset](https://doi.org/10.3929/ethz-b-000354035) · [TIFF reader](../../../pipeline/fetch/tiff_header.py)

| Property | Observed value |
| --- | --- |
| Density file | `TotalNumber_perSquareMeter.tif`, BigTIFF, 43,201 × 21,121 float64 values |
| Density row stride | 345,608 bytes (43,201 × 8) |
| Companion per-100-g file | Classic TIFF, float32, 172,804 bytes per row |
| Coverage | Northern tiepoint 88°. The output spans 88° N to 88° S. |
| Server responses on August 16 | The TIFF endpoint accepted byte ranges (206) and multipart ranges. A HEAD request omitted the content length. |

Access varied between HTTP clients during the August run. The successful single-connection transfers ran at 70–120 KB/s, with 21 retries across 277 requests and no failed output rows. These timings describe the historical extraction only. For reproducible processing from downloaded files or bounded remote reads, use the [current extraction guide](../../../pipeline/fetch/CONSERVATIVE.md).

## Comparison with the published estimate

| August comparison | Individuals |
| --- | ---: |
| Integral of the coarsened density grid | 4.617 × 10²⁰ |
| Paper’s global estimate | (4.4 ± 0.64) × 10²⁰ |
| Ratio to the paper’s central estimate | 1.049 |

The initial check tested whether the two values were within 5%. That comparison uses the study estimate as a broad reference. The subsequent full-row work separately measures errors introduced by reduction and texture encoding against the actual input grid. The original low-resolution cube and its checks remain available for comparison. [August atlas metadata](../../../app/public/data/cube_plain_128/cube_meta.json) · [Full-row conversion checks](2026-09-22-local-soil-extraction.md#numerical-checks)

## Fixture tests and follow-up

From `pipeline/`, `uv run --no-sync pytest fetch/` exercised the synthetic BigTIFF reader. The nine tests in the August run covered header parsing, bounded range reads, contiguous box averaging, invalid data and weighted integrals. The maximum box-average difference from NumPy was 1.15 × 10⁻⁷. Doubling source values doubled the integral (ratio 2.000000). A striped fixture detected accidental point sampling, and a latitude test checked the spherical area excluding the two caps beyond ±88°.

The August sample inferred geographic coordinates from tiepoint and scale. The September extractor decoded the GeoKeys and georeferenced density and bootstrap CV separately. Their grids differ by one row and column. Feeding groups still use the coarser extraction with a separate support mask. The original August run did not inspect all sixteen rasters or establish the geometry of every extrapolation mask. [September extraction evidence](2026-09-22-local-soil-extraction.md#download-and-extraction-evidence)

## Licensing

The ETH records carry the In Copyright — Non-Commercial Use Permitted rights statement. It applies to the derived soil fields and their imagery. [Dataset rights and credits](../../../THIRD_PARTY_NOTICES.md) · [Rights statement](https://rightsstatements.org/page/InC-NC/1.0/)
