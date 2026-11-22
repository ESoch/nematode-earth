# Conservative soil extraction

Run from `pipeline/`. Legacy `eth_raster.py extract` and `extract_grid` remain
available to reproduce the old central-row-block products. `conservative` is a
separate interface; it never overwrites existing final products.

Legacy row bands are fetched in sample-aligned subrequests within the HTTP
client's 32 MiB limit; splitting does not change the samples or averaging.
Before fetching, every strip offset/count is checked for the supported
contiguous, uncompressed, single-channel float32/float64, one-row-per-strip
layout. Padding, gaps or unsupported layouts are rejected, not decoded as data.
Legacy `extract` now fails before writing final files if any requested output
row failed or the resulting arrays contain non-finite values. The Python
`extract_grid` diagnostic result still exposes `failed_rows` for callers.

## Full extraction

```sh
uv run fetch/eth_raster.py conservative \
  https://www.research-collection.ethz.ch/server/api/core/bitstreams/670c47cd-b642-4932-8f50-769bf71f19c6/content \
  --source-version 670c47cd-b642-4932-8f50-769bf71f19c6 \
  --citation https://doi.org/10.3929/ethz-b-000354035 \
  --licence https://rightsstatements.org/page/InC-NC/1.0/ \
  --out-width 2048 --out-height 1024 --block-rows 64 \
  --cache-dir out/soil-conservative-cache \
  --out-dir out/soil-conservative-v1 --tag soil-full-2048x1024
```

Use 64-row blocks at this output resolution to stay below the default 80 MiB
reduced-checkpoint budget. At 1024x512 the default 32-row blocks suffice. Native
requests are limited to 32 MiB. Neither mode persists native TIFF rows. The
budget covers all namespaces already in the selected cache directory; existing
data is never evicted. Run only one writer per cache/output directory at a time.

The source UUID identifies an immutable archive revision; a new revision must
use its own identity. The header's actual GeoKeys must identify geographic
EPSG:4326 and PixelIsArea or PixelIsPoint. Missing/unsupported keys are errors.
PixelIsArea uses cell boundaries derived from the tiepoint. PixelIsPoint uses
midpoint cell boundaries, half a native pixel around each sample, explicitly
recorded as a reconstruction convention. Both are clipped to the requested
destination rectangle, default longitude [-180,180], latitude [88,-88].

`--sampling-depth` and `--source-date` default to literal `unknown`; replace them
only with verified source information. `--units` defaults to `nematodes/m^2` and
must be explicitly changed for CV or other non-density sources. `--licence`
records a rights statement; it does not grant additional rights.

For bootstrap CV, use a **separate** 80 MiB cache directory. Both complete caches
can exceed 100 MiB in aggregate. Confirm sufficient disk space for final products and temporary
writes as well as caches before running both concurrently.

```sh
uv run fetch/eth_raster.py conservative \
  https://www.research-collection.ethz.ch/server/api/core/bitstreams/137ecc0c-2673-40fd-a9db-a4b4839bde33/content \
  --source-version 137ecc0c-2673-40fd-a9db-a4b4839bde33 \
  --citation https://doi.org/10.3929/ethz-b-000354035 \
  --licence https://rightsstatements.org/page/InC-NC/1.0/ --units dimensionless \
  --out-width 2048 --out-height 1024 --block-rows 64 --cache-budget-mib 80 \
  --cache-dir out/soil-conservative-cv-cache \
  --out-dir out/soil-conservative-cv-v1 --tag cv-full-2048x1024
```

## Already-downloaded TIFFs

The same conservative reducer can read a complete local download. Add
`--local-file /absolute/path/to/TotalNumber_perSquareMeter.tif` to the density
command above, or `--local-file /absolute/path/to/TotalNumber_Bootstrap_CoefVar.tif`
to the CV command. Keep the archive URL, immutable source version, citation,
units and rights statement: local transport does not replace provenance.
Choose fresh versioned cache/output directories and tags for the local run.

Local mode makes no HTTP requests and does not duplicate or modify the TIFF.
It hashes the complete file once, then uses bounded inclusive byte ranges and
the same geographic overlap reduction as HTTP mode. Size, inode and modification
identity are checked around reads and before publication, including a fully
cached resume. A changed or truncated source fails without a completion marker.

The content SHA-256 participates in checkpoint identity; old HTTP checkpoints
cannot be mistaken for verified local-file blocks. A local hash identifies the
downloaded bytes; it is **not** an independently verified publisher checksum.
The shared internal checkpoint key still uses the historical name
`http_identity`, with a local `(size, sha256, null)` identity in this mode.

Metadata explicitly records `provenance.transport = "local-file"`, the source
basename/size/hash, and `local_reads_made`. `requests_made` and `retries_used`
are zero, and `source_http_identity` is null. Legacy `bytes_fetched` and
`header_bytes_fetched` count local range bytes, not network traffic; they exclude
the initial full-file hash pass. HTTP pacing settings are unused in local mode.

Complete extraction is not viewer approval. The separate candidate builder
still checks auxiliary-channel support, conservation and encoding, and no local
extraction command writes public assets.

## Bounded pilot

Add `--pilot-start-row 10496 --pilot-rows 64` and choose a pilot output directory
and tag. This reads 64 native rows, not a full global extraction. Its metadata
always has `publishable=false`, even if a tiny fixture happens to be fully read.
The density numerator and processed coverage only include the selected rows;
do not render a partial product as a full map.

## Product contract

`TAG_grid.npz` contains five float64 arrays (load with `allow_pickle=False`):

| Key | Shape | Meaning |
| --- | --- | --- |
| `density` | H x W | Known soil contribution divided by full destination cell area |
| `valid_model_fraction` | H x W | Finite, non-nodata source overlap area divided by destination area |
| `processed_fraction` | H x W | Fetched native cell footprint divided by destination area |
| `lon_edges` | W+1 | Strictly increasing geographic longitude boundaries, degrees |
| `lat_edges` | H+1 | Strictly decreasing geographic latitude boundaries, degrees |

Zero density plus zero valid fraction is **unknown coverage**, not measured zero
and not an ocean classifier. A half-valid equal-area cell with value 100 over
its valid half has `density=50`, `valid_model_fraction=0.5`. Do not multiply this
density by fractional validity again. Zero-support sentinel masking is separate.
For non-areal CV, this generic reducer yields a support-diluted numerator;
divide by its **own** valid fraction downstream and retain that support during
interpolation. A mean native CV is not uncertainty of an aggregated prediction.

`TAG_meta.json` is written last and is the completion marker. Check its
`grid_sha256` against the named `grid_file`, schema `soil-conservative-v1`,
`complete` and `publishable` before integration. It records exact source grid,
raster type, no-data value, source/HTTP/config identities, explicit output edges,
row selection and coverage, native/output integrals, relative conservation
error, successful payload/header bytes, retries, wall time, and provenance.
`complete` means all native rows were read, not that every geographical cell
has a prediction. Consult `processed_fraction` and `valid_model_fraction`.

Area weights are longitude overlap times the difference of latitude-edge sines,
using a sphere of radius 6,371,000 m. Fractional overlaps are integrated exactly
for the stated piecewise-constant cell-support reconstruction. This is not a
claim of ellipsoidal area exactness or ecological predictive accuracy. Native
and reduced integrals are calculated independently on the same clipped domain.

## Resume and failures

Cache namespaces hash the source revision, HTTP identity (size/ETag/Last-Modified
when available), native layout/georeferencing, destination edges and block size.
Each completed block stores only reduced mass, valid area, processed area and
its native integral, with a separate SHA-256 completion marker. Missing, corrupt
or incomplete blocks are refetched; verified matching blocks are reused.
Checksums protect against corruption, not a malicious rewrite of both files.

HTTP must return 206 with exactly matching numeric Content-Range and bounded
identity-encoded payload. Source identity changes, short/oversized responses and
unresolved retries fail the run. The CLI batches most header reads into a
512 KiB in-memory prefix to reduce request bursts. HTTP 429 remains an archive
rate limit, not grounds to bypass validation. Wait and retry the same command.

The standalone conservative Python API constructs/primes a default or fresh
client with a validated 16-byte range **before** choosing a checkpoint namespace.
A client already verified during header parsing is reused without another
probe; pass a fresh client for a new extraction session. A same-length source
with a changed ETag therefore cannot reuse the old namespace. API metadata
reports `identity_probe_bytes` separately from pixel bytes. Existing correctly
primed CLI cache namespaces are unchanged by this fix.

`TiffInfo.source_identity` also binds the **parsed header** to its HTTP revision.
`TiffHeaderReader(url, client.fetch).parse()` captures that identity from the
verified client. For a bounded prefix/wrapper fetch function, use
`TiffHeaderReader(url, wrapped_fetch, source_identity=client.source_identity)`.
The extraction API rejects unbound header metadata or a header/client identity
mismatch before opening checkpoints or fetching pixels. Reparse through the
current client and retry; never pair stale dimensions/tiepoints/strip offsets
with a freshly verified different revision. This binding field does not change
the checkpoint namespace or product schema. The CLI's existing shared client
and prefix buffer are explicitly bound, without adding a network request.

CLI pacing defaults to `--request-interval 1 --rate-limit-cooldown 60` (seconds).
The interval covers every request start, including headers, successful data
reads and retries. HTTP 429 waits at least the configured cooldown. Numeric
and HTTP-date `Retry-After` on 429/503 are respected even when longer than the
ordinary exponential-backoff cap. Cooldown messages appear in stderr. Pacing
settings are recorded in metadata and do not change cache identity.

For main's low-rate resume after the observed archive 429, append:

```sh
--request-interval 3 --rate-limit-cooldown 120
```

This is a conservative starting policy, not a guarantee the archive will accept
that rate. Reuse the same cache directory, grid, source revision and tag when no
final product exists. Do not restart concurrently or discard retained blocks.

No final files are produced for download/extraction failures. A process crash
during final publication can leave a `.part` or orphan grid; no JSON completion
marker means it is not publishable. Choose a new tag if a final grid exists;
the reduced cache remains reusable. Concurrent writers are not supported.

Python API: `conservative.extract_conservative(url, info, lon_edges, lat_edges,
*, cache_dir, source_version, client=None, block_rows=32, row_start=0,
row_stop=None, cache_budget_bytes=80*1024**2, progress=False)`. It returns the
arrays plus accounting fields. The CLI adds provenance, `publishable`, file
checksum and publication markers. No bake or renderer files are changed here.
