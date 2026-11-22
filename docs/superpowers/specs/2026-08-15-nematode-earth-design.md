# Nematode Earth — Design

Design proposal recorded August 15, 2026. It includes the original organism-scale descent and city-suppression hypothesis. Subsequent data and rendering work changed both. For the implemented exhibit, see the [viewer guide](../../../README.md#what-the-map-shows), [city comparison](../results/2026-08-16-urban-claim-ruling.md), and [zoom policy](../results/2026-09-22-science-making-audit.md).

An interactive WebGPU visualization of Nathan Cobb's 1914 claim: that if all matter except
nematodes vanished, Earth would remain "dimly recognizable" as a film of nematodes.

The piece renders that film from published data, lets you sweep the matter away and bring it
back, and lets you descend from orbit to a single animal.

Personal interest project. Non-commercial, ad-free, no paywall.

---

## 1. Cobb’s thought experiment

The August design proposed comparing each part of Cobb’s imagined landscape with a modern field. The available evidence differs by habitat. [Cobb’s original passage](https://archive.org/details/yoa1914/page/472/mode/2up) · [2019 soil study](https://doi.org/10.1038/s41586-019-1418-6)

| Part of the imagined landscape | What the exhibit can show |
|---|---|
| Mountains, hills and valleys | The 2019 study estimates a global topsoil population. Rendering an optically visible film requires additional assumptions about the animals and how they are lit. |
| Oceans | The viewer uses a terrestrial soil model. A separate [marine experiment](../results/2026-09-19-marine-experiment.md) examines sediment cores from two regions. |
| Towns | The original design proposed city suppression. The [twenty-city comparison](../results/2026-08-16-urban-claim-ruling.md) gave no basis for adding that factor to the mapped field. |
| Tree outlines | The abundance rasters cannot resolve individual trees. |
| High latitudes | The paper attributed 38% of the modelled total to sub-Arctic regions and 21% to tropical regions. Both poles lie outside the source raster. |

A static image cannot hold a correction. The Cobb-vs-data toggle (§6.3) is the thesis made
operable, and is the reason this is interactive rather than a poster.

*4.40 ± 0.64 × 10²⁰ is van den Hoogen et al. 2019's own published figure (Table 1 and abstract,
±14.6%) — not this project's precision. Quote it with the error bar; a bare point estimate
claims more certainty than the source does.*

### 1.1 Illustrative optical calculation

For one column with areal abundance `N_A` and assumed average projected animal area `E[A]`, the proposed rendering model gives optical depth `τ = E[A] · N_A`. Randomly placed opaque silhouettes then have expected coverage `1 − exp(−τ)`. For example, at `τ = 0.04` that expectation is about 3.9%.

The original design reported 4.06% as a land-wide mean. The subsequent audit could not reproduce it from the shipped atlas. Its 27-fold biome contrast came from a synthetic field. A global coverage estimate would require averaging the predicted coverage over area, along with evidence for the model's physical assumptions. [Rendering calculation](../../../app/src/math/optical.ts) · [Science audit](../results/2026-09-22-science-making-audit.md)

---

## 2. Non-goals

- Not scientifically publishable. Defensible and sourced, not peer-reviewed.
- Not a general-purpose globe. One dataset, one story.
- Not mobile-first. WebGPU required; mobile is a reduced tier.
- Not a live simulation. Nematodes do not interact. Static field, animated presentation.

---

## 3. Scale architecture

Orbit (1.27 × 10⁷ m) to a single organism (~10⁻³ m) — **ten orders of magnitude.**

### 3.1 Why the statistical field stops at ~1 cm

Surface number density is `n(0) = 3.9 × 10⁷ m⁻³ = 39 /cm³`.

| Field of view | Nematodes in view |
|---|---|
| 10 cm | ~39,000 |
| 3 cm | ~1,000 |
| 1 cm | 39 |
| 3 mm | 1 |
| 1 mm | 0.039 |

Below ~1 cm the frame is mostly empty. The piece uses this rather than fighting it: the
statistical field runs to ~1 cm, then the camera finds **one animal** and holds. The
emptiness is the finding — the veil really is that thin.

> **Never quote a bare "per cm³."** `39 /cm³` is the *surface* density; `20–23 /cm³` is the
> *0–15 cm depth mean*. Both circulate and both are right under their own definition.

### 3.2 The conceit at the bottom

With all non-nematode matter gone there is no soil, no pore, no water film. The hero organism
is in vacuum and therefore **swims** (`U/c ≈ 0.08`) rather than crawls (`0.67`). A worm
thrashing in nothing, going nowhere — physically coherent with the premise and the right
emotional note.

Consequence: no refractive background. Shade with rim light and internal scattering, not the
dielectric-tube-over-background model that soil would permit.

---

## 4. The density field

### 4.1 Factorization

The field is **not** a 3D texture. It factorizes:

```
ρ(dir, z) = A(dir) · g(z)
```

`A(dir)` is a 2D cube map of areal density; `g(z)` is an analytic normalized depth profile.
The shader contract `sampleDensity(dir, z)` is preserved — one field remains the source of
truth, it simply is not stored volumetrically.

**Why**, on three legs:

1. `maxTextureDimension3D` = 2048 (hard WebGPU default; the 2D limit is 8192).
2. No toolchain exists. Basis Universal's encoder supports mipmaps, arrays, cubemaps and
   cubemap arrays — **not volumes**. `KTX2Loader.js:1230` throws `'Unsupported pixelDepth.'`
   in the compressed branch. No `Compressed3DTexture` class exists.
3. The depth axis is *modelled, not measured* — storing it spends bytes encoding an invention.

> Do **not** cite `copyExternalImageToTexture`'s 2D-only constraint as justification. A
> GPU-side 2D-staging → `copyTextureToTexture` path into a 3D texture is legal, and that
> argument will not survive review.

What factorization buys beyond unblocking: the vertical integral becomes analytic, no 3D
seam/gutter subsystem, no sun light-march, and most per-step march cost disappears.

### 4.2 The key invariant

```
τ_nadir = E[A] · N_A
```

**Independent of the depth profile.** The van den Hoogen per-m² raster maps *directly* to
nadir optical depth with no depth model in the shader loop. Put this in the shader contract.

### 4.3 The depth profile is modelled and labelled

Every published raster is a single 0–15 cm depth-integrated slab; the vertical axis does not
exist in the data. Modelled as `exp(−z/z₀)` renormalized so the 0–15 cm integral reproduces
the published per-m² value exactly — mass-conserving, inventing no nematodes.

Extension below 15 cm is an explicit labelled toggle, never a default, and never feeds a
quoted global number.

### 4.4 Projection: equi-angular cubemap

The design proposed a cubemap to limit latitude-dependent distortion. The source gives high abundance in sub-Arctic regions and ends at ±88°.
EAC warp on top: it reduces worst-case texel solid-angle ratio from `3√3 = 5.196:1` to
exactly `√2 = 1.414:1` (not ~1:1 — minimum at face-edge midpoints), costing ~30–60 ALU per
sample (not 4 — the "4 ops" figure quotes the encoder-side inverse).

Texel ground size at R = 6371 km: N=1024 → 9.77 km, N=2048 → 4.89 km, N=4096 → 2.44 km.
Quote the level actually shipped.

**Bake-time requirement — geodetic → geocentric latitude conversion.**
`tan(φ_gc) = (1−f)²·tan(φ_gd)`. Maximum difference 11.545 arcmin at φ=45.1° = **21.40 km of
ground displacement = 1.81 px @1080p, 3.62 px @2160p**. Larger than the sphere-vs-spheroid
silhouette error, and it is a *content* error — continents in the wrong place. Free to fix
at bake time. Do this regardless of every other geometry decision.

### 4.5 Channels and encoding

| Channel | Content | Encoding |
|---|---|---|
| R | areal density (per m²) | log-encoded, R16Float |
| G | functional-group composition index | lossy OK |
| B | provenance class | **strictly lossless** |
| A | land fraction / validity | lossless |

Log encoding is required — land values span ~6 × 10⁵ to ~2 × 10⁷ per m², and linear 8-bit
crushes the desert-to-boreal gradient to nothing.

- **Mip in linear σ, never in log.** Box-filtering `log₁₀(ρ)` computes the *geometric* mean
  and systematically darkens every mip level, worst where variance is highest — coastlines,
  biome boundaries, city edges. Decode, filter, re-encode.
- **Use R16Float.** `r32float` is not filterable without the optional `float32-filterable`
  feature (52% availability on iOS). `r16float` is filterable everywhere.
- **One grayscale WebP per channel.** Lossy WebP encodes YUV 4:2:0; the RGB→YCbCr rotation
  mixes all three inputs before chroma subsampling.
- `createImageBitmap(blob, {colorSpaceConversion:'none', premultiplyAlpha:'none'})` and
  `texture.colorSpace = NoColorSpace`, or every density texel silently acquires an sRGB curve.

---

## 5. Data

### 5.1 The prediction rasters

The published global prediction surface exists, but the URL printed in the Nature paper's own
Data Availability statement 302s to a login page. Working route: GitLab README → two ETH
Research Collection DOIs → 16 GeoTIFFs, 87.6 GB.

Byte-verified format: 43201 × 21121, EPSG:4326, 30 arc-sec, uncompressed, **RowsPerStrip=1**,
origin (−180.0, 88.0). Latitude clipped at ±88, not ±90.

**We never download 87 GB.** The server honors byte ranges; with one strip per scanline and no
compression, any row is directly addressable — read `StripOffsets`, seek, read `row_stride`.
The atlas bakes from ~2,000 range GETs.

- **Fetch contiguous row blocks and average**, never every Nth row. Point-decimating a
  high-frequency prediction surface aliases hard and contradicts mass conservation.
- Download only `TotalNumber_perSquareMeter.tif` plus the five `*_per100gSoil.tif`. The
  per-100g → per-m² factor is identical across functional groups at a given pixel, so group
  *fractions* are identical in both unit systems — compute composition from the cheap float32
  files and skip 36.5 GB of float64.
- Three of the sixteen rasters use a different grid (43200 × 21120, different extent). The
  extrapolation mask is among them; resample explicitly.
- `/vsicurl` may fail — the ETH HEAD response carries no `content-length`, breaking GDAL's
  size discovery. Hand-rolled range logic is the fallback.
- The provenance chain is fragile. Archive the bitstreams locally and checksum them.

### 5.2 Other sources

| Purpose | Source | License |
|---|---|---|
| Elevation and bathymetry | NOAA ETOPO 2022 | CC0 1.0 |
| Built-up surface | JRC GHS-BUILT-S R2023A | European Commission reuse notice |
| Marine seafloor (proposed) | Depth and particulate organic carbon flux | No marine layer shipped in the viewer |
| Sample-point overlay | Figshare sample dataset, 6,825 records | CC0 1.0 |

### 5.3 Licensing

Both ETH raster deposits carry the In Copyright — Non-Commercial Use Permitted rights statement. It applies to the derived soil atlases and imagery. The Figshare item containing the measured samples carries CC0 1.0. Each point locates a sample. Computing a sampled fraction of land would require sampling footprints. [Source records and reuse terms](../../../THIRD_PARTY_NOTICES.md)

The 2019 study compiled 6,759 samples and fitted the geospatial model to 1,876 non-Antarctic map pixels. The 2020 Figshare release contains 6,825 samples, including 66 additional records from Ireland. [2019 study](https://doi.org/10.1038/s41586-019-1418-6) · [2020 data paper](https://doi.org/10.1038/s41597-020-0437-3)

### 5.4 Validation gate

The original design proposed comparing the integral with the paper’s published **4.4 × 10²⁰** estimate. The implemented pipeline checks conservation against its own source integral. The full extraction integrates to about 4.615 × 10²⁰ under the project’s area convention. Cube resampling and float16 encoding differ from that value by 0.056%. [Extraction and acceptance record](../results/2026-09-22-local-soil-extraction.md)

Known trap it must handle: `_perSquareMeter` means nematodes per m² *of pixel*, not of soil —
land fraction is baked in, so coastal pixels are systematically diluted. Uncorrected, every
coastline gets a dark fringe that reads as a rendering bug.

---

## 6. Presentation

### 6.1 The sweep

| Beat | t | Content |
|---|---|---|
| Cold open | 0:00–0:03 | Photoreal Earth, slow rotation. Deliberately ordinary. Cobb's line types in |
| Erosion | 0:03–0:09 | Spatially propagating dissolve, not a crossfade. Atmosphere, then water, then rock |
| Collapse | 0:09–0:13 | Surface drops to bathymetry. The true shape of the planet |
| What remains | 0:13–0:18 | The veil alone, with contrast across the regions covered by the soil model |
| Release | 0:18–0:20 | Camera decelerates, controls fade in |

**Governing rule:** the veil renders at full opacity from frame zero, *occluded* by matter
rather than faded in. It was always there. The thesis should be true in the render, not
merely narrated.

### 6.2 Visual language

The design called for a near-monochrome palette from bone-white to cold cyan. Contrast helps reveal geographic variation. At orbital altitude the near and far walls of the rendered shell can both contribute to the image, changing its apparent brightness. The observed field records abundance in soil. Section 1.1 explains the additional assumptions needed to convert that abundance into an illustrative opacity.

No viridis, no turbo. Hue-mapped density reads as a dashboard and double-encodes what optical
depth already carries. Hue is spent where nothing else can go:

- **Provenance** — modelled regions desaturate toward grey. Uncertainty visible in the image.
- **Functional groups** — five restrained tints, resolvable only near the surface.

Post: convolution bloom, slight rim dispersion, low grain to break dark-volumetric banding,
ACES. **No** lens flare, god rays, or vignette.

### 6.3 The proposed Cobb inversion

A slider between Cobb's prediction and the data. Same geometry, same shader, one uniform:

```
m_cobb(u) = 1 + 3u                    towns as bright massings
m_data(u) = 1 − 1.70u + 0.75u²        towns as voids
```

The thesis of the project, made interactive, for the cost of a uniform. A headline beat, not
a settings item.

The suppression curve above was rejected. The implemented soil-model multiplier is 1 at every built-up fraction. The illustration retains `1 + 3u` as a visual interpretation of Cobb. [City comparison and limitations](../results/2026-08-16-urban-claim-ruling.md)

### 6.4 Count vs mass

Bacterivores are 43.6% of individuals; omnivores dominate biomass (10.24 of 31.44 Mt C).
Switching the globe from "number" to "mass" visibly reorganizes the planet. One extra channel,
second interactive beat.

### 6.5 The hero stat

Every nematode on Earth, compacted, is a film **0.745 ± 0.084 µm** thick over the planet's
surface (2.55 ± 0.29 µm over land alone). That is inside the visible-light band — Cobb's film
is thinner than the wavelength of the light you would need to see it.

Use **302.30 ± 33.99 Mt** fresh biomass (van den Hoogen et al. 2019 Table 1, ±11.2% — a
*different* figure from §1's ±14.6% count uncertainty; the two are not interchangeable), not
the rounded "0.3 Gt" — the rounding makes the carbon cross-check appear 0.8% off.

---

## 7. Rendering

### 7.1 Geometry: oblate spheroid

WGS84 `a = 6378.137 km`, `b = 6356.752314 km`.

Against a mean sphere the true surface is **+7.137 km at the equator and −14.248 km at the
poles** — up to 142% of the 10 km regime handoff altitude. A camera at "10 km" would be
underground at the equator or 24 km up at the poles. The spheroid is not a refinement; it is
required for the camera to work.

**Ray/shell intersection runs in sphere space.** Pre-multiply origin and direction by
`S = diag(1/a, 1/a, 1/b)`, normalize the scaled direction, solve the unit-sphere quadratic,
recover `t = t'/|d'|`. Roots come back along the original ray with no back-transform —
confirmed against Cesium's `IntersectionTests.rayEllipsoid`. Cost ≈ 8 ALU/pixel. Concentricity
and the shared-`L²` trick survive intact. Residual shell non-uniformity is 0.335% (215 m on
64 km) — invisible.

Rejected: "render a sphere and displace" turns an exactly-analytic 2-root problem into a
root-find for a 0.335% correction.

### 7.2 Numerical precision

- Use Ray Tracing Gems Ch.7 method 6 **consistently**: share `L²` across the two shell radii
  for the discriminants, but form each `t` through the `c/q` branch, and compute segment
  **length directly as `2s`** — never by differencing two ~3 × 10⁷ m values.
- Form `2RH + H²` on the CPU and upload it. Never as a difference of two f32 squares.
- **Method 6 fixes the discriminant, not the tangent singularity.** `d(chord)/dL` diverges at
  tangency; one f32 ULP of ray direction (~6 × 10⁻⁸ rad) displaces the impact parameter ~1.8 m
  at 30,000 km, measured as 1,660 m of chord error on a grazing sample. The analytic silhouette
  antialias (smoothstep on the discriminant normalized by pixel angular footprint) and a
  step-count floor from the pixel footprint are **requirements, not niceties**.
- f32 ULP at 6371 km is **0.5 m exactly**. f32 is scale-invariant — "work in km" is about
  readability, not accuracy.
- Shell thickness floor `H_s ≥ 200 m` in the descent ramp (f32 conditioning of `R_out² − R_in²`
  gives 19% chord error at H=1 m, 0.095% at 200 m).
- **Reverse-Z + a single frustum.** Pass `reversedDepthBuffer` to the renderer; it is
  implemented in the WebGPU backend today. Avoid `logarithmicDepthBuffer` (writes `frag_depth`,
  kills early-Z). The Cesium-style multi-frustum pair is *not* needed.

### 7.3 The shell

`H_s = 32 km`, `H_max = 6·H_s = 192 km`, inner bound at datum.

```
ρ(z) = ρ₀·exp(−z/H_s)  over [0, H_max], × smoothstep taper over the last e-fold
σ₀   = τ_nadir / (H_s·(1 − e^(−H_max/H_s)))
```

`z` is measured **from the local terrain surface** — soil sits on the ground — i.e.
`z = r − R − k·h(dir)`, sampled from the elevation cube inside the march. Since
`k·h_max = 26.5 km ≪ H_max`, analytic concentric bounds are unaffected; let the terrain
depth-clip terminate the march.

**Updated: that premise held only while matter exists.** "Inner bound at datum" and "let the
terrain depth-clip terminate the march," above, describe a solid Earth that is always there to
stop the ray — true throughout the sweep's erosion/collapse beats, and unchanged code for that
case. But spec 6.1's own governing rule is that matter can fully dissolve, and once it has,
nothing depth-clips the inner bound anymore: a ray that would have hit the terrain instead
keeps going — through empty interior space where solid Earth used to be — and re-enters the
shell on the planet's *far* side before exiting the outer boundary. That is a real second
crossing of the density field (`raySpheroid.ts`'s `intersectShell` already reports it as two
segments), not a continuation of the first, so it is marched as its own segment rather than
folded into a longer near-wall chord. Gated per-frame on two conditions, not marched
unconditionally: matter fully gone (`SweepState.matter <= 0`) — while it exists it genuinely
occludes the far wall, so marching it would both waste GPU work and add light that would be
physically blocked — AND altitude at or above §7.7's `REGIME2_TOP_ALT`, i.e. outside the Nelson
Max partition band, which assumes exactly one density column per direction and would be
over-weighted by an uncounterparted second one. See `regimes/Shell.ts`'s `ShellFarWallOpts`.

- `H_max ≥ 6·H_s` is required. Truncating at 2 e-folds leaves a **20% alpha discontinuity**
  on a hairline circle at the exact focal point of the composition; at 6 e-folds it is 0.41%.
  The smoothstep taper closes the rest C¹-continuously.
- Descent schedule: `H_s(alt) = clamp(alt/6, 0.2 km, 32 km)`.
- **Cap rendered nadir depth so the rim never blows:**
  `τ_rendered = min(τ_data, 4·sqrt(H_s/(2πR)))`. Without it the naive ramp drives τ_limb from
  1.7 to 35 during descent. The disc *fading* as you descend is correct — the LOD partition is
  handing that energy to the regime-2 instances.
- **Rate-limit `dH_s/dt`.** A per-frame scale height invalidates TAA history and ghosts
  specifically in the rim band — the one feature the piece exists to show.

**Exponential over uniform slab on shape grounds too:** a slab's chord collapses to zero at
the outer silhouette, so the ring is a spike with a null outside it; the exponential peaks at
the ground and decays monotonically, far better behaved for AA and TAA.

### 7.4 The limb is an authoring decision, not physics

**This corrects a claim made earlier in design.** The physical limb is over-saturated
(`τ_limb ≈ 860`, range 280–1,300, against a saturation threshold of ~5) **and invisible**:
the entire physical shell is 0.15–0.45 m of radial screen extent ≈ **10⁻⁵ pixels** at every
framing in the piece, contributing ~10⁻⁴ of one pixel's radiance.

Limb brightening does not "fall out of the physics for free." Any limb you see is *entirely*
an artefact of the chosen render thickness `H_s`. It is authored.

At the ruled shell (`Q = sqrt(2πR/H_s) = 35.37`) it does not saturate: `τ_limb = 1.66` at the
global mean (rim 81% opaque — strong, readable, unclipped), 4.2 in tundra/boreal, 0.14 in hot
desert. Saturation onset is `τ_nadir > 0.113`.

Tuning inversion if art direction wants a different rim: `H_s = 2πR·(τ_nadir/τ_limb)²`.

**The ring gets *wider* in pixels as you descend** (2.7 px @1080p at full-globe, 16.2 px at
400 km altitude) — the opposite of the earlier stated intent that it should tighten.

**Formula of record.** Uniform slab tangent chord: `2·sqrt(2RH + H²)`, ratio `2·sqrt(2R/H + 1)`.
Exponential profile — the one shipped: `Q = sqrt(2πR/H_s)`, set by the *profile scale height*,
not the geometric bound. "H = 64 km" is meaningless until you state how many e-folds live in
it; always resolve to `H_s` + `H_max`.

### 7.5 Terrain exaggeration

**k = 3.0 shipped, k = 3.6 hard maximum, ramped to 1.0 on descent.**

Rule: exaggerated Everest must not exceed one e-folding of the shell — `k ≤ H_s / 8.849 km`,
so `k_max = 3.62` at `H_s = 32 km`. Ship 3.0 for headroom (Everest 26.55 km, 2.25 px @1080p).

**10× is dropped.** It puts Everest at 88.5 km — 2.8 e-folds *outside* the shell, 7.49 px
against the shell's own 2.7 px e-fold, so the mountain reads as taller than the film it is
supposed to sit in. If more relief is wanted, raise `H_s`, not `k`.

Ramp: `k(alt) = clamp(H_s(alt)/8.849 km, 1.0, 3.6)` — one parameter drives shell and terrain
together, converging to true geometry by the regime-2 handoff.

Mandatory regardless of k: **depth-clip the raymarch against the terrain front face**, or the
shell renders over the mountains and the image inverts.

**Bathymetry:** clamp the shell's inner boundary to `r = R` (datum) regardless of seafloor
depth, and use a separate smaller `k_sea`. Marine nematodes occupy a 1–3 cm sheet, so their
shell is thinner and `Q` correspondingly larger — treat marine as its own profile class.

### 7.6 Slant column: closed-form Chapman

Two `exp()` per step. The slant-column LUT is **probably a pessimization** for the view ray:
it presumes profile parameters are constant along the whole slant path, but a tangent ray
sweeps 16.2° of arc ≈ 1,799 km of ground track across many biomes. It is exact only at nadir,
where N=1 anyway.

If a LUT ships (sun transmittance only, and only if it wins a benchmark): use the Bruneton
segment identity `column(entry→z_max) = C(entry) − C(z_max)` — two fetches into the *same* 2D
LUT, exact, no third axis — plus a ~10-class discrete biome axis. Drop the continuous
per-pixel shape exponent, which would destroy the class decomposition.

### 7.7 The LOD partition

```
j_inst = w·g,   j_vol = (1−w)·g
```

Energy-conserving **in expectation** at any optical depth, because the RTE at fixed τ is
linear in the source term. Derived from Nelson Max (IEEE TVCG 1995) eq. (4) — cite it as
"derived from", never as "Max states", since Max's derivation assumes identical spheres.

**Two conditions, both load-bearing:**
1. Every instance must be multiplied by transmittance sampled from the same `σ_t` field.
2. The medium must be emission-only or single-scattering.

**Do not write "exactly conserving."** Per-pixel Poisson error is `1/sqrt(N_pix)` — 10–14% at
target density, a visible brightness fluctuation as `w` grows.

**Splat widening is the primary mechanism, not a fallback.** Once population exceeds the draw
budget, rendered instances per pixel pins at `B/n_pixels` — at 1080p with B = 4 × 10⁶ that is
1.93/px = 72% Poisson noise at *every* altitude. `σ_pix = 3` lifts effective samples ~28× to
~55 (14% noise) and is the only reason the budget closes. Schedule `σ_pix` against the LOD
fraction: ~3 px at the 10 km end, tightening to 0.5–1 px as the handoff completes. Widening is
free in energy and fill but spends sharpness, so it is unusable low in regime 2.

**Normalize the kernel in the discrete pixel domain** (Schneegans-style weighted 2×2 splat).
A kernel normalized as a continuous function of screen area does not discretely sum to 1 at
pixel centres, so energy scintillates as instances drift across pixel boundaries. A minimum
radius is a cheap proxy, not the fix.

Per-instance cross-sections must be **drawn from the distribution, not set to the mean** —
A is heavily right-skewed with mean/median = 2.02.

### 7.8 The regime 2→3 handoff is unsolved and gets its own workstream

Full-population instances per pixel at nadir is `N_pix ≈ 2.20·d²`: 1 at d = 0.67 m, 55 at
d = 5 m. Below ~0.7 m one nematode exceeds a pixel; at 1 cm it spans ~4,500 px.

In the 1–5 m band the population exceeds any budget **and** `N_pix` is far below target **and**
kernel widening is unusable because the worms are individually resolvable. Additionally,
per-instance scalar intensity is only valid for Max's *spheres* — a 957 × 41 µm nematode has a
side-on/end-on projected-area ratio of **29.7**, so orientation-independent intensity cannot
smoothly meet a geometry-shaded hero organism.

**The `g = ρ·I` framework does not extend below ~5 m.** Assume oriented, geometry-shaded
proxies from ~5 m down. Budget this as its own design item (Exp-13).

---

## 8. Interaction

### 8.1 Camera

Two properties that cannot be retrofitted:

- **Scale-coupled motion.** Every parameter derives from altitude. Scroll is logarithmic
  descent, so constant input gives constant perceptual approach across all ten orders.
- **Floating origin.** Camera at origin, world translates. At millimetre scale, float32 world
  coordinates anchored at the planet's centre lose catastrophic precision.

Note sphere-space altitude `h = sqrt(L² + s²) − 1` is a *reduced* altitude, not geodetic;
error O(f·H) ≈ 215 m in a 64 km shell, below the profile's resolution.

### 8.2 Controls and layers

Drag orbits. Scroll descends. Double-click flies to a point. Escape returns to orbit.

Layers: veil (always), terrain (off), provenance desaturation (on), functional-group tints
(auto below ~1 km), depth scrub (auto near surface), sample points (off), extrapolation mask
(off).

**Matter slider** — scrubs the sweep manually. The cinematic is the first pass; the slider is
what people actually play with.

### 8.3 The readout

Altitude, local density, provenance class, and **estimated count in view**. Watching that
counter fall from 10¹⁹ to 1 during the descent is what converts the number into a feeling.
Every other UI element serves it.

---

## 9. Modules

```
pipeline/                       Python · uv · offline
  fetch/eth_raster.py           range-GET row extraction
  fetch/etopo.py  ghsl.py  marine.py
  build/reproject.py            equirect → EAC cube, geodetic→geocentric
  build/compose.py              fuse land + marine + urban
  build/encode.py               log-encode, quantize, pack, per-channel WebP
  build/validate.py             proposed integral check and global percentiles (see §5.4)
  out/                          → app/public/data

app/src/
  core/Renderer.ts              WebGPU init, frame loop, post
  core/ScaleCamera.ts           altitude-coupled + floating origin + reverse-Z
  core/Field.ts                 cube sampler + analytic depth
  regimes/Shell.ts              raymarch, Chapman column, silhouette AA
  regimes/Particles.ts          compute instancing, indexed indirect draw
  regimes/Proxies.ts            oriented geometry, 5 m → 1 cm  (§7.8)
  regimes/Hero.ts               single organism, swim gait
  regimes/Blend.ts              Max partition, σ_pix scheduling
  sweep/Timeline.ts  Dissolve.ts
  ui/Store.ts  Panel.ts  CobbToggle.ts
  shaders/                      TSL modules
```

`Field.ts` is the load-bearing interface: it exposes `sampleDensity(dir, z)` and all regimes
consume only that. Swap the atlas and every regime changes together.

`Blend.ts` is isolated so it can be exercised against a synthetic field with no data pipeline,
no UI, no sweep.

---

## 10. Platform constraints

**Pin `three@0.185.1` exactly.** `@types/three` at `^0.185.4` — three ships no `types` key.

- **Indexed vs non-indexed indirect draw.** `drawIndirect` reads 4×u32; `drawIndexedIndirect`
  reads 5 words with a **signed** `baseVertex`. The canonical three.js example is the
  *non-indexed* layout — copying it onto indexed geometry lands `firstInstance` in `baseVertex`
  and silently mis-draws. Regime-2 quads are indexed → use the 5-word layout.
- **`firstInstance` must be 0.** Nonzero without the `indirect-first-instance` feature makes
  the entire draw a **silent no-op** — invisible on a discrete-GPU dev box, blank screen on a
  reviewer's laptop. Compact survivors so the live range starts at 0.
- **Design for one frame of latency** between the compute that writes indirect args and the
  draw that consumes them. The official example deliberately runs compute *after* render.
- **Camera nodes cannot reach compute shaders** (#27570, open since 2024). Pass
  `uniform(new THREE.Vector3())` and `uniform(new THREE.Matrix4())` explicitly.
- **Explicit LOD is mandatory** inside any fragment-stage march containing a transmittance
  `Break()`. WGSL's `derivative_uniformity` rule has default severity *error*. Use `.level(...)`.
- **Seamless cube filtering in compute is free.** WGSL has no `textureLoad` overload for
  `texture_cube` at all; `textureSampleLevel`/`textureSampleGrad` carry no fragment-only
  restriction and get spec-guaranteed seamless cross-face filtering. Regime 2 shares the field
  with Regime 1 at zero additional code — no hand-written face-crossing bilinear.
- **`VolumeNodeMaterial` is geometry-bounded, not box-bounded** — a sphere mesh supplies the
  outer shell bound with no custom code. Template: `RaymarchingBox` in
  `three/addons/tsl/utils/Raymarching.js`; swap `hitBox` for two ray-sphere intersections.
- **GPU timestamps are quantized to 100 µs in Chrome.** On a 2 ms pass that is 20 buckets —
  any adaptive controller must average across frames.
- **Browser support is not "all three engines."** Firefox 141+ is a partial implementation
  with **no Linux and no Intel macOS**; Chrome is full only from 144 (Linux limited to Intel
  Gen12+); Safari 26 is solid. Ship a `WebGPU.isAvailable()` gate and a static poster, and
  describe it honestly as accepting a known desktop-Linux / Intel-Mac gap.

---

## 11. Budget

**First-paint ceiling: 12 MB.** That is what closes 3 s at 50 Mbps with realistic overhead.

Current projection **does not close**: atlas ≈ 8.3 MB + elevation cube 22.6 MB + three/webgpu
0.26 MB = **31.2 MB ≈ 6.8 s realistic**.

Compression on this data class is **~2.3×, not 21.8×**. Measured on the same
quantile-random-forest raster class: order-0 entropy 7.19 bits/texel, brotli 1.47×, WebP
lossless 1.63×, WebP q95 2.26×. Coarser texels compress *worse* (1.65× at 20 km/px), so
capping the pyramid buys no bytes.

**The elevation cube is 94% of the payload and is the asset to attack** — ship coarse at first
paint, stream refinement. Serve pre-compressed `.br` with explicit `Content-Encoding` (CDNs do
not brotli `application/octet-stream` by default; in-page `DecompressionStream` supports only
gzip/deflate).

BC4 is re-opened: fixed 2.0× (3.15 MB, half the VRAM, zero decode) is within 13% of WebP q95
on wire size. Real-data PSNR is **40.6 dB**, not the 60.6 dB measured on a synthetic smooth
field. If BC4 ships, encode no-data as a sentinel outside the density range — BC4 interpolates
across each 4×4 block, so zero no-data bleeds into coastline texels.

**Performance is entirely unmeasured.** No WebGPU raymarching timings exist publicly. The
1.5–2.5 ms figure is a **hypothesis to falsify in week one, never a budget** — the published
anchors (Sakmary, Horizon Zero Dawn) do not survive scrutiny as extrapolation bases.

---

## 12. Day-one experiments

Ordered by value. Each has a pass/fail criterion.

| # | Experiment | Pass | Fail → |
|---|---|---|---|
| 1 | Shell march cost, N=128, full-globe framing, three machines | ≤ 2.5 ms @1080p | half-res + temporal upsample first |
| 2 | Inspect generated WGSL for the march loop | explicit LOD, clean compile, no unroll past ~8× | restructure as outer loop + inner flag |
| 3 | LUT vs closed-form Chapman benchmark | closed form wins/ties → drop view-ray LUT | ship LUT with Bruneton segment identity |
| 4 | three.js #33795 same-frame storage-texture hazard (20 lines) | values match → design nothing around it | insert `convertToTexture()`, benchmark |
| 5 | TAA on a 1–3 px limb ring, 30 s orbit | no ghosting, no ring erosion | exclude rim band from history clamping |
| 6 | Splat width perception at the 10 km handoff | σ_pix = 3 indistinguishable from raymarch | narrow band / raise budget / move crossover |
| 7 | Per-instance transmittance cost, 4e6 instances | ≤ 1 ms | screen-space transmittance buffer from regime 1 |
| 8 | Compression on the *real* rasters | first paint ≤ 12 MB | cut elevation res, stream refinement |
| 9 | Reverse-Z single frustum, near 1 m / far 1e9 m | no z-fighting on the descent | only then consider a second frustum |
| 10 | Indexed indirect draw without `indirect-first-instance` | correct instance count | — |
| 11 | iOS Safari ceiling on real hardware, throttled | no OOM, first paint < 8 s | reduced mobile tier |
| 12 | Silhouette stability, 30 s orbit at 30,000 km | no shimmer on the rim | widen smoothstep, floor N from footprint |
| 13 | Regime 2→3 handoff probe, 5 m → 0.5 m | no pop, no brightness discontinuity | becomes its own workstream (§7.8) |

**#33795 is unsettled, not a constraint** — zero comments, no maintainer triage, and the repro
exercises only `RenderPipeline`. The factorized field is baked, so the project may never touch
that path. Run Exp-4 before designing around it.

---

## 13. Build sequence

**Session 1 — falsify.** Exp-1, 2, 4, 9. No data pipeline, no UI, no sweep: a sphere, a
synthetic density function, and a shell march. Every one of these can invalidate an
architectural decision, and all four are cheap. The original plan aimed session 1 at the LOD
handoff; that turned out to be the *solved* part.

**Session 2 — the field is real.** Range-GET extraction, EAC bake with geodetic→geocentric,
the proposed integral check (§5.4), global percentiles. Shell renders real data.

**Session 3 — the sweep.** Timeline, dissolve, bathymetry collapse, matter slider, Cobb
inversion toggle.

**Session 4 — descent.** Regime 2 instancing, the partition, σ_pix scheduling. Exp-5, 6, 7, 12.

**Session 5 — the bottom.** Regime 2→3 proxies and the hero organism (§7.8). Scoped separately
because it is genuinely unsolved.

**Session 6 — polish.** Post chain, readout, provenance overlay, mobile tier, load budget.

---

## 14. Attribution

- van den Hoogen, J. et al. *Soil nematode abundance and functional group composition at a
  global scale.* Nature 572, 194–198 (2019). doi:10.1038/s41586-019-1418-6
- Prediction rasters: ETH Research Collection doi:10.3929/ethz-b-000354035 and
  doi:10.3929/ethz-b-000354394 — In Copyright, Non-Commercial Use Permitted
- Sample records: Geisen, van den Hoogen, Data Providers and Crowther, *Nematode_abundance_dataset*, version 1. The CSV item carries CC0 1.0. [Figshare dataset](https://doi.org/10.6084/m9.figshare.11733897.v1)
- ETOPO 2022, NOAA NCEI — CC0
- Built-up surface: Pesaresi and Politis, European Commission Joint Research Centre, GHS-BUILT-S R2023A. The catalogue specifies the European Commission reuse notice. [Dataset record](https://data.jrc.ec.europa.eu/dataset/9f06f36f-4b11-47ec-abb0-4f8b7b1d72ea)
- Optical model derived from Max, N. *Optical Models for Direct Volume Rendering.*
  IEEE TVCG 1(2):99–108 (1995)
- Screen normalization: Schneegans, Kreskowski & Gerndt, *Smaller than Pixels*, EG 2025

Ship the derivation notebook. Every number in the UI regenerates from source, and that is a
credibility asset rather than an appendix.
