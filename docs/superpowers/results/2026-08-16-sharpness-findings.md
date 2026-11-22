# Terrain resolution and coastlines, 16 August 2026

The August prototype drew soft coastlines and muted relief. Two layers contributed independently. Terrain height came from an elevation texture with 256 cells per cube face edge. Coast color used the coverage channel of an older 128-face soil texture.

## Terrain probes

The investigation sampled two ETOPO-derived elevation atlases at the same locations using an independent latitude/longitude-to-cube lookup. The figures below are atlas readings in metres. Each averages terrain over the cell footprint, which varies across a cube face. They are unsuitable as survey elevations for a summit or a named point. [NOAA ETOPO 2022](https://doi.org/10.25921/fd45-gt74)

| Probe | 256-face atlas (39.1 km nominal spacing) | 1024-face atlas (9.8 km nominal spacing) |
| --- | ---: | ---: |
| Everest | 5,480 | 7,048 |
| Challenger Deep | −8,012 | −10,024 |
| Kilimanjaro | 3,166 | 4,260 |
| Death Valley | +536 | −69 |
| Netherlands | 5 | 18 |
| Fezzan, Sahara | 745 | 738 |
| Mid-Pacific abyssal plain | −4,284 | −4,324 |

At Death Valley, the larger footprint included enough higher ground to put its mean above sea level. The smaller footprint put it below sea level. The change was much smaller on the broad Saharan and deep-Pacific probes. Six uncompressed 1024-face RGBA16Float textures occupied 48 MiB. Progressive loading allowed a coarse view to appear first. [256-face elevation metadata](../../../app/public/data/elevation_plain_256/cube_meta.json) · [1024-face elevation metadata](../../../app/public/data/elevation_plain_1024/cube_meta.json)

## Coast at 10° S, Brazil

| Longitude | Coverage in the historical soil atlas | Elevation in the 1024-face atlas |
| ---: | ---: | ---: |
| −37.5° | 1.000 | 205 m |
| −37.0° | 0.928 | 62 m |
| −36.5° | 0.951 | 140 m |
| −36.0° | 0.680 | −5 m |
| −35.5° | 0.128 | −1,612 m |
| −35.0° | 0.001 | −3,492 m |

The old coverage layer went from 1.000 to 0.001 over 2.5° of longitude, roughly 275 km. The elevation samples bracketed sea level in a single 0.5° step. Using the coarse soil coverage for the coastline blurred the transition independently of the terrain resolution.

## Why bed elevation alone was insufficient

The 1024-face bed elevation was −245 m at Vostok and −913 m at the Byrd Subglacial Basin. An elevation-sign rule would classify those ice-covered locations as ocean. The first proposed replacement counted ice thickness alone, which also admitted floating ice. The subsequent terrain bake uses bed and ice-surface height in a flotation approximation. It stores the resulting area-weighted land fraction in the elevation atlas alongside height. Below-sea-level basins and partially covered coastal cells remain limitations. [Land-mask calculation](../../../pipeline/build/bake_elevation_cube.py) · [Recorded mask checks](../../../app/public/data/elevation_plain_512/cube_meta.json)
