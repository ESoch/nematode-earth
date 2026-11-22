# Why the soil model has no extra city multiplier

The exhibit offers two views. **Soil model** displays the published abundance field with a multiplier of 1. **Cobb illustration** applies `1 + 3u`, where `u` is the local built-up fraction. The density multiplier reaches four at full coverage. It is an artistic interpretation of Cobb’s description. Cobb supplied no number for it. [Cobb, p. 472](https://archive.org/details/yoa1914/page/472/mode/2up) · [Viewer implementation](../../../app/src/field/cobb.ts)

## The twenty-city comparison

In August 2026, the project compared modelled soil abundance and built-up surface across twenty one-degree city windows. Each city’s reference was the mean density of its own pixels with less than 2% built-up cover. The analysis excluded missing density values and trimmed pooled abundance ratios below the 0.5th and above the 99.5th percentile. It fitted the remaining ratios with a quadratic, `m(u) = 1 + a·u + b·u²`, pooled across cities. The intercept was fixed at 1. [Soil raster](https://doi.org/10.3929/ethz-b-000354035) · [Built-up surface](https://doi.org/10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA) · [Window selection](../../../pipeline/build/city_windows.py) · [Fitting code](../../../pipeline/build/city_calibration.py)

| Quantity | Recorded result |
| --- | ---: |
| Mapped pixels after trimming | 243,294 |
| Linear coefficient, a | −0.0085 |
| Nominal 95% pixel-bootstrap interval for a | −0.035 to +0.019 |
| R² | −0.0003 |

The fitted curve did slightly worse than predicting the sample mean. The interval comes from 2,000 independent pixel resamples with seed 42. Nearby pixels share spatial structure and each window shares a rural baseline. The interval for the linear coefficient omits those dependencies and cannot establish the full urban–rural contrast, which also depends on the quadratic coefficient. This comparison gave us no basis for applying the originally proposed suppression curve to the model. [Methods and interpretation](https://nematode.earth/about#claim)

## What the field studies support

The updated soil database contains 6,825 sample records aggregated into 1,933 covariate pixels. In the project’s historical analysis of those pixels, the correlation between log human population density and log nematode abundance was +0.091. Within the city-window analysis, descriptive correlations differed in sign, including +0.61 for Moscow and −0.27 for Amsterdam. These associations cannot establish what happens to soil populations beneath pavement. [2020 data paper](https://doi.org/10.1038/s41597-020-0437-3) · [Aggregated covariate dataset](https://doi.org/10.6084/m9.figshare.11733891)

Li et al. estimated 27% higher total abundance in the urban category than in primary habitat. The urban total did not differ significantly from the other land-use categories. Their analysis concerns soil samples classified by regional land use. It cannot justify multiplying every urban pixel in this exhibit by a fixed factor. [Li et al. (2022), results §3.1 and Fig. S4](https://doi.org/10.1111/1365-2656.13744)

This account incorporates the September review of the original design note’s ecological interpretations. [Science and methods audit](2026-09-22-science-making-audit.md)
