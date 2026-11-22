"""The density cube's G channel: structure fraction.

Per the team-lead's binding channel ruling, G carries a single continuous
scalar -- `(omnivores + predators) / (sum of the five functional groups)`
-- not a categorical "dominant group" index and not a 5-way composition
recoverable from one channel. Two properties this shape was chosen for:

  - Continuous, so it survives `LinearFilter` magnification (a categorical
    index would blend into meaningless numbers between texels).
  - It carries the beat the urban-claim ruling identified as real:
    omnivores and predators collapse 45-80% in urban soils while
    bacterivores/fungivores rise, i.e. cities change WHICH nematodes, not
    how many -- this is the one number that shows that shift.

**The summation trap** (the reason this module exists rather than just
dividing by `TotalNumber`): the five `*_per100gSoil.tif` group rasters were
modelled INDEPENDENTLY of `TotalNumber_per100g.tif`, so they do not sum to
it. Measured on the real 512x256 extraction (see `rebake_gb.py`'s report):
per-pixel `sum_five/TotalNumber` ranges from 0.54 to 2.33 (median 1.086),
and the global area-weighted ratio is 1.0875 -- the five groups sum to
~8.75% MORE than the independently-modelled total, on average. Normalising
G by `TotalNumber` would silently encode that model disagreement as
ecology. This module always normalises by `sum_five`, never `TotalNumber`
-- see `summation_ratio`/`global_summation_ratio` for the check that
justifies this instead of assuming it.

Every function here is a pure array transform -- no I/O, no network -- so
this module is exercised entirely from `test_functional_group.py`'s
synthetic fixtures (real-data numbers are pinned as regression constants
there, not refetched).
"""

from __future__ import annotations

import numpy as np

GROUP_NAMES = ["bacterivores", "fungivores", "herbivores", "omnivores", "predators"]


def group_sum(
    bacterivores: np.ndarray,
    fungivores: np.ndarray,
    herbivores: np.ndarray,
    omnivores: np.ndarray,
    predators: np.ndarray,
) -> np.ndarray:
    """Elementwise sum of the five functional-group grids (the ruling's
    chosen G denominator -- see module docstring)."""
    return bacterivores + fungivores + herbivores + omnivores + predators


def structure_fraction(
    omnivores: np.ndarray, predators: np.ndarray, sum_five: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """`(omnivores + predators) / sum_five`, 0.0 where sum_five <= 0
    (ocean / no-data -- matches `eth_raster.extract_grid`'s NaN->0
    mass-conserving convention, and the density bake's DENSITY_FLOOR
    reasoning: a 0/0 here must decode to a defined, finite value, never
    NaN or inf reaching the shader).

    Returns `(g, n_clipped, max_clip_extent)`:
      - `g`: the fraction, clipped to [0,1] as defensive insurance
        (matching the existing pipeline's clip-after-compute convention
        on R/A) against RF regression predicting a slightly negative
        per-group count near zero, which could otherwise push the raw
        ratio fractionally outside [0,1].
      - `n_clipped`: how many cells actually needed clipping -- reported,
        not swept under the rug (measured 0 of 32,916 land cells on the
        real 512x256 extraction).
      - `max_clip_extent`: the largest |raw - clipped| amount seen, for
        audit purposes (0.0 if n_clipped == 0).
    """
    safe_denom = np.where(sum_five > 0, sum_five, 1.0)
    raw = (omnivores + predators) / safe_denom
    raw = np.where(sum_five > 0, raw, 0.0)
    g = np.clip(raw, 0.0, 1.0)
    clip_amount = np.abs(raw - g)
    out_of_range = clip_amount > 0
    n_clipped = int(out_of_range.sum())
    max_clip_extent = float(clip_amount.max()) if n_clipped else 0.0
    return g, n_clipped, max_clip_extent


def summation_ratio(sum_five: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Per-pixel `sum_five / total`, NaN where total <= 0 (undefined, not
    0 -- unlike `structure_fraction`, "no data" here genuinely has no
    sensible ratio, so this is deliberately NOT mass-conserving-zeroed).
    This is the diagnostic that establishes the summation trap; it is
    never used to compute G itself."""
    denom = np.where(total > 0, total, np.nan)
    return sum_five / denom


def global_summation_ratio(
    sum_five: np.ndarray, total: np.ndarray, area_weights: np.ndarray
) -> float:
    """Area-weighted GLOBAL `sum_five/total` ratio -- a pooled ratio of
    sums, `Sum(area*sum_five) / Sum(area*total)`, not a mean of per-pixel
    ratios (which would overweight small/noisy denominators). Cells with
    total <= 0 (ocean) drop out of both sums cleanly, no masking needed."""
    mask = total > 0
    numerator = float(np.sum((area_weights * sum_five)[mask]))
    denominator = float(np.sum((area_weights * total)[mask]))
    return numerator / denominator


def global_structure_fraction(
    omnivores: np.ndarray,
    predators: np.ndarray,
    sum_five: np.ndarray,
    area_weights: np.ndarray,
) -> float:
    """Area- AND abundance-weighted global average of the structure
    fraction:

        Sum(area_i * (omni_i + pred_i)) / Sum(area_i * sum_five_i)

    This is the number to compare against Table 1's published
    "omnivores 8.9% + predators 4.5% = 13.4% of individuals" -- that
    figure is itself pooled across individuals (not averaged per sample
    site), so the matching computation here pools by LOCAL abundance
    (sum_five, a per100g proxy for "how many individuals live here" --
    valid per the ruling's point that per-100g and per-m2 group
    *fractions* are identical) as well as by true spherical area (so the
    equirect grid's polar over-representation doesn't skew the figure the
    way an unweighted per-cell mean would).

    Ocean / no-data cells (sum_five == 0) contribute 0 to both the
    numerator and denominator automatically -- no separate masking
    needed, and no risk of a 0/0 division since the division happens once
    on the aggregate, not per-pixel.
    """
    numerator = float(np.sum(area_weights * (omnivores + predators)))
    denominator = float(np.sum(area_weights * sum_five))
    return numerator / denominator
