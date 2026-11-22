"""Geodetic <-> geocentric latitude conversion, ported line-for-line from
`app/src/math/wgs84.ts` (read-only reference; not modified, not imported --
this is a separate Python port so the pipeline has no JS runtime dependency).

WGS84 ellipsoid. `tan(phi_gc) = (1-f)^2 * tan(phi_gd)`, implemented via
atan2 (not tan/atan) so the poles do not go through infinity, matching the
TS reference exactly.
"""

from __future__ import annotations

import math

A = 6_378_137.0
INV_F = 298.257223563
F = 1.0 / INV_F
B = A * (1.0 - F)
R_MEAN = 6_371_000.0

# (1 - f)^2, the factor relating the two latitude conventions.
K = (1.0 - F) * (1.0 - F)


def geodetic_to_geocentric_lat(lat_rad: float) -> float:
    return math.atan2(K * math.sin(lat_rad), math.cos(lat_rad))


def geocentric_to_geodetic_lat(lat_rad: float) -> float:
    return math.atan2(math.sin(lat_rad), K * math.cos(lat_rad))


def max_lat_offset_arcmin() -> float:
    """Maximum |phi_gd - phi_gc| in arcminutes, found by ternary search --
    same algorithm as the TS reference's `maxLatOffsetArcmin`."""

    def offset(lat: float) -> float:
        return lat - geodetic_to_geocentric_lat(lat)

    lo, hi = 0.0, math.pi / 2
    for _ in range(200):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if offset(m1) < offset(m2):
            lo = m1
        else:
            hi = m2
    return offset((lo + hi) / 2) * 180 * 60 / math.pi
