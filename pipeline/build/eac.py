"""Cube-face direction math for two projections, matching the face-embedding
convention already established in `app/src/field/bakeCube.ts::faceDirection`
(read-only reference, not modified/imported) and this app's Z-polar
lat/lon convention (`app/src/math/wgs84.ts`'s doc comment, `synthetic.ts`'s
`absLat = asin(z)`, `orbitInput.ts`'s `projectedUp` doc, `ScaleCamera.ts`'s
`[r*cos(lat)*cos(lon), r*cos(lat)*sin(lon), r*sin(lat)]`).

Face order [0..5] = [+X,-X,+Y,-Y,+Z,-Z], matching `THREE.CubeTexture`'s
`images` array order. `faceDirection`'s per-face embedding (u,v in [-1,1]):

    face 0 (+X): [ 1, -v, -u]      face 1 (-X): [-1, -v,  u]
    face 2 (+Y): [ u,  1,  v]      face 3 (-Y): [ u, -1, -v]
    face 4 (+Z): [ u, -v,  1]      face 5 (-Z): [-u, -v, -1]

Two projections share this exact embedding, differing only in what is fed
into it as (u,v):

  - PLAIN (gnomonic): raw linear NDC (u,v) -- this is what `bakeCube.ts`'s
    existing bake already does.
  - EAC (equi-angular): the arctan-warped tangent `(tan(u*pi/4), tan(v*pi/4))`.

Team-lead direction: bake BOTH and measure, rather than assume EAC is
worth its runtime warp cost (see cube-bake-both-report.md). Both use the
same embedding table so they stay directly comparable -- the warp is the
only delta.
"""

from __future__ import annotations

import math

import numpy as np

FACE_COUNT = 6
FACE_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def eac_warp(t: np.ndarray) -> np.ndarray:
    """Linear NDC in [-1,1] -> EAC-warped tangent-space coordinate."""
    return np.tan(t * (math.pi / 4))


def eac_unwarp(t: np.ndarray) -> np.ndarray:
    """Inverse of `eac_warp`."""
    return np.arctan(t) * (4.0 / math.pi)


def plain_warp(t: np.ndarray) -> np.ndarray:
    """Identity: plain gnomonic feeds linear NDC straight into the face
    embedding, unwarped. Named (not just "use raw u,v") so call sites read
    the same whether they're building the plain or EAC path."""
    return t


def plain_unwarp(t: np.ndarray) -> np.ndarray:
    return t


WARPS = {"plain": (plain_warp, plain_unwarp), "eac": (eac_warp, eac_unwarp)}


def _embed(face: int, uw, vw):
    """Shared face-embedding table (see module docstring), given ALREADY
    warped (or unwarped, for plain) tangent-space coordinates."""
    if face == 0:
        return np.ones_like(uw), -vw, -uw
    if face == 1:
        return -np.ones_like(uw), -vw, uw
    if face == 2:
        return uw, np.ones_like(uw), vw
    if face == 3:
        return uw, -np.ones_like(uw), -vw
    if face == 4:
        return uw, -vw, np.ones_like(uw)
    if face == 5:
        return -uw, -vw, -np.ones_like(uw)
    raise ValueError(f"face index {face} out of range [0,5]")


def face_direction(
    face: int, u: np.ndarray, v: np.ndarray, projection: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UNNORMALIZED direction for linear NDC (u,v) in [-1,1] on `face`,
    under `projection` ("plain" or "eac"). Caller normalizes."""
    warp, _ = WARPS[projection]
    return _embed(face, warp(u), warp(v))


def direction_to_face_uv(
    x: float, y: float, z: float, projection: str
) -> tuple[int, float, float]:
    """Inverse of `face_direction`: unit (or any-scale) direction ->
    (face, u, v) with u,v in [-1,1].

    Derived by solving each face's embedding equation for (u_tan, v_tan)
    given the dominant axis, then un-warping. E.g. face 0: (x,y,z) is
    proportional to (1,-v,-u), so dividing through by x gives
    u_tan=-z/x, v_tan=-y/x. See module docstring for the full table; the
    other five faces follow the same substitution pattern.
    """
    _, unwarp = WARPS[projection]
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        face = 0 if x > 0 else 1
        if face == 0:
            u_tan, v_tan = -z / x, -y / x
        else:
            u_tan, v_tan = -z / x, y / x
    elif ay >= ax and ay >= az:
        face = 2 if y > 0 else 3
        if face == 2:
            u_tan, v_tan = x / y, z / y
        else:
            u_tan, v_tan = -x / y, z / y
    else:
        face = 4 if z > 0 else 5
        if face == 4:
            u_tan, v_tan = x / z, -y / z
        else:
            u_tan, v_tan = x / z, y / z
    return face, float(unwarp(u_tan)), float(unwarp(v_tan))


def lonlat_to_direction(
    lon_rad: np.ndarray, lat_rad: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This app's Z-polar convention: x=cos(lat)cos(lon), y=cos(lat)sin(lon),
    z=sin(lat) -- exactly `ScaleCamera.ts`'s position formula with r=1."""
    cl = np.cos(lat_rad)
    return cl * np.cos(lon_rad), cl * np.sin(lon_rad), np.sin(lat_rad)


def direction_to_lonlat(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `lonlat_to_direction` for a UNIT vector: lon=atan2(y,x),
    lat=asin(z)."""
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    return lon, lat


def rect_solid_angle(x1: float, x2: float, y1: float, y2: float) -> float:
    """Exact solid angle subtended by the axis-aligned rectangle
    [x1,x2]x[y1,y2] on the plane z=1, viewed from the origin. Standard
    closed form for a rectangular light/cubemap-texel solid angle:

        Omega = F(x2,y2) - F(x1,y2) - F(x2,y1) + F(x1,y1)
        F(x,y) = atan2(x*y, sqrt(1+x^2+y^2))

    Verified against the known whole-face result: integrating over the
    full face [-1,1]x[-1,1] gives exactly 2*pi/3 (1/6 of the full sphere's
    4*pi) -- see `test_bake.py::test_rect_solid_angle_full_face`.
    """

    def f(x: float, y: float) -> float:
        return math.atan2(x * y, math.sqrt(1.0 + x * x + y * y))

    return f(x2, y2) - f(x1, y2) - f(x2, y1) + f(x1, y1)


def _rect_solid_angle_vec(x1, x2, y1, y2):
    """Vectorized `rect_solid_angle`, broadcasting over arrays."""

    def f(x, y):
        return np.arctan2(x * y, np.sqrt(1.0 + x * x + y * y))

    return f(x2, y2) - f(x1, y2) - f(x2, y1) + f(x1, y1)


def texel_solid_angles(face_size: int, projection: str) -> np.ndarray:
    """Exact solid angle (steradians, on the unit sphere) for every texel
    of one face under `projection`, shape (face_size, face_size). Identical
    for all 6 faces by symmetry, so computed once and reused. Vectorized
    (not a face_size^2 Python double loop) so it stays cheap even at the
    fine face_size needed for the plain-gnomonic ratio to approach its
    5.196:1 continuum limit -- see `test_plain_texel_solid_angle_variation_*`
    for why that matters: at coarse face_size (e.g. 64) the discretized
    ratio is still well short of 5.196 (~5.03), converging slowly."""
    warp, _ = WARPS[projection]
    edges = np.linspace(-1.0, 1.0, face_size + 1)
    warped_edges = warp(edges)  # tangent-space face-plane coordinates
    x1 = warped_edges[:-1][None, :]
    x2 = warped_edges[1:][None, :]
    y1 = warped_edges[:-1][:, None]
    y2 = warped_edges[1:][:, None]
    return _rect_solid_angle_vec(x1, x2, y1, y2)
