#!/usr/bin/env python3
"""Versioned soil candidate builder, separate from the unchanged legacy bake.

R is an all-cell mean; A is valid-model support, used only as an empty-cell
gate. B is a support-normalized mean native bootstrap CV, NOT an uncertainty
estimate for an aggregate count. G retains separately documented coarse data.
No network, public-directory writes, replacement, or automatic promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from eac import FACE_NAMES, face_direction, texel_solid_angles
from wgs84 import K, R_MEAN

SCHEMA = "soil-cube-candidate-v1"
GROUP_NAMES = ("bacterivores", "fungivores", "herbivores", "omnivores", "predators")
MAX_PRODUCT_BYTES = 100_000_000
# Authored optical conversion used by app/src/math/optical.ts, not a new
# ecological measurement. Retained only to audit the loader's second rounding.
OPTICAL_AREA = 1.49e-8


@dataclass
class Grid:
    """Area-mean numerator and its own support on explicit geographic edges."""

    values: np.ndarray
    support: np.ndarray
    lon_edges: np.ndarray
    lat_edges: np.ndarray
    provenance: dict

    def __post_init__(self):
        for name in ("values", "support", "lon_edges", "lat_edges"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if self.values.ndim != 2 or min(self.values.shape) < 1:
            raise ValueError("Grid must be a nonempty two-dimensional field")
        h, w = self.values.shape
        if (self.support.shape != (h, w) or self.lon_edges.shape != (w + 1,)
                or self.lat_edges.shape != (h + 1,)):
            raise ValueError("Grid/support/geographic edge shapes disagree")
        if not all(np.isfinite(a).all() for a in
                   (self.values, self.support, self.lon_edges, self.lat_edges)):
            raise ValueError("Nonfinite grid values/edges/support are forbidden")
        if (np.any(self.values < 0) or np.any(self.support < 0) or np.any(self.support > 1)
                or np.any((self.support == 0) & (self.values != 0))):
            raise ValueError("Invalid support fraction or unsupported numerator")
        if (np.any(np.diff(self.lon_edges) <= 0) or np.any(np.diff(self.lat_edges) >= 0)
                or self.lon_edges[0] < -180 or self.lon_edges[-1] > 180
                or self.lat_edges[0] > 90 or self.lat_edges[-1] < -90):
            raise ValueError("Invalid geographic cell edges")

    def integral(self):
        area = (np.sin(np.deg2rad(self.lat_edges[:-1]))
                - np.sin(np.deg2rad(self.lat_edges[1:])))[:, None]
        area = area * np.deg2rad(np.diff(self.lon_edges))[None, :] * R_MEAN**2
        return float(np.sum(self.values * area))


class SupportError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("Missing auxiliary support: " + json.dumps(diagnostics))


def _indices(centers, query):
    """Linear interpolation on increasing, possibly unequal cell centers."""
    hi = np.clip(np.searchsorted(centers, query, side="right"), 0, len(centers) - 1)
    lo = np.maximum(hi - 1, 0)
    span = centers[hi] - centers[lo]
    f = np.divide(query - centers[lo], span, out=np.zeros_like(query), where=span > 0)
    return lo, hi, np.clip(f, 0, 1)


def sample_grid(grid, lon, lat):
    """Return interpolated numerator/support; no borrowing another grid's CRS.

    Latitude clamps only within the explicit extent. Only 360-degree grids
    wrap longitude. Outside extent has zero support, never extrapolated data.
    """
    lon, lat = np.broadcast_arrays(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Nonfinite query coordinates")
    xc = (grid.lon_edges[:-1] + grid.lon_edges[1:]) / 2
    yc = -(grid.lat_edges[:-1] + grid.lat_edges[1:]) / 2
    periodic = np.isclose(grid.lon_edges[-1] - grid.lon_edges[0], 360, rtol=0, atol=1e-10)
    inside = (lat <= grid.lat_edges[0]) & (lat >= grid.lat_edges[-1])
    if periodic:
        lon = (lon - grid.lon_edges[0]) % 360 + grid.lon_edges[0]
        extended = np.r_[xc[-1] - 360, xc, xc[0] + 360]
        c0, c1, fx = _indices(extended, lon)
        c0, c1 = (c0 - 1) % len(xc), (c1 - 1) % len(xc)
    else:
        inside &= (lon >= grid.lon_edges[0]) & (lon <= grid.lon_edges[-1])
        c0, c1, fx = _indices(xc, lon)
    r0, r1, fy = _indices(yc, -lat)

    def interpolate(a):
        value = ((1 - fy) * ((1 - fx) * a[r0, c0] + fx * a[r0, c1])
                 + fy * ((1 - fx) * a[r1, c0] + fx * a[r1, c1]))
        return np.where(inside, value, 0)

    return interpolate(grid.values), interpolate(grid.support)


def _channels(density, cv, groups, lon, lat, *, separate_groups=False):
    if cv is None or groups is None:
        raise ValueError("Both CV and functional-group grids are required; missing channel")
    r, a = sample_grid(density, lon, lat)
    bn, bs = sample_grid(cv, lon, lat)
    gn, gs = sample_grid(groups, lon, lat)
    supported = a > 0
    b = np.divide(bn, bs, out=np.zeros_like(bn), where=bs > 0)
    g = np.divide(gn, gs, out=np.zeros_like(gn), where=gs > 0)
    if np.any((g < 0) | (g > 1)):
        raise ValueError("Functional-group fraction outside [0,1]")
    channels = np.stack((np.where(supported, r, 0), gn if separate_groups else np.where(supported, g, 0),
                         np.where(supported, b, 0), a), axis=-1)
    return channels, supported & (bs <= 0), supported & (gs <= 0), bs, gs


def sample_channels(density, cv, groups, lon, lat):
    channels, bm, gm, _, _ = _channels(density, cv, groups, lon, lat)
    if bm.any() or gm.any():
        raise SupportError({"cv_missing_texels": int(bm.sum()), "groups_missing_texels": int(gm.sum())})
    return channels


def cube_coordinates(face, u, v):
    x, y, z = face_direction(face, u, v, "plain")
    radius = np.sqrt(x*x + y*y + z*z)
    gc = np.arcsin(np.clip(z / radius, -1, 1))
    gd = np.arctan2(np.sin(gc), K * np.cos(gc))
    # Map geocentric cube solid angle to the extractor's geodetic-latitude
    # sphere area convention. This is not ellipsoid surface area.
    jacobian = K**2 / (K**2 * np.cos(gc)**2 + np.sin(gc)**2)**1.5
    return np.rad2deg(np.arctan2(y, x)), np.rad2deg(gd), jacobian


def _relative(value, reference):
    return abs(value - reference) / max(abs(reference), 1e-300)


def renderer_density(face):
    """Texel-center loadCube.ts + empty-support gate, normalized back to /m².

    three.js DataUtils first converts to float32 then truncates the mantissa
    to half precision (toward zero), unlike numpy's nearest-even conversion.
    No ray integration, texture filtering, lighting, or display gain is modeled.
    """
    decoded = face.astype(float)
    log_tau32 = (decoded[..., 0] + np.log(OPTICAL_AREA)).astype(np.float32)
    rounded = log_tau32.astype(np.float16)
    overshot = np.abs(rounded.astype(float)) > np.abs(log_tau32.astype(float))
    truncated = np.where(overshot, np.nextafter(rounded, np.float16(0)), rounded).astype(float)
    return np.where(decoded[..., 3] > 0, np.exp(truncated) / OPTICAL_AREA, 0)


def build_group_support(groups, *, face_size):
    """Independent composition coverage, sampled on the same geographic grid.

    The v2 density cube's G is its numerator. Decode G / this support only
    where support > 0; a zero mask means unknown, not zero composition.
    """
    coordinates = (np.arange(face_size) + .5) / face_size * 2 - 1
    u, v = np.meshgrid(coordinates, coordinates)
    return [sample_grid(groups, *cube_coordinates(face, u, v)[:2])[1].astype("<f2")
            for face in range(6)]


def build_cube(density, cv, groups, *, face_size=512, integral_rtol=.01,
               groups_policy="require-complete"):
    """Bilinear reprojection with measured conservation gates, not a claim of
    exactly conservative cube reprojection. Never rescale to a published total.
    """
    if groups_policy not in ("require-complete", "separate-coverage"):
        raise ValueError("Unknown groups policy")
    separate_groups = groups_policy == "separate-coverage"
    if (not isinstance(face_size, int) or face_size < 1
            or 6 * face_size**2 * (10 if separate_groups else 8) + 1_000_000 > MAX_PRODUCT_BYTES):
        raise ValueError("Face size exceeds bounded product budget")
    if not np.isfinite(integral_rtol) or not 0 < integral_rtol <= .01:
        raise ValueError("Integral tolerance must be positive and at most 1 percent")
    coordinates = (np.arange(face_size) + .5) / face_size * 2 - 1
    u, v = np.meshgrid(coordinates, coordinates)
    area = texel_solid_angles(face_size, "plain") * R_MEAN**2
    faces = []
    totals = dict(preencode_geodetic_integral=0., decoded_geodetic_integral=0.,
                  decoded_supported_integral=0., renderer_equivalent_integral=0.,
                  renderer_geocentric_integral=0.,
                  decoded_ungated_integral=0.)
    holes = dict(cv_missing_texels=0, groups_missing_texels=0,
                 missing_area_m2=0., missing_density_integral=0.)
    diagnostics = dict(cv_support_difference_area_m2=0., groups_support_difference_area_m2=0.,
                       coverage_quantization_lost_texels=0, known_zero_density_texels=0,
                       groups_coverage_quantization_lost_texels=0)
    group_checksums = {}
    regional = {f"{south}:{south+30}": dict(source=0., cube=0., coastal_cube=0.)
                for south in range(-90, 90, 30)}
    for face in range(6):
        lon, lat, jac = cube_coordinates(face, u, v)
        fields, bm, gm, bs, gs = _channels(density, cv, groups, lon, lat,
                                         separate_groups=separate_groups)
        r, g, b, a = np.moveaxis(fields, -1, 0)
        known = a > 0
        holes["cv_missing_texels"] += int(bm.sum())
        holes["groups_missing_texels"] += int(gm.sum())
        holes["missing_area_m2"] += float(np.sum(area * jac * (bm | gm)))
        holes["missing_density_integral"] += float(np.sum(r * area * jac * (bm | gm)))
        diagnostics["cv_support_difference_area_m2"] += float(np.sum(np.abs(a - bs) * area * jac))
        diagnostics["groups_support_difference_area_m2"] += float(np.sum(np.abs(a - gs) * area * jac))
        diagnostics["known_zero_density_texels"] += int((known & (r == 0)).sum())
        if separate_groups:
            group_mask = gs.astype("<f2")
            diagnostics["groups_coverage_quantization_lost_texels"] += int(((gs > 0) & (group_mask == 0)).sum())
            group_checksums[f"groups_support_{face}.bin"] = hashlib.sha256(group_mask.tobytes()).hexdigest()
        # Finite floor represents supported zero to negligible absolute error;
        # no-data R=0 is a finite sentinel suppressed by A, never a density1.
        encoded = fields.copy()
        encoded[..., 0] = np.where(known, np.log(np.maximum(r, 1e-30)), 0)
        with np.errstate(over="ignore"):
            packed = encoded.astype("<f2")
        if not np.isfinite(packed).all():
            raise ValueError("Channel is not representable as finite RGBA16Float")
        decoded = packed.astype(float)
        valid = decoded[..., 3] > 0
        diagnostics["coverage_quantization_lost_texels"] += int((known & ~valid).sum())
        decoded_density = np.exp(decoded[..., 0])
        supported_density = np.where(valid, decoded_density, 0)  # NEVER multiply by A
        totals["preencode_geodetic_integral"] += float(np.sum(r * area * jac))
        totals["decoded_geodetic_integral"] += float(np.sum(supported_density * area * jac))
        totals["decoded_supported_integral"] += float(np.sum(decoded_density[valid] * area[valid]))
        rendered = renderer_density(packed)
        # The source uses geographic sine-latitude area, so its conservation
        # comparison must use the same Jacobian as the preencode/decoded gates.
        # Raw renderer-sphere area remains useful, but is a separate diagnostic.
        totals["renderer_equivalent_integral"] += float(np.sum(rendered * area * jac))
        totals["renderer_geocentric_integral"] += float(np.sum(rendered * area))
        totals["decoded_ungated_integral"] += float(np.sum(decoded_density * area))
        for south in range(-90, 90, 30):
            mask = (lat >= south) & (lat < south + 30)
            record = regional[f"{south}:{south+30}"]
            record["cube"] += float(np.sum(supported_density * area * jac * mask))
            record["coastal_cube"] += float(np.sum(supported_density * area * jac * mask * (a > 0) * (a < 1)))
        faces.append(packed)
    if holes["cv_missing_texels"] or (holes["groups_missing_texels"] and not separate_groups):
        raise SupportError(holes)
    source = density.integral()
    for south in range(-90, 90, 30):
        top = np.minimum(density.lat_edges[:-1], south + 30)
        bottom = np.maximum(density.lat_edges[1:], south)
        row_area = np.maximum(0, np.sin(np.deg2rad(top)) - np.sin(np.deg2rad(bottom)))
        area_source = row_area[:, None] * np.deg2rad(np.diff(density.lon_edges))[None, :] * R_MEAN**2
        record = regional[f"{south}:{south+30}"]
        record["source"] = float(np.sum(density.values * area_source))
        record["relative_error"] = _relative(record["cube"], record["source"])
    errors = {key: _relative(totals[key], source) for key in
              ("preencode_geodetic_integral", "decoded_geodetic_integral", "renderer_equivalent_integral")}
    passed = (all(error <= integral_rtol for error in errors.values())
              and diagnostics["coverage_quantization_lost_texels"] == 0
              and diagnostics["groups_coverage_quantization_lost_texels"] == 0)
    meta = dict(schema=SCHEMA, projection="plain", field="density", face_size=face_size,
                face_order=FACE_NAMES, bytes_per_face=face_size**2 * 8,
                total_bytes=6 * face_size**2 * 8, publication_status="candidate-not-promoted",
                format="RGBA16Float little endian; row-major py,px,channel; flipY=false",
                channels={"R": "ln(all-cell area-mean nematodes/m^2), supported zero floor 1e-30",
                          "G": "coarse functional fraction (omnivores+predators)/sum(five groups), normalized support",
                          "B": "support-weighted mean native pixel bootstrap CV, not CV of aggregate count",
                          "A": "valid-model area fraction; NOT ocean mask; empty gate only, never multiply R again"},
                georeferencing="independent channel geographic cell edges; cube geocentric to WGS84 geodetic latitude",
                source={"density": density.provenance, "cv": cv.provenance, "groups": groups.provenance},
                geography={name: {"west": float(field.lon_edges[0]), "east": float(field.lon_edges[-1]),
                                   "north": float(field.lat_edges[0]), "south": float(field.lat_edges[-1]),
                                   "shape": list(field.values.shape)}
                           for name, field in (("density", density), ("cv", cv), ("groups", groups))},
                validation=dict(source_integral=source, **totals, relative_errors=errors,
                                tolerance=integral_rtol, passed=passed, support=holes,
                                diagnostics=diagnostics, latitude_bands=regional,
                                area_convention="R=6371000 sphere; all source conservation gates include the geocentric-to-geodetic Jacobian; renderer_geocentric_integral uses raw solid angle and is not gated against the geodetic source",
                                limitations="texel-center quadrature; not a filtered/raymarched GPU image integral; regional/coastal diagnostics require independent review",
                                renderer_conversion={"optical_area": OPTICAL_AREA, "rounding": "three.js float32 then float16 toward zero; density recovered by dividing exp(logTau) by optical area"},
                                published_total_used_as_target=False))
    if separate_groups:
        meta["schema"] = "soil-cube-candidate-v2"
        meta["channels"]["G"] = "coarse composition numerator: fraction times independent groups_support; unknown when groups_support=0; never normalize with A"
        meta["group_support"] = dict(format="R16Float", bytes_per_face=face_size**2*2,
                                     total_bytes=6*face_size**2*2,
                                     face_order=FACE_NAMES, sha256=group_checksums,
                                     decode="filter G and groups_support at identical directions/LOD, divide only when support > 0; otherwise unavailable",
                                     source_fidelity="legacy sparse 512x256 groups; not upgraded by this cube")
        meta["groups_policy"] = groups_policy
    return faces, meta


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_conservative(meta_path, units):
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    if (meta.get("schema") != "soil-conservative-v1" or meta.get("complete") is not True
            or meta.get("publishable") is not True or meta.get("row_coverage_actual") != 1):
        raise ValueError("Only complete, non-pilot conservative inputs are accepted")
    if meta.get("provenance", {}).get("units") != units:
        raise ValueError(f"Expected units {units}")
    filename = meta["grid_file"]
    if Path(filename).name != filename:
        raise ValueError("Grid filename must be adjacent to metadata")
    archive = meta_path.parent / filename
    if _sha(archive) != meta["grid_sha256"]:
        raise ValueError("Conservative grid checksum mismatch")
    with np.load(archive, allow_pickle=False) as data:
        grid = Grid(data["density"], data["valid_model_fraction"], data["lon_edges"],
                    data["lat_edges"], {"kind": "conservative", "metadata_path": str(meta_path.resolve()),
                                       "metadata_sha256": _sha(meta_path), "extraction": meta})
        processed = data["processed_fraction"]
        if (processed.shape != grid.values.shape or not np.isfinite(processed).all()
                or np.any(processed < grid.support - 1e-12) or np.any(processed > 1 + 1e-12)):
            raise ValueError("Invalid processed source footprint")
    integral = grid.integral()
    for key in ("native_integral", "output_integral"):
        if not np.isfinite(meta[key]) or _relative(integral, meta[key]) > 1e-9:
            raise ValueError(f"Source integral disagreement: {key}")
    return grid


def load_legacy(directory, tag, extent):
    """Historical virtual registration is explicit; never upgraded to full rows.

    These arrays have lost exact native sample support. Their historic nominal
    extent is supplied by the caller, recorded as an approximation, and NOT
    inferred from the density product or array dimensions.
    """
    if tag not in (*GROUP_NAMES, "bootstrap_coefvar"):
        raise ValueError("Unknown legacy channel")
    directory = Path(directory)
    meta_path = directory / f"{tag}_meta_512x256.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("failed_rows") != [] or not 0 < meta["row_coverage_actual"] <= 1:
        raise ValueError("Legacy extraction has failed/unknown rows")
    values_path = directory / f"{tag}_density_512x256.npy"
    support_path = directory / f"{tag}_landfrac_512x256.npy"
    values = np.load(values_path, allow_pickle=False)
    support = np.load(support_path, allow_pickle=False)
    if values.shape != (meta["out_height"], meta["out_width"]):
        raise ValueError("Legacy metadata shape mismatch")
    west, east, north, south = extent
    return Grid(values, support, np.linspace(west, east, values.shape[1] + 1),
                np.linspace(north, south, values.shape[0] + 1),
                {"kind": "legacy-sparse", "extraction": meta,
                 "citation": "https://doi.org/10.3929/ethz-b-" + ("000354035" if tag == "bootstrap_coefvar" else "000354394"),
                 "source_units": "dimensionless CV" if tag == "bootstrap_coefvar" else "nematodes/100g dry soil",
                 "rights_record": "http://rightsstatements.org/page/InC-NC/1.0/",
                 "rights_verification": "THIRD_PARTY_NOTICES.md, 2026-09-19; no intended-use classification by this builder",
                 "sha256": {p.name: _sha(p) for p in (meta_path, values_path, support_path)},
                 "nominal_extent": list(extent), "registration": "explicit historical virtual grid; approximate native registration",
                 "limitation": "sparse rows; not all-native-row data, not upgraded by larger output cube"})


def load_legacy_groups(directory, extent):
    groups = {name: load_legacy(directory, name, extent) for name in GROUP_NAMES}
    reference = groups[GROUP_NAMES[0]]
    for field in groups.values():
        if not all(np.array_equal(getattr(field, name), getattr(reference, name))
                   for name in ("support", "lon_edges", "lat_edges")):
            raise ValueError("Legacy functional groups have unequal support/georeference")
    denominator = sum(field.values for field in groups.values())
    numerator = groups["omnivores"].values + groups["predators"].values
    fraction = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    support = np.where(denominator > 0, reference.support, 0)
    return Grid(fraction * support, support, reference.lon_edges, reference.lat_edges,
                {"kind": "legacy-functional-groups", "groups": {k: v.provenance for k, v in groups.items()},
                 "formula": "(omnivores+predators)/sum(five independently modeled groups), then support-weighted interpolation",
                 "limitation": "coarse sparse-row source support; zero denominator is missing composition, not fraction zero"})


def write_candidate(destination, faces, meta, *, group_support_faces=None):
    destination = Path(destination).resolve()
    if "public" in destination.parts:
        raise ValueError("Builder cannot write public assets; independent promotion required")
    if destination.exists():
        raise FileExistsError("Choose a new versioned candidate directory")
    if not meta["validation"]["passed"]:
        raise ValueError("Candidate failed validation gate")
    shape = (meta["face_size"], meta["face_size"], 4)
    if any(face.shape != shape or face.dtype != np.dtype("<f2") for face in faces):
        raise ValueError("Face shape/size or dtype disagrees with metadata")
    support_files = {}
    if meta.get("schema") == "soil-cube-candidate-v2":
        if group_support_faces is None or len(group_support_faces) != 6:
            raise ValueError("Separate group support files are required")
        for index, mask in enumerate(group_support_faces):
            name = f"groups_support_{index}.bin"
            if (mask.shape != shape[:2] or mask.dtype != np.dtype("<f2")
                    or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1))
                    or hashlib.sha256(mask.tobytes()).hexdigest() != meta["group_support"]["sha256"][name]):
                raise ValueError("Invalid group support shape, values or checksum")
            support_files[name] = mask
    elif group_support_faces is not None:
        raise ValueError("Unexpected group support files for legacy contract")
    payload_bytes = sum(face.nbytes for face in faces) + sum(mask.nbytes for mask in support_files.values())
    if len(faces) != 6 or payload_bytes + 1_000_000 > MAX_PRODUCT_BYTES:
        raise ValueError("Product exceeds bounded output budget")
    json.dumps(meta, allow_nan=False)  # reject invalid metadata before any write
    destination.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for index, face in enumerate(faces):
        path = destination / f"face_{index}.bin"
        with path.open("xb") as stream:
            stream.write(face.astype("<f2").tobytes())
        checksums[path.name] = _sha(path)
    for name, mask in support_files.items():
        with (destination / name).open("xb") as stream:
            stream.write(mask.tobytes())
    meta = {**meta, "face_sha256": checksums}
    # Completion marker written last. Interrupted directories are never reused.
    with (destination / "cube_meta.json").open("x") as stream:
        stream.write(json.dumps(meta, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-meta", type=Path, required=True)
    cv = parser.add_mutually_exclusive_group(required=True)
    cv.add_argument("--cv-meta", type=Path, help="conservative bootstrap product with units dimensionless")
    cv.add_argument("--legacy-cv-dir", type=Path)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--groups-policy", choices=("require-complete", "separate-coverage"),
                        default="require-complete",
                        help="separate-coverage stores G numerator plus independently valid support files; never invents missing composition")
    parser.add_argument("--legacy-extent", type=float, nargs=4, metavar=("WEST", "EAST", "NORTH", "SOUTH"), required=True)
    parser.add_argument("--legacy-cv-extent", type=float, nargs=4, metavar=("WEST", "EAST", "NORTH", "SOUTH"))
    parser.add_argument("--face-size", type=int, default=512)
    parser.add_argument("--integral-rtol", type=float, default=.01)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists() or "public" in args.out_dir.resolve().parts:
        parser.error("Choose a new non-public versioned output directory")
    density = load_conservative(args.density_meta, "nematodes/m^2")
    groups = load_legacy_groups(args.groups_dir, args.legacy_extent)
    if args.cv_meta:
        bootstrap = load_conservative(args.cv_meta, "dimensionless")
    else:
        if args.legacy_cv_extent is None:
            parser.error("Legacy CV requires its own explicit --legacy-cv-extent")
        bootstrap = load_legacy(args.legacy_cv_dir, "bootstrap_coefvar", args.legacy_cv_extent)
    try:
        faces, meta = build_cube(density, bootstrap, groups, face_size=args.face_size,
                                integral_rtol=args.integral_rtol, groups_policy=args.groups_policy)
    except SupportError as error:
        print(json.dumps({"status": "rejected", "support": error.diagnostics}, indent=2))
        return 2
    print(json.dumps(meta["validation"], indent=2, allow_nan=False))
    if not meta["validation"]["passed"]:
        return 2
    group_support = (build_group_support(groups, face_size=args.face_size)
                     if args.groups_policy == "separate-coverage" else None)
    write_candidate(args.out_dir, faces, meta, group_support_faces=group_support)
    return 0


if __name__ == "__main__":
    sys.exit(main())
