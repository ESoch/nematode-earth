"""Numeric product-contract tests; synthetic fixtures only, no network/GPU."""
import hashlib
import importlib
import importlib.util
import json
import math

import numpy as np
import pytest


@pytest.fixture
def product():
    assert importlib.util.find_spec("soil_product") is not None, "new soil-product builder missing"
    return importlib.import_module("soil_product")


def grid(product, values, support=None, lon=None, lat=None):
    values = np.asarray(values, dtype=float)
    h, w = values.shape
    return product.Grid(
        values, np.ones_like(values) if support is None else np.asarray(support, dtype=float),
        np.linspace(-180, 180, w + 1) if lon is None else np.asarray(lon, dtype=float),
        np.linspace(90, -90, h + 1) if lat is None else np.asarray(lat, dtype=float),
        {"fixture": True},
    )


def test_two_cell_coast_preserves_all_cell_mean_and_normalizes_cv(product):
    # Equal-area cells: one density100/CV.4/G.2, one missing. At their
    # boundary R=50,A=.5 but B=.4 and G=.2, NOT .2 and .1.
    r = grid(product, [[100, 0]], [[1, 0]])
    b = grid(product, [[.4, 0]], [[1, 0]])
    g = grid(product, [[.2, 0]], [[1, 0]])
    got = product.sample_channels(r, b, g, np.array([0.]), np.array([0.]))
    np.testing.assert_allclose(got, [[50, .2, .4, .5]])


def test_unequal_shifted_cv_grid_uses_own_registration_and_support(product):
    r = grid(product, [[100, 100]])
    # Centers -80,20,120. Query20 lies on CV.8, independent of R's grid.
    b = grid(product, [[.1, .8, .3]], lon=[-130, -30, 70, 170])
    g = grid(product, [[.2]])
    got = product.sample_channels(r, b, g, np.array([20.]), np.array([0.]))
    np.testing.assert_allclose(got, [[100, .2, .8, 1]])


def test_support_weighted_cv_is_ratio_of_interpolated_numerator_and_support(product):
    r = grid(product, [[50, 100]], [[.5, 1]])
    b = grid(product, [[.1, .8]], [[.5, 1]])  # native CV means .2,.8
    g = grid(product, [[.1, .2]], [[.5, 1]])
    got = product.sample_channels(r, b, g, np.array([0.]), np.array([0.]))
    np.testing.assert_allclose(got, [[75, .2, .6, .75]])


def test_cv_unequal_latitude_cells_keep_their_own_centers(product):
    r = grid(product, [[100]])
    b = grid(product, [[.1], [.8], [.3]], lat=[80, 0, -10, -80])
    g = grid(product, [[.2]])
    got = product.sample_channels(r, b, g, np.array([0.]), np.array([-5.]))
    np.testing.assert_allclose(got, [[100, .2, .8, 1]])


@pytest.mark.parametrize("missing", ["cv", "groups"])
def test_absent_or_unsupported_auxiliary_channel_fails(product, missing):
    r = grid(product, [[100]])
    good = grid(product, [[.2]])
    for absent in (None, grid(product, [[0]], [[0]])):
        b, g = (absent, good) if missing == "cv" else (good, absent)
        with pytest.raises(ValueError, match="support|missing"):
            product.sample_channels(r, b, g, np.array([0.]), np.array([0.]))


def test_explicit_extent_masks_poles_without_extrapolating(product):
    r = grid(product, [[100]], lat=[88, -88])
    b = grid(product, [[.4]])
    g = grid(product, [[.2]])
    got = product.sample_channels(r, b, g, np.array([0., 0.]), np.array([89., 0.]))
    np.testing.assert_allclose(got, [[0, 0, 0, 0], [100, .2, .4, 1]])


def test_cv_extent_is_independent_and_cannot_borrow_density_extent(product):
    r = grid(product, [[100]])
    b = grid(product, [[.4]], lat=[50, -50])
    g = grid(product, [[.2]])
    with pytest.raises(ValueError, match="support"):
        product.sample_channels(r, b, g, np.array([0.]), np.array([70.]))


def test_geocentric_direction_converts_to_geodetic_before_sampling(product):
    lon, lat, _ = product.cube_coordinates(0, np.array([[-1.]]), np.array([[0.]]))
    assert lon.item() == pytest.approx(0)
    assert lat.item() == pytest.approx(45.19242321598, abs=1e-9)


def test_encode_decode_and_renderer_integrals_do_not_multiply_coverage_again(product):
    r = grid(product, [[50]], [[.5]])
    b = grid(product, [[.2]], [[.5]])
    g = grid(product, [[.1]], [[.5]])
    faces, meta = product.build_cube(r, b, g, face_size=16)
    expected = 50 * 4 * math.pi * 6371000**2
    checks = meta["validation"]
    assert checks["source_integral"] == pytest.approx(expected)
    assert checks["renderer_equivalent_integral"] == pytest.approx(expected, rel=.002)
    assert checks["decoded_supported_integral"] == pytest.approx(expected, rel=.002)
    # Independently checked against installed three.js DataUtils and E_A.
    assert checks["renderer_geocentric_integral"] == pytest.approx(
        50.02533734598561 * 4 * math.pi * 6371000**2)
    assert checks["passed"]
    for face in faces:
        np.testing.assert_allclose(face.astype(float)[..., 1:4],
                                   np.broadcast_to([.2, .4, .5], (16, 16, 3)), rtol=.001)


def test_nodata_log_sentinel_contributes_no_renderer_mass(product):
    r = grid(product, [[100, 0]], [[1, 0]])
    b = grid(product, [[.2]])
    g = grid(product, [[.1]])
    faces, meta = product.build_cube(r, b, g, face_size=32)
    assert meta["validation"]["passed"]
    # Renderer contract can be independently integrated from the stored bytes.
    total = 0.
    from eac import texel_solid_angles
    for face in faces:
        decoded = face.astype(float)
        total += np.sum(np.where(decoded[..., 3] > 0, np.exp(decoded[..., 0]), 0)
                        * texel_solid_angles(32, "plain")) * 6371000**2
    assert total == pytest.approx(meta["validation"]["decoded_supported_integral"])


def test_renderer_loader_quantization_and_empty_gate(product):
    # three.js 0.185.1 DataUtils.toHalfFloat(logDensity + ln(1.49e-8))
    # yields logTau=-14.109375 for stored logDensity=3.912109375.
    face = np.array([[[3.912109375, .2, .4, .5], [3.912109375, .2, .4, 0]]], dtype="<f2")
    got = product.renderer_density(face)
    np.testing.assert_allclose(got, [[50.02533734598561, 0]], rtol=1e-12)


def test_renderer_conservation_gate_matches_geodetic_source_area(product):
    # Analytic cell means of 1 + 1000*sin(geodetic_latitude)^20. On the
    # geodetic sphere the exact global mean is 1 + 1000/21. Concentrating
    # density near the poles exposes the different geocentric area measure;
    # a spatially constant fixture cannot reliably catch this error.
    latitude_edges = np.linspace(90., -90., 1025)
    sine = np.sin(np.deg2rad(latitude_edges))
    means = 1 + 1000 * (sine[:-1]**21 - sine[1:]**21) / (21 * (sine[:-1] - sine[1:]))
    r = grid(product, means[:, None], lat=latitude_edges)
    b = grid(product, [[.2]])
    g = grid(product, [[.1]])
    _, meta = product.build_cube(r, b, g, face_size=128)
    checks = meta["validation"]
    expected = (1 + 1000/21) * 4 * math.pi * 6371000**2
    assert checks["source_integral"] == pytest.approx(expected, rel=1e-12)
    assert checks["relative_errors"]["renderer_equivalent_integral"] < .01
    assert checks["passed"]
    assert checks["renderer_equivalent_integral"] == pytest.approx(expected, rel=.01)
    # Keep the raw renderer sphere integral, but it must not drive this gate.
    assert abs(checks["renderer_geocentric_integral"] / expected - 1) > .005


def test_bake_support_holes_reject_with_diagnostic_counts(product):
    r = grid(product, [[100]])
    b = grid(product, [[.2]])
    g = grid(product, [[0]], [[0]])
    with pytest.raises(product.SupportError) as error:
        product.build_cube(r, b, g, face_size=8)
    assert error.value.diagnostics["groups_missing_texels"] > 0
    assert error.value.diagnostics["missing_density_integral"] > 0


def test_separate_group_coverage_keeps_density_and_marks_unknown(product, tmp_path):
    r = grid(product, [[50]])
    b = grid(product, [[.2]])
    missing = grid(product, [[0]], [[0]])
    faces, meta = product.build_cube(r, b, missing, face_size=8,
                                    groups_policy="separate-coverage")
    masks = product.build_group_support(missing, face_size=8)
    assert meta["validation"]["passed"]
    assert meta["validation"]["support"]["groups_missing_texels"] == 6*8*8
    for face, mask in zip(faces, masks):
        np.testing.assert_allclose(np.exp(face[..., 0].astype(float)), 50, rtol=.002)
        assert np.all(face[..., 3] == 1)  # composition gaps must not erase R
        assert np.all(mask == 0)  # unknown composition, not a measured zero
    destination = tmp_path / "separate"
    product.write_candidate(destination, faces, meta, group_support_faces=masks)
    saved = json.loads((destination / "cube_meta.json").read_text())
    assert saved["group_support"]["format"] == "R16Float"
    for name, sha in saved["group_support"]["sha256"].items():
        payload = (destination / name).read_bytes()
        assert len(payload) == 8*8*2
        assert hashlib.sha256(payload).hexdigest() == sha


def test_group_numerator_and_own_support_survive_quantization(product):
    r = grid(product, [[100]])
    b = grid(product, [[.2]])
    # Composition .4 on .25 supported area. G must be .1, with a SEPARATE
    # .25 support, never falsely normalized against density A=1.
    g = grid(product, [[.1]], [[.25]])
    faces, meta = product.build_cube(r, b, g, face_size=8,
                                    groups_policy="separate-coverage")
    masks = product.build_group_support(g, face_size=8)
    for face, mask in zip(faces, masks):
        np.testing.assert_allclose(face[..., 1].astype(float), .1, rtol=.001)
        np.testing.assert_allclose(mask, .25)
        np.testing.assert_allclose(face[..., 1].astype(float)/mask, .4, rtol=.001)
    assert meta["schema"] == "soil-cube-candidate-v2"


def test_separate_groups_still_requires_cv_and_complete_support_files(product, tmp_path):
    r = grid(product, [[50]])
    b = grid(product, [[.2]])
    g = grid(product, [[0]], [[0]])
    with pytest.raises(product.SupportError):
        product.build_cube(r, grid(product, [[0]], [[0]]), g,
                           face_size=8, groups_policy="separate-coverage")
    faces, meta = product.build_cube(r, b, g, face_size=8,
                                    groups_policy="separate-coverage")
    with pytest.raises(ValueError, match="group support"):
        product.write_candidate(tmp_path / "no-masks", faces, meta)
    with pytest.raises(ValueError, match="group support"):
        product.write_candidate(tmp_path / "bad-masks", faces, meta,
                                group_support_faces=[np.zeros((4, 4), dtype="<f2")]*6)
    assert not (tmp_path / "no-masks").exists()
    assert not (tmp_path / "bad-masks").exists()


def make_conservative(tmp_path, *, complete=True):
    archive = tmp_path / "density_grid.npz"
    np.savez_compressed(archive, density=[[50.]], valid_model_fraction=[[.5]],
                        processed_fraction=[[1.]], lon_edges=[-180., 180.], lat_edges=[90., -90.])
    mass = 50 * 4 * math.pi * 6371000**2
    meta = {"schema": "soil-conservative-v1", "complete": complete, "publishable": complete,
            "row_coverage_actual": 1., "native_integral": mass, "output_integral": mass,
            "conservation_relative_error": 0., "grid_file": archive.name,
            "grid_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "provenance": {"units": "nematodes/m^2", "citation": "fixture"}}
    path = tmp_path / "density_meta.json"
    path.write_text(json.dumps(meta))
    return path, archive


def test_loader_accepts_extractor_contract_and_rejects_pilot_and_corruption(product, tmp_path):
    path, archive = make_conservative(tmp_path)
    result = product.load_conservative(path, "nematodes/m^2")
    assert result.values.item() == 50
    assert result.support.item() == .5
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        product.load_conservative(path, "nematodes/m^2")
    make_conservative(tmp_path, complete=False)
    with pytest.raises(ValueError, match="complete|pilot"):
        product.load_conservative(path, "nematodes/m^2")


def test_loader_rejects_wrong_units_or_integral(product, tmp_path):
    path, _ = make_conservative(tmp_path)
    with pytest.raises(ValueError, match="units"):
        product.load_conservative(path, "dimensionless")
    meta = json.loads(path.read_text())
    meta["native_integral"] *= 2
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="integral"):
        product.load_conservative(path, "nematodes/m^2")


def test_versioned_output_cannot_overwrite_or_write_public(product, tmp_path):
    r = grid(product, [[50]], [[.5]])
    b = grid(product, [[.2]])
    g = grid(product, [[.1]])
    faces, meta = product.build_cube(r, b, g, face_size=8)
    destination = tmp_path / "candidate-v1"
    product.write_candidate(destination, faces, meta)
    assert len(list(destination.glob("face_*.bin"))) == 6
    assert json.loads((destination / "cube_meta.json").read_text())["publication_status"] == "candidate-not-promoted"
    with pytest.raises(FileExistsError):
        product.write_candidate(destination, faces, meta)
    with pytest.raises(ValueError, match="public"):
        product.write_candidate(tmp_path / "app" / "public" / "new", faces, meta)


def test_failed_conservation_gate_never_writes_faces(product, tmp_path):
    r = grid(product, [[50]])
    b = grid(product, [[.2]])
    g = grid(product, [[.1]])
    faces, meta = product.build_cube(r, b, g, face_size=8, integral_rtol=1e-12)
    assert not meta["validation"]["passed"]  # half-log quantization exceeds this bound
    with pytest.raises(ValueError, match="gate"):
        product.write_candidate(tmp_path / "rejected", faces, meta)
    assert not (tmp_path / "rejected").exists()


def test_invalid_fraction_and_uncovered_positive_numerator_rejected(product):
    with pytest.raises(ValueError):
        grid(product, [[50]], [[1.5]])
    with pytest.raises(ValueError):
        grid(product, [[50]], [[0]])


def write_legacy(tmp_path, tag, value, support=.5):
    values = np.array([[value]], dtype=float)
    np.save(tmp_path / f"{tag}_density_512x256.npy", values)
    np.save(tmp_path / f"{tag}_landfrac_512x256.npy", [[support]])
    (tmp_path / f"{tag}_meta_512x256.json").write_text(json.dumps({
        "failed_rows": [], "row_coverage_actual": .02424, "out_height": 1, "out_width": 1,
        "source_url": f"https://example.test/{tag}",
    }))


def test_legacy_groups_keep_sparse_provenance_and_actual_fraction(product, tmp_path):
    # Independently modeled group totals20,30,40,8,2: structure fraction .1.
    for tag, value in zip(product.GROUP_NAMES, [20, 30, 40, 8, 2]):
        write_legacy(tmp_path, tag, value)
    groups = product.load_legacy_groups(tmp_path, [-180, 180, 88, -88])
    assert groups.values.item() == pytest.approx(.05)  # fraction times own support
    assert groups.support.item() == .5
    for record in groups.provenance["groups"].values():
        assert record["extraction"]["row_coverage_actual"] == .02424
    write_legacy(tmp_path, "predators", 2, support=.25)
    with pytest.raises(ValueError, match="unequal support"):
        product.load_legacy_groups(tmp_path, [-180, 180, 88, -88])


def test_cli_candidate_roundtrip_and_no_overwrite(product, tmp_path):
    path, _ = make_conservative(tmp_path)
    for tag, value in zip(product.GROUP_NAMES, [20, 30, 40, 8, 2]):
        write_legacy(tmp_path, tag, value)
    write_legacy(tmp_path, "bootstrap_coefvar", .2)
    destination = tmp_path / "candidate-v1"
    args = ["--density-meta", str(path), "--groups-dir", str(tmp_path),
            "--legacy-extent", "-180", "180", "90", "-90", "--legacy-cv-dir", str(tmp_path),
            "--legacy-cv-extent", "-180", "180", "90", "-90",
            "--face-size", "8", "--out-dir", str(destination)]
    assert product.main(args) == 0
    meta = json.loads((destination / "cube_meta.json").read_text())
    for name, sha in meta["face_sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == sha
        rgba = np.fromfile(destination / name, dtype="<f2").reshape(8, 8, 4).astype(float)
        assert np.exp(rgba[..., 0]).mean() == pytest.approx(50, rel=.002)
        assert rgba[..., 1].mean() == pytest.approx(.1, rel=.001)
        assert rgba[..., 2].mean() == pytest.approx(.4, rel=.001)
        assert rgba[..., 3].mean() == .5
    with pytest.raises(SystemExit):
        product.main(args)
    write_legacy(tmp_path, "bootstrap_coefvar", 0, support=0)
    args[-1] = str(tmp_path / "unsupported-v2")
    assert product.main(args) == 2
    assert not (tmp_path / "unsupported-v2").exists()


def test_positive_support_underflow_blocks_candidate(product):
    r = grid(product, [[50]], [[1e-10]])
    b = grid(product, [[.2]])
    g = grid(product, [[.1]])
    _, meta = product.build_cube(r, b, g, face_size=8)
    assert not meta["validation"]["passed"]
    assert meta["validation"]["diagnostics"]["coverage_quantization_lost_texels"] == 6*8*8


def test_zero_cv_is_valid_only_when_supported(product):
    r = grid(product, [[100]])
    b = grid(product, [[0]])
    g = grid(product, [[.2]])
    np.testing.assert_allclose(product.sample_channels(r, b, g, np.array([0.]), np.array([0.])),
                               [[100, .2, 0, 1]])


def test_source_zero_bands_report_finite_errors(product):
    r = grid(product, [[100]], lat=[10, -10])
    b = grid(product, [[.2]])
    g = grid(product, [[.1]])
    # Coarse quadrature may reject, but diagnostic JSON must stay serializable.
    _, meta = product.build_cube(r, b, g, face_size=32)
    json.dumps(meta, allow_nan=False)


def test_write_candidate_rejects_wrong_face_shapes(product, tmp_path):
    r = grid(product, [[50]])
    b = grid(product, [[.2]])
    g = grid(product, [[.1]])
    faces, meta = product.build_cube(r, b, g, face_size=8)
    faces[0] = faces[0][:4]
    with pytest.raises(ValueError, match="shape|size"):
        product.write_candidate(tmp_path / "broken", faces, meta)
    assert not (tmp_path / "broken").exists()
