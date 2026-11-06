"""Inverse geometry must match OpenCV fx/fy, including dimension rounding."""
import json

import cv2
import numpy as np
import pytest

from tracking.classical_motion import OpenCVMotionEstimator


@pytest.mark.parametrize("width,height,scale", [
    (1280, 720, .25), (1281, 721, .25), (101, 99, .25),
    (99, 101, .5), (64, 64, 1.), (32, 32, .99),
])
def test_sample_locations_follow_actual_cv2_coordinate_ramps(width, height, scale):
    estimator = OpenCVMotionEstimator("farneback", processing_scale=scale)
    yy, xx = np.indices((height, width), dtype=np.float32)
    xmap = cv2.resize(xx, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ymap = cv2.resize(yy, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ph, pw = xmap.shape
    gx, gy = np.meshgrid(np.arange(4, pw, 8), np.arange(4, ph, 8))
    expected = np.column_stack([xmap[gy, gx].ravel(), ymap[gy, gx].ravel()])
    field = estimator._sample(np.zeros((ph, pw, 2), np.float32), width, height)
    np.testing.assert_allclose(field.points_xy, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("width,height,scale", [(101, 99, .25), (1281, 721, .25), (32, 32, .99)])
def test_displacement_uses_inverse_configured_scale_or_whole_image_identity(width, height, scale):
    estimator = OpenCVMotionEstimator("farneback", processing_scale=scale)
    shape = estimator._gray(np.zeros((height, width), np.uint8)).shape
    raw = np.ones((*shape, 2), np.float32)
    field = estimator._sample(raw, width, height)
    inverse = 1. if shape == (height, width) else 1. / scale
    np.testing.assert_array_equal(field.vectors_xy, np.full_like(field.vectors_xy, inverse))
    np.testing.assert_array_equal(raw, np.ones_like(raw))


def test_explicit_legacy_sampling_preserves_previous_coordinate_and_scale_formula():
    estimator = OpenCVMotionEstimator("farneback", processing_scale=.25, geometry_version="legacy_v1")
    raw = np.ones((25, 25, 2), np.float32)
    field = estimator._sample(raw, 101, 99)
    gx, gy = np.meshgrid(np.arange(4, 25, 8), np.arange(4, 25, 8))
    expected = np.column_stack([gx.ravel() * (101 / 25), gy.ravel() * (99 / 25)]).astype(np.float32)
    np.testing.assert_array_equal(field.points_xy, expected)
    np.testing.assert_array_equal(field.vectors_xy, np.tile(np.array([101 / 25, 99 / 25], np.float32), (len(expected), 1)))


@pytest.mark.parametrize("force", [False, True])
def test_old_classical_cache_requires_explicit_legacy_and_remains_unchanged(tmp_path, force):
    from scripts.run_bdd_video_rate_experiment import build_motion_cache
    data, meta = tmp_path / "dis_quarter.npz", tmp_path / "dis_quarter.json"
    data.write_bytes(b"saved historical field")
    meta.write_text(json.dumps(dict(source="dis_quarter", fps=30.)), encoding="utf-8")
    before = (data.read_bytes(), meta.read_bytes())
    with pytest.raises(RuntimeError, match="legacy_v1"):
        build_motion_cache("dis_quarter", tmp_path / "absent.mov", tmp_path,
                           fps=30., max_dt=.02, force=force)
    old = build_motion_cache("dis_quarter", tmp_path / "absent.mov", tmp_path,
                             fps=30., max_dt=.02, force=False, geometry_version="legacy_v1")
    assert old["geometry_version"] == "legacy_v1"
    assert before == (data.read_bytes(), meta.read_bytes())


def test_classical_factory_passes_geometry_version():
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    assert make_motion_estimator("dis_quarter", 30., .02).geometry_version == "resize_v2"
    assert make_motion_estimator("dis_quarter", 30., .02, geometry_version="legacy_v1").geometry_version == "legacy_v1"


def test_classical_tracker_directory_rejects_unversioned_geometry_by_default(tmp_path):
    from scripts.run_flyvis_tracker import validate_output_protocol
    meta = dict(mode="dis", classical_scale=.25, video_ids=["v"])
    (tmp_path / "_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy_v1"):
        validate_output_protocol(tmp_path, dict(meta, geometry_version="resize_v2"))
    validate_output_protocol(tmp_path, dict(meta, geometry_version="legacy_v1"))


@pytest.mark.parametrize("method", ["farneback", "dis"])
def test_native_flow_recovers_known_translation_in_original_pixels(method):
    texture = np.random.default_rng(71).integers(0, 256, (96, 128), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.)
    first = cv2.resize(texture, (512, 384), interpolation=cv2.INTER_NEAREST)
    second = cv2.warpAffine(first, np.array([[1, 0, 8], [0, 1, 4]], np.float32),
                            (512, 384), borderMode=cv2.BORDER_REFLECT)
    estimator = OpenCVMotionEstimator(method, processing_scale=.25)
    estimator.observe(first)
    field = estimator.observe(second)
    interior = np.all((field.points_xy >= [64, 64]) & (field.points_xy <= [448, 320]), axis=1)
    np.testing.assert_allclose(np.median(field.vectors_xy[interior], axis=0), [8, 4], atol=.8, rtol=0)


def test_train_cache_preflight_preserves_selection_before_any_model_load(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.build_train_caches as script
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(videos={"v": dict(alignment_ok=True, split="train")})), encoding="utf-8")
    output = tmp_path / "output"
    motion = output / "v" / "motion"
    motion.mkdir(parents=True)
    (motion / "dis_quarter.npz").write_bytes(b"saved motion")
    (motion / "dis_quarter.json").write_text('{"fps": 30}', encoding="utf-8")
    selection = output / "selection.json"
    selection.write_bytes(b"historical selection")
    args = SimpleNamespace(manifest=manifest, output=output, per_split="train=1,dev=0,test=0",
                           geometry_version="resize_v2")
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(script, "build_detection_cache", lambda *a, **k: pytest.fail("loaded detector before preflight"))
    with pytest.raises(RuntimeError, match="legacy_v1"):
        script.main()
    assert selection.read_bytes() == b"historical selection"
    assert (motion / "dis_quarter.npz").read_bytes() == b"saved motion"


@pytest.mark.parametrize("width,height", [(1280, 720), (101, 99)])
def test_retina_dis_manual_resize_matches_direct_classical_geometry(width, height):
    from scripts.build_dis_retina import restore_source_field
    direct = OpenCVMotionEstimator("farneback", processing_scale=.25)
    reduced = direct._gray(np.zeros((height, width), np.uint8))
    raw = np.ones((*reduced.shape, 2), np.float32)
    local = OpenCVMotionEstimator("farneback")._sample(raw, reduced.shape[1], reduced.shape[0])
    restored = restore_source_field(local, (height, width), reduced.shape, "resize_v2")
    expected = direct._sample(raw, width, height)
    np.testing.assert_array_equal(restored.points_xy, expected.points_xy)
    np.testing.assert_array_equal(restored.vectors_xy, expected.vectors_xy)


def test_retina_dis_legacy_cache_not_promoted_or_modified(tmp_path):
    from scripts.build_dis_retina import check_geometry, dis_on_retina
    path = tmp_path / "motion" / "v" / "dis_quarter.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"legacy control")
    with pytest.raises(RuntimeError, match="legacy_v1"):
        dis_on_retina(("v", tmp_path / "absent.mov", tmp_path, "resize_v2"))
    assert dis_on_retina(("v", tmp_path / "absent.mov", tmp_path, "legacy_v1")) == "v"
    check_geometry(path, "legacy_v1")
    assert path.read_bytes() == b"legacy control" and not path.with_suffix(".json").exists()


def test_retina_control_refuses_mismatched_raw_dis_before_rewriting(tmp_path):
    from scripts.build_dis_retina import rewrite
    original = tmp_path / "raw.npz"
    original.write_bytes(b"legacy raw DIS")
    with pytest.raises(RuntimeError, match="legacy_v1"):
        rewrite(("v", tmp_path / "absent_features.npz", original, tmp_path / "new_output", "resize_v2"))
    assert not (tmp_path / "new_output").exists()
