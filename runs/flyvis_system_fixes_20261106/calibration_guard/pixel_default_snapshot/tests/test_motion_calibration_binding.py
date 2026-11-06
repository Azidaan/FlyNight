"""Preserve empirical decoder calibration while fixing pixel coordinates."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracking.flyvis_motion import (
    GEOMETRY_VERSION, GEOMETRY_VERSIONS, SYNTHETIC_FLOW_GAIN,
    FlyVisMotionEstimator, TileLayout, fixed_tile_bounds, resolve_flow_gain,
)


def bare_model(version, bounds, gain):
    model = FlyVisMotionEstimator.__new__(FlyVisMotionEstimator)
    model.eye_width = model.eye_height = 391
    model.kernel_area = 169
    model.flow_gain = gain
    model.geometry_version = version
    model._bounds = bounds
    model._local_centers_yx = np.array([[0, 0], [-65, 65], [65, -65]], np.float32)
    return model


@pytest.mark.parametrize("mode", ["full", "tiled"])
@pytest.mark.parametrize("size", [(1280, 720), (720, 1280), (391, 391)])
@pytest.mark.parametrize("gain", [0.0001, 1., 38.84157, SYNTHETIC_FLOW_GAIN, 1e6])
@pytest.mark.parametrize("aggregation", ["mean", "last", "sum"])
def test_stable_vectors_exactly_preserve_legacy_calibration(mode, size, gain, aggregation):
    bounds = [(0, 0, *size)] if mode == "full" else fixed_tile_bounds(*size, TileLayout())
    decoded = torch.from_numpy(np.random.default_rng(163).normal(size=(len(bounds), 7, 2, 3)).astype(np.float32))
    old = bare_model("legacy_v1", bounds, gain)
    stable = bare_model(GEOMETRY_VERSION, bounds, gain)
    old.last_decoded = stable.last_decoded = decoded
    a, b = old.field_from_last(aggregation), stable.field_from_last(aggregation)
    np.testing.assert_array_equal(a.vectors_xy, b.vectors_xy)
    np.testing.assert_array_equal(np.median(a.vectors_xy, axis=0), np.median(b.vectors_xy, axis=0))
    research = bare_model("resize_v2", bounds, gain)
    research.last_decoded = decoded
    np.testing.assert_array_equal(b.points_xy, research.field_from_last(aggregation).points_xy)
    assert not np.array_equal(a.points_xy, b.points_xy)


def test_stable_default_and_explicit_research_gain():
    assert GEOMETRY_VERSION == "pixel_centers_v2"
    assert GEOMETRY_VERSIONS == ("legacy_v1", "pixel_centers_v2", "resize_v2")
    for version in ("legacy_v1", "pixel_centers_v2"):
        assert resolve_flow_gain(None, version) == SYNTHETIC_FLOW_GAIN
        assert resolve_flow_gain(None, version, empirical_default=38.84157) == 38.84157
    with pytest.raises(ValueError, match="resize_v2"):
        resolve_flow_gain(None, "resize_v2")
    assert resolve_flow_gain(1., "resize_v2") == 1.


def test_research_implicit_gain_fails_before_network_load(monkeypatch):
    monkeypatch.setattr("tracking.flyvis_motion.activate_flyvis", lambda **kw: pytest.fail("runtime was loaded"))
    with pytest.raises(ValueError, match="resize_v2"):
        FlyVisMotionEstimator(geometry_version="resize_v2")
    from scripts.fly_finetune_k1 import Fly
    with pytest.raises(ValueError, match="resize_v2"):
        Fly(30., geometry_version="resize_v2")


@pytest.mark.parametrize("version", GEOMETRY_VERSIONS)
def test_cache_gain_binding_preserves_files_and_requires_explicit_research(tmp_path, version):
    from scripts.run_bdd_video_rate_experiment import load_motion_cache
    path = tmp_path / "flyvis_tiled.npz"
    points = np.array([[1., 2.]], np.float32)
    vectors = np.array([[[2., 3.]]], np.float32)
    np.savez(path, points=points, vectors=vectors)
    meta = path.with_suffix(".json")
    meta.write_text(json.dumps(dict(source="flyvis_tiled", geometry_version=version)), encoding="utf-8")
    before = path.read_bytes(), meta.read_bytes()
    if version == "resize_v2":
        with pytest.raises(RuntimeError, match="прежней калибровкой"):
            load_motion_cache(path, 38.84157)
        p, v = load_motion_cache(path, 3., allow_unvalidated_resize=True)
    else:
        p, v = load_motion_cache(path, 3.)
    np.testing.assert_array_equal(p, points)
    np.testing.assert_array_equal(v, vectors * 3.)
    assert before == (path.read_bytes(), meta.read_bytes())


def test_legacy_unversioned_cache_remains_readable(tmp_path):
    from scripts.run_bdd_video_rate_experiment import load_motion_cache
    path = tmp_path / "flyvis_full.npz"
    np.savez(path, points=np.zeros((1, 2)), vectors=np.ones((1, 1, 2)))
    np.testing.assert_array_equal(load_motion_cache(path, 102.596)[1], np.full((1, 1, 2), 102.596))


def test_classical_new_version_aliases_have_identical_correct_geometry():
    from tracking.classical_motion import OpenCVMotionEstimator
    flow = np.ones((25, 25, 2), np.float32)
    a = OpenCVMotionEstimator("farneback", processing_scale=.25, geometry_version="pixel_centers_v2")._sample(flow, 101, 99)
    b = OpenCVMotionEstimator("farneback", processing_scale=.25, geometry_version="resize_v2")._sample(flow, 101, 99)
    np.testing.assert_array_equal(a.points_xy, b.points_xy)
    np.testing.assert_array_equal(a.vectors_xy, b.vectors_xy)
    np.testing.assert_array_equal(a.vectors_xy, np.full(a.vectors_xy.shape, 4., np.float32))


def test_historical_k1_checkpoint_cannot_be_relabelled_stable():
    from scripts.fly_finetune_k1 import validate_arm_geometry
    with pytest.raises(RuntimeError, match="legacy_v1"):
        validate_arm_geometry({"scale": {"log_gain": 0.}}, GEOMETRY_VERSION)


@pytest.mark.parametrize("module", ["scripts.run_flyvis_tracker", "scripts.evaluate_motion_sources"])
def test_research_cli_implicit_gain_fails_before_output(tmp_path, monkeypatch, module):
    import importlib
    script = importlib.import_module(module)
    output = tmp_path / "new"
    args = SimpleNamespace(geometry_version="resize_v2", flow_gain=None, mode="full", output=output)
    monkeypatch.setattr(script, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="resize_v2"):
        script.main()
    assert not output.exists()


def test_video_rate_research_tracker_fails_before_data_or_output(tmp_path, monkeypatch):
    import scripts.run_bdd_video_rate_experiment as script
    args = SimpleNamespace(flyvis_geometry_version="resize_v2", motion_only=False,
                           trackers=["flyvis_tiled_calibrated"], output=tmp_path / "new")
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(script, "resolve_paths", lambda _: pytest.fail("data preparation started"))
    with pytest.raises(ValueError, match="прежние эмпирические gains"):
        script.main()
    assert not args.output.exists()


def test_k1_research_cli_fails_before_output(tmp_path, monkeypatch):
    from scripts.fly_finetune_k1 import main
    output = tmp_path / "new"
    monkeypatch.setattr("sys.argv", ["k1", "--geometry-version", "resize_v2", "--out", str(output)])
    with pytest.raises(ValueError, match="resize_v2"):
        main()
    assert not output.exists()
