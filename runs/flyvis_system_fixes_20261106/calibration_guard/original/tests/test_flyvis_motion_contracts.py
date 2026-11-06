"""Geometry, configuration and device regressions without CUDA calculations."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tracking.flyvis_motion import (
    FlyVisMotionEstimator, TileLayout, fixed_tile_bounds, integration_plan,
)


def conversion_model(width=1280, height=720, version="resize_v2"):
    model = FlyVisMotionEstimator.__new__(FlyVisMotionEstimator)
    model.eye_width = model.eye_height = 391
    model.kernel_area = 169
    model.flow_gain = 1.0
    model.geometry_version = version
    model._bounds = [(20, 30, 20 + width, 30 + height)]
    model._local_centers_yx = np.array([[0, 0], [-65, 65]], np.float32)
    return model


@pytest.mark.parametrize("width,height", [(720, 720), (1280, 720), (492, 400), (200, 600)])
def test_vectors_invert_anisotropic_resize_without_mutating_decoder(width, height):
    model = conversion_model(width, height)
    # Known displacement of (3,-2) pixels of the rendered square eye, encoded
    # using the retained normalized-flow convention. No learned prediction.
    decoded = torch.tensor([[[3., 3.], [2., 2.]]]) * (169 / 391)
    untouched = decoded.clone()
    field = model._points_and_vectors(decoded)
    np.testing.assert_allclose(field.vectors_xy, np.tile([3 * width / 391, -2 * height / 391], (2, 1)), rtol=1e-6)
    assert torch.equal(decoded, untouched)


def test_square_vectors_keep_previous_scale_exactly():
    current = conversion_model(720, 720)
    legacy = conversion_model(720, 720, "legacy_v1")
    decoded = torch.tensor([[[.15, -.8], [-.2, .7]]])
    assert np.array_equal(current._points_and_vectors(decoded).vectors_xy,
                          legacy._points_and_vectors(decoded).vectors_xy)


def test_receptor_locations_match_actual_area_resize_for_integer_downsampling():
    # Each output pixel averages a 2x3 source block: its center can be read
    # independently by resizing coordinate ramps with the real OpenCV kernel.
    model = conversion_model(782, 1173)
    yy, xx = np.indices((1173, 782), dtype=np.float32)
    xmap = cv2.resize(xx, (391, 391), interpolation=cv2.INTER_AREA)
    ymap = cv2.resize(yy, (391, 391), interpolation=cv2.INTER_AREA)
    indices = (model._local_centers_yx + 195).astype(int)
    expected = np.array([[20 + xmap[y, x], 30 + ymap[y, x]] for y, x in indices])
    np.testing.assert_allclose(model._points_and_vectors(torch.zeros(1, 2, 2)).points_xy,
                               expected, atol=1e-4, rtol=0)


def test_identity_resize_preserves_integer_receptor_positions():
    model = conversion_model(391, 391)
    expected = model._local_centers_yx[:, ::-1] + [215, 225]
    np.testing.assert_allclose(model._points_and_vectors(torch.zeros(1, 2, 2)).points_xy,
                               expected, atol=2e-5, rtol=0)


def test_legacy_geometry_reproduces_old_formula():
    model = conversion_model(version="legacy_v1")
    flow = torch.tensor([[[.15, -.8], [-.2, .7]]])
    expected_vectors = flow.numpy()[0].T.copy()
    expected_vectors[:, 0] *= 720 / 169
    expected_vectors[:, 1] *= -720 / 169
    centers = model._local_centers_yx + 195
    expected_points = np.column_stack([20 + centers[:, 1] / 390 * 1280,
                                       30 + centers[:, 0] / 390 * 720]).astype(np.float32)
    field = model._points_and_vectors(flow)
    assert np.array_equal(field.points_xy, expected_points)
    assert np.array_equal(field.vectors_xy, expected_vectors)


@pytest.mark.parametrize("shape", [(2, 2, 2), (1, 3, 2), (1, 2, 3)])
def test_decoder_layout_mismatch_is_rejected(shape):
    with pytest.raises(ValueError, match="поле декодера"):
        conversion_model()._points_and_vectors(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0., -1.])
@pytest.mark.parametrize("field", ["source_fps", "max_dt"])
def test_nonfinite_or_nonpositive_time_configuration_rejected(field, value):
    kwargs = {"source_fps": 30., "max_dt": .02, field: value}
    with pytest.raises(ValueError):
        integration_plan(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(flow_gain=float("nan")), dict(flow_gain=float("inf")),
    dict(fade_in_seconds=float("nan")), dict(fade_in_seconds=float("inf")),
    dict(fade_in_seconds=-1.), dict(geometry_version="unregistered")])
def test_bad_configuration_fails_before_runtime_load(monkeypatch, kwargs):
    def forbidden(**kw):
        pytest.fail("invalid input reached model initialization")
    monkeypatch.setattr("tracking.flyvis_motion.activate_flyvis", forbidden)
    with pytest.raises(ValueError):
        FlyVisMotionEstimator(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(rows=1.5), dict(columns=True), dict(rows=0)])
def test_tile_counts_are_positive_integers(kwargs):
    with pytest.raises(ValueError):
        TileLayout(**kwargs)


@pytest.mark.parametrize("dimensions", [(0, 10), (10, -1), (2.5, 10)])
def test_tile_dimensions_rejected(dimensions):
    with pytest.raises(ValueError):
        fixed_tile_bounds(*dimensions, TileLayout())


@pytest.mark.parametrize("device_name", ["cpu", "cuda:2"])
def test_synchronization_uses_model_device_not_global_availability(device_name):
    calls = []
    model = FlyVisMotionEstimator.__new__(FlyVisMotionEstimator)
    model.network = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device(device_name))]))
    # No is_available method: calling it instead of inspecting the model fails.
    model.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: calls.append(str(d))))
    model._synchronize()
    assert calls == ([device_name] if device_name.startswith("cuda") else [])


def test_zero_and_substep_fade_initialization():
    sentinel = object()
    calls = []
    model = FlyVisMotionEstimator.__new__(FlyVisMotionEstimator)
    model.dt = .02
    model.network = SimpleNamespace(
        _param_api=lambda: "params",
        _initial_state=lambda params, batch: calls.append((params, batch)) or sentinel,
        fade_in_state=lambda duration, dt, first: calls.append((duration, dt, tuple(first.shape))) or sentinel,
    )
    rendered = torch.zeros(2, 1, 1, 721)
    model.fade_in_seconds = 0
    assert model._initial_state(rendered) is sentinel
    assert calls.pop() == ("params", 2)
    model.fade_in_seconds = .005
    assert model._initial_state(rendered) is sentinel
    assert calls.pop() == (.005, .005, (2, 1, 721))


@pytest.mark.parametrize("force", [False, True])
def test_old_motion_cache_cannot_be_reused_or_overwritten_as_new(tmp_path, force):
    import json
    from scripts.run_bdd_video_rate_experiment import build_motion_cache
    data = tmp_path / "flyvis_full.npz"
    meta = tmp_path / "flyvis_full.json"
    data.write_bytes(b"existing data must remain untouched")
    meta.write_text(json.dumps({"source": "flyvis_full", "fps": 30.}), encoding="utf-8")
    before = (data.read_bytes(), meta.read_bytes())
    with pytest.raises(RuntimeError, match="геометрия legacy_v1"):
        build_motion_cache("flyvis_full", tmp_path / "absent.mov", tmp_path,
                           fps=30., max_dt=.02, force=force)
    assert (data.read_bytes(), meta.read_bytes()) == before
    old = build_motion_cache("flyvis_full", tmp_path / "absent.mov", tmp_path,
                             fps=30., max_dt=.02, force=False, geometry_version="legacy_v1")
    assert old["geometry_version"] == "legacy_v1"
    assert (data.read_bytes(), meta.read_bytes()) == before


def test_motion_factory_forwards_geometry_and_device(monkeypatch):
    import scripts.run_bdd_video_rate_experiment as script
    monkeypatch.setattr(script, "FlyVisMotionEstimator", lambda **kw: kw)
    opts = script.make_motion_estimator("flyvis_full", 30., .02,
                                        geometry_version="legacy_v1", device="cpu")
    assert opts["geometry_version"] == "legacy_v1" and opts["device"] == "cpu"
    assert script.make_motion_estimator("flyvis_full", 30., .02)["geometry_version"] == "resize_v2"


@pytest.mark.parametrize("options", [dict(fps=60., max_dt=.02), dict(fps=30., max_dt=.01)])
def test_motion_cache_rejects_changed_time_configuration(tmp_path, options):
    import json
    from scripts.run_bdd_video_rate_experiment import build_motion_cache
    data = tmp_path / "flyvis_full.npz"
    meta = tmp_path / "flyvis_full.json"
    data.write_bytes(b"old data")
    meta.write_text(json.dumps(dict(source="flyvis_full", fps=30., geometry_version="resize_v2",
                                    integration=dict(requested_max_dt_seconds=.02))), encoding="utf-8")
    before = (data.read_bytes(), meta.read_bytes())
    with pytest.raises(RuntimeError, match="не совпадает"):
        build_motion_cache("flyvis_full", tmp_path / "absent.mov", tmp_path, force=False, **options)
    assert before == (data.read_bytes(), meta.read_bytes())


def test_tracker_output_protocol_rejects_mixed_geometry_and_allows_explicit_legacy(tmp_path):
    import json
    from scripts.run_flyvis_tracker import validate_output_protocol
    metadata = dict(mode="full", source_fps=30., video_ids=["a"])
    path = tmp_path / "_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="legacy_v1"):
        validate_output_protocol(tmp_path, dict(metadata, geometry_version="resize_v2"))
    validate_output_protocol(tmp_path, dict(metadata, geometry_version="legacy_v1", video_ids=["b"]))
    assert path.read_bytes() == before


def test_tracker_output_protocol_rejects_orphan_results(tmp_path):
    from scripts.run_flyvis_tracker import validate_output_protocol
    (tmp_path / "video.json").write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="без протокола"):
        validate_output_protocol(tmp_path, dict(mode="full", geometry_version="resize_v2"))


@pytest.mark.parametrize("source_version", [None, "resize_v2"])
def test_activity_builder_uses_source_cache_geometry(tmp_path, monkeypatch, source_version):
    import json
    import scripts.build_flyvis_activity_cache as script
    root = tmp_path / "source" / "v" / "motion"
    root.mkdir(parents=True)
    meta = dict(fps=30.)
    if source_version:
        meta["geometry_version"] = source_version
    (root / "flyvis_full.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez(root / "flyvis_full.npz", points=np.zeros((1, 2)), vectors=np.zeros((1, 1, 2)))
    calls = []
    class StopAfterVersion(Exception):
        pass
    def factory(*args, **kwargs):
        calls.append(kwargs["geometry_version"])
        raise StopAfterVersion
    monkeypatch.setattr(script, "make_motion_estimator", factory)
    args = SimpleNamespace(output=tmp_path / "output", flow_root=tmp_path / "source",
                           source="flyvis_full", force=False, max_dt=.02)
    with pytest.raises(StopAfterVersion):
        script.build_video("v", args)
    assert calls == [source_version or "legacy_v1"]


@pytest.mark.parametrize("force", [False, True])
def test_activity_builder_preserves_incompatible_cache(tmp_path, force):
    import json
    from scripts.build_flyvis_activity_cache import build_video
    source = tmp_path / "source" / "v" / "motion"
    source.mkdir(parents=True)
    (source / "flyvis_full.json").write_text(json.dumps(dict(fps=30., geometry_version="resize_v2")), encoding="utf-8")
    output = tmp_path / "output" / "v"
    output.mkdir(parents=True)
    data = output / "flyvis_full_activity.npz"
    meta = output / "flyvis_full_activity.json"
    data.write_bytes(b"old activity")
    meta.write_text("{}", encoding="utf-8")
    before = (data.read_bytes(), meta.read_bytes())
    args = SimpleNamespace(output=tmp_path / "output", flow_root=tmp_path / "source",
                           source="flyvis_full", force=force, max_dt=.02)
    with pytest.raises(RuntimeError, match="геометрии не совпадает"):
        build_video("v", args)
    assert (data.read_bytes(), meta.read_bytes()) == before


@pytest.mark.parametrize("module_name,extra,filename", [
    ("flyvis_motion_calibration", [], "result.json"),
    ("flyvis_video_rate_calibration", ["--source-fps", "30"], "result.json"),
    ("benchmark_tracker_runtime", [], "result.json"),
    ("evaluate_motion_sources", [], "motion_source_metrics.json"),
])
def test_completed_diagnostic_is_protected_before_loading_models(tmp_path, monkeypatch, module_name, extra, filename):
    import importlib
    import sys
    script = importlib.import_module("scripts." + module_name)
    path = tmp_path / filename
    path.write_text('{"historical": true}', encoding="utf-8")
    output = tmp_path if module_name == "evaluate_motion_sources" else path
    monkeypatch.setattr(sys, "argv", [module_name, "--output", str(output), *extra])
    def forbidden(**kw):
        pytest.fail("completed report was not checked before model initialization")
    monkeypatch.setattr(script, "FlyVisMotionEstimator", forbidden)
    with pytest.raises(FileExistsError):
        script.main()
    assert path.read_text(encoding="utf-8") == '{"historical": true}'


@pytest.mark.parametrize("version", ["legacy_v1", "resize_v2"])
def test_k1_training_conversion_matches_streaming_module(version):
    from scripts.fly_finetune_k1 import Fly, to_px
    base = conversion_model(version=version)
    fly = Fly.__new__(Fly)
    fly.base, fly.torch, fly.dev, fly.bounds = base, torch, "cpu", base._bounds
    points, scales = fly.points_scale()
    raw = torch.tensor([[[.3, -.4], [.2, -.1]]])
    expected = base._points_and_vectors(raw)
    actual = to_px(raw, torch.from_numpy(scales), torch.tensor(0.))
    assert np.array_equal(points, expected.points_xy)
    np.testing.assert_allclose(actual.numpy(), expected.vectors_xy, rtol=1e-6)


def test_k1_anisotropic_conversion_has_correct_finite_gradients():
    from scripts.fly_finetune_k1 import to_px
    raw = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3).requires_grad_()
    scales = torch.tensor([[2., 3.], [5., 7.]], dtype=torch.float64)
    log_gain = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    value = to_px(raw, scales, log_gain)
    value.sum().backward()
    expected = (scales * torch.tensor([1., -1.]))[:, :, None].expand(2, 2, 3) * np.exp(.2)
    torch.testing.assert_close(raw.grad, expected)
    torch.testing.assert_close(log_gain.grad, value.detach().sum())
    assert torch.isfinite(raw.grad).all() and torch.isfinite(log_gain.grad)


def test_k1_old_scalar_conversion_remains_explicitly_available():
    from scripts.fly_finetune_k1 import to_px
    raw = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    scales = torch.tensor([2., 4.])
    expected = (raw.permute(0, 2, 1) * scales[:, None, None] * torch.tensor([1., -1.])).reshape(-1, 2)
    torch.testing.assert_close(to_px(raw, scales, torch.tensor(0.)), expected)


def test_k1_old_weights_and_output_require_explicit_legacy(tmp_path):
    from scripts.fly_finetune_k1 import validate_arm_geometry, validate_output_geometry
    arms = {"decoder": dict(log_gain=0.)}
    path = tmp_path / "arms.pt"
    path.write_bytes(b"historical checkpoint")
    with pytest.raises(RuntimeError, match="legacy_v1"):
        validate_arm_geometry(arms, "resize_v2")
    with pytest.raises(RuntimeError, match="legacy_v1"):
        validate_output_geometry(tmp_path, "resize_v2")
    validate_arm_geometry(arms, "legacy_v1")
    validate_output_geometry(tmp_path, "legacy_v1")
    assert path.read_bytes() == b"historical checkpoint"


def test_k1_eval_only_never_starts_training_when_weights_missing(tmp_path, monkeypatch):
    import sys
    import scripts.fly_finetune_k1 as script
    monkeypatch.setattr(sys, "argv", ["k1", "--eval-only", "--out", str(tmp_path)])
    monkeypatch.setattr(script, "OUT", tmp_path)
    monkeypatch.setattr(script, "Fly", lambda *a, **kw: pytest.fail("initialized without weights"))
    with pytest.raises(FileNotFoundError, match="eval-only"):
        script.main()


def test_k1_requires_known_frame_geometry_for_scales():
    from scripts.fly_finetune_k1 import Fly
    fly = Fly.__new__(Fly)
    fly.bounds = None
    with pytest.raises(RuntimeError, match="tiles"):
        fly.points_scale()


@pytest.mark.parametrize("device_name", ["cpu", "cuda:2"])
def test_detection_cache_timing_follows_actual_predictor_device(tmp_path, monkeypatch, device_name):
    import sys
    import scripts.run_bdd_video_rate_experiment as script
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("queried global CUDA availability"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda dev: calls.append(("sync", str(dev))))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: calls.append(("name", str(dev))) or "fake GPU")
    model = SimpleNamespace(predictor=SimpleNamespace(device=torch.device(device_name)),
                            predict=lambda *a, **kw: [SimpleNamespace(boxes=None)])
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: model, __version__="test"))
    class Capture:
        def __init__(self):
            self.count = 0
        def isOpened(self):
            return True
        def read(self):
            self.count += 1
            return (True, np.zeros((4, 4, 3), np.uint8)) if self.count == 1 else (False, None)
        def release(self):
            pass
    monkeypatch.setattr(script.cv2, "VideoCapture", lambda path: Capture())
    video, weights = tmp_path / "video.mov", tmp_path / "weights.pt"
    video.write_bytes(b"video")
    weights.write_bytes(b"weights")
    meta = script.build_detection_cache(video, weights, tmp_path / "cache.npz", imgsz=32,
                                       conf=.1, iou=.7, device="cpu", force=False)
    assert meta["software"]["prediction_device"] == device_name
    assert calls == ([] if device_name == "cpu" else [("sync", device_name)] * 2 + [("name", device_name)])
    assert meta["software"]["gpu"] == (None if device_name == "cpu" else "fake GPU")


@pytest.mark.parametrize("device_name", ["cpu", "cuda:2"])
def test_benchmark_sync_uses_requested_model_device(monkeypatch, device_name):
    from scripts.benchmark_tracker_runtime import synchronize
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("queried global CUDA availability"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda dev: calls.append(str(dev)))
    synchronize(device_name)
    assert calls == ([] if device_name == "cpu" else [device_name])
