"""Bind newly produced motion caches to the actual video; preserve legacy reads."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from scripts import run_bdd_video_rate_experiment as script


def write_video(path, shift):
    texture = np.random.default_rng(445).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5., (64, 48))
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.roll(texture, index * shift, axis=1))
    writer.release()


@pytest.fixture
def inputs(tmp_path):
    right, left = tmp_path / "right.avi", tmp_path / "left.avi"
    write_video(right, 2)
    write_video(left, -3)
    return right, left, tmp_path / "motion"


def build(video, output, force=False):
    return script.build_motion_cache("farneback", video, output, fps=5., max_dt=.02, force=force)


def saved(output):
    return tuple((output / ("farneback" + suffix)).read_bytes() for suffix in (".npz", ".json"))


def test_new_cache_records_resolved_video_path_and_content_hash(inputs):
    right, _, output = inputs
    result = build(right, output)
    assert result["video_identity"] == {"video_path": str(right.resolve()), "video_sha256": script.sha256(right)}
    assert json.loads((output / "farneback.json").read_text(encoding="utf-8"))["video_identity"] == result["video_identity"]


def test_same_video_reuses_exact_arrays_metadata_without_model(inputs, monkeypatch):
    right, _, output = inputs
    first = build(right, output)
    before = saved(output)
    monkeypatch.setattr(script, "make_motion_estimator", lambda *a, **kw: pytest.fail("cache reuse constructed a model"))
    assert build(right, output) == first
    assert saved(output) == before


@pytest.mark.parametrize("change", ["other_video", "same_path", "same_bytes_other_path"])
@pytest.mark.parametrize("force", [False, True])
def test_changed_video_is_rejected_before_model_or_writes(inputs, monkeypatch, change, force):
    right, left, output = inputs
    build(right, output)
    before = saved(output)
    if change == "same_path":
        right.write_bytes(left.read_bytes())
        requested = right
    elif change == "same_bytes_other_path":
        left.write_bytes(right.read_bytes())
        requested = left
    else:
        requested = left
    monkeypatch.setattr(script, "make_motion_estimator", lambda *a, **kw: pytest.fail("mismatched cache constructed a model"))
    with pytest.raises(RuntimeError, match="видео"):
        build(requested, output, force=force)
    assert saved(output) == before


def test_explicit_force_rebuilds_same_video_and_matches_fresh_native_result(inputs, monkeypatch):
    right, _, output = inputs
    build(right, output)
    calls = []
    original_factory = script.make_motion_estimator
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original_factory(*args, **kwargs)
    monkeypatch.setattr(script, "make_motion_estimator", counted)
    changed = build(right, output, force=True)
    assert len(calls) == 1
    build(right, output.parent / "fresh")
    actual = script.load_motion_cache(output / "farneback.npz", 1.)
    expected = script.load_motion_cache(output.parent / "fresh/farneback.npz", 1.)
    assert changed["video_identity"]["video_sha256"] == script.sha256(right)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_failed_forced_backend_preserves_previous_cache(inputs, monkeypatch):
    right, _, output = inputs
    build(right, output)
    before = saved(output)
    def failed(*args, **kwargs):
        raise RuntimeError("backend failed")
    monkeypatch.setattr(script, "make_motion_estimator", failed)
    with pytest.raises(RuntimeError, match="backend failed"):
        build(right, output, force=True)
    assert saved(output) == before


def test_source_change_during_forced_build_is_rejected_before_publishing(inputs, monkeypatch):
    right, left, output = inputs
    build(right, output)
    before = saved(output)
    original_factory = script.make_motion_estimator
    def change_after_identity(*args, **kwargs):
        right.write_bytes(left.read_bytes())
        return original_factory(*args, **kwargs)
    monkeypatch.setattr(script, "make_motion_estimator", change_after_identity)
    with pytest.raises(RuntimeError, match="видео"):
        build(right, output, force=True)
    assert saved(output) == before


@pytest.mark.parametrize("failed_call", [1, 2])
def test_hash_read_failure_preserves_previous_cache(inputs, monkeypatch, failed_call):
    right, _, output = inputs
    build(right, output)
    before = saved(output)
    original_sha = script.sha256
    calls = 0
    def fail_read(path):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise OSError("video hash read failed")
        return original_sha(path)
    monkeypatch.setattr(script, "sha256", fail_read)
    with pytest.raises(OSError, match="hash read failed"):
        build(right, output, force=True)
    assert saved(output) == before


def test_legacy_cache_with_absent_video_remains_readable_without_assigned_identity(tmp_path):
    path = tmp_path / "farneback.npz"
    path.write_bytes(b"historical arrays unchanged")
    path.with_suffix(".json").write_text(json.dumps({"source": "farneback", "fps": 5.}), encoding="utf-8")
    before = saved(tmp_path)
    result = build(tmp_path / "absent.mov", tmp_path)
    assert "video_identity" not in result
    assert saved(tmp_path) == before


@pytest.mark.parametrize("device", [None, "cpu"])
def test_builder_forwards_explicit_device_and_preserves_old_default(inputs, monkeypatch, device):
    right, _, output = inputs
    calls = []
    class BoundaryReached(Exception):
        pass
    def factory(*args, **kwargs):
        calls.append(kwargs)
        raise BoundaryReached
    monkeypatch.setattr(script, "make_motion_estimator", factory)
    options = {} if device is None else {"device": device}
    with pytest.raises(BoundaryReached):
        script.build_motion_cache("flyvis_full", right, output, fps=5., max_dt=.02, force=False, **options)
    assert calls[0].get("device") == device


@pytest.mark.parametrize("selector, logical_device", [(None, "cuda:0"), ("2", "cuda:0"), ("cpu", "cpu")])
@pytest.mark.parametrize("detection_cache_hit", [False, True])
def test_cli_shares_resolved_detector_device_with_flyvis(inputs, monkeypatch, selector, logical_device, detection_cache_hit):
    import sys
    import torch
    from ultralytics.utils import torch_utils
    right, _, output = inputs
    truth = output.parent / "truth.json"
    truth.write_text('[{"name":"f.jpg"}]', encoding="utf-8")
    argv = ["video_rate", "--motion-only", "--motion-sources", "flyvis_full", "--output", str(output)]
    if selector is not None:
        argv.extend(["--device", selector])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(script, "resolve_paths", lambda args: (right, truth, []))
    monkeypatch.setattr(script, "build_alignment", lambda *a, **kw: ([{}], {"fps": 5., "decoded_frame_count": 4}))
    selections = []
    def select(device, verbose):
        selections.append(device)
        return torch.device(logical_device)
    monkeypatch.setattr(torch_utils, "select_device", select)
    def detection_cache(*args, **kwargs):
        if not detection_cache_hit:
            assert kwargs["resolve_device"]() == torch.device(logical_device)
        return {}
    monkeypatch.setattr(script, "build_detection_cache", detection_cache)
    monkeypatch.setattr(script, "load_detection_cache", lambda *a: {"frame_count": 4})
    calls = []
    class BoundaryReached(Exception):
        pass
    def factory(*args, **kwargs):
        calls.append(kwargs)
        raise BoundaryReached
    monkeypatch.setattr(script, "make_motion_estimator", factory)
    with pytest.raises(BoundaryReached):
        script.main()
    assert calls[0]["device"] == logical_device
    assert selections == [selector or "0"]


def test_cached_flyvis_motion_does_not_select_a_cuda_device(inputs):
    video, _, output = inputs
    output.mkdir()
    path = output / "flyvis_full.npz"
    path.write_bytes(b"existing motion")
    path.with_suffix(".json").write_text(json.dumps({"source": "flyvis_full", "fps": 5.0}), encoding="utf-8")
    result = script.build_motion_cache("flyvis_full", video, output, fps=5., max_dt=.02, force=False,
                                       device="0", resolve_device=lambda: pytest.fail("selected CUDA on cache hit"))
    assert result["source"] == "flyvis_full"
