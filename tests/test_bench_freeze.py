"""Цель 0: заморозка стенда — тесты сравнения, тождества кэшей и порядка видео."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.freeze_bench import (
    build_manifest,
    compare_scalabel,
    detection_caches_identical,
    run_xywh_on_sequences,
    tracking_reads_ground_truth,
)
from tracking.offline_association import _DetSlice


def _frame(index: int, labels: list[dict]) -> dict:
    return {
        "videoName": "v",
        "name": f"v-{index:07d}.jpg",
        "frameIndex": index,
        "labels": labels,
    }


def _label(track_id: str, x1: float = 10.0) -> dict:
    return {
        "id": track_id,
        "category": "car",
        "score": 0.9,
        "box2d": {"x1": x1, "y1": 10.0, "x2": x1 + 20.0, "y2": 30.0},
    }


def test_compare_scalabel_identical_rows_report_zero_mismatches() -> None:
    rows = [_frame(0, [_label("1")]), _frame(6, [_label("1", 12.0)])]
    report = compare_scalabel(rows, json.loads(json.dumps(rows)))
    assert report["identical"] is True
    assert report["frames"] == 2
    assert report["labels"] == 2
    assert report["mismatches"] == []


def test_compare_scalabel_detects_changed_id_with_locator() -> None:
    published = [_frame(0, [_label("1")]), _frame(6, [_label("1", 12.0)])]
    rerun = [_frame(0, [_label("1")]), _frame(6, [_label("2", 12.0)])]
    report = compare_scalabel(published, rerun)
    assert report["identical"] is False
    assert report["mismatches"][0]["frameIndex"] == 6
    assert report["mismatches"][0]["field"] == "id"


def test_compare_scalabel_detects_changed_box_coordinate() -> None:
    published = [_frame(0, [_label("1")])]
    rerun = [_frame(0, [_label("1", 10.5)])]
    report = compare_scalabel(published, rerun)
    assert report["identical"] is False
    assert report["mismatches"][0]["field"] == "box2d.x1"


def _write_cache(path: Path, *, conf_shift: float = 0.0, decode_seed: int = 0) -> None:
    rng = np.random.default_rng(decode_seed)
    np.savez(
        path,
        xyxy=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32),
        conf=np.asarray([0.5 + conf_shift, 0.1], np.float32),
        cls=np.asarray([2, 0], np.int16),
        frame_idx=np.asarray([0, 1], np.int32),
        frame_count=np.asarray(2, np.int32),
        decode_seconds=rng.random(2),
        detector_seconds=rng.random(2),
    )


def test_detection_caches_identical_ignores_timing_arrays(tmp_path: Path) -> None:
    _write_cache(tmp_path / "a.npz", decode_seed=1)
    _write_cache(tmp_path / "b.npz", decode_seed=2)
    report = detection_caches_identical(tmp_path / "a.npz", tmp_path / "b.npz")
    assert report["identical"] is True
    assert report["detections"] == 2


def test_detection_caches_identical_detects_changed_confidence(tmp_path: Path) -> None:
    _write_cache(tmp_path / "a.npz")
    _write_cache(tmp_path / "b.npz", conf_shift=1e-3)
    report = detection_caches_identical(tmp_path / "a.npz", tmp_path / "b.npz")
    assert report["identical"] is False
    assert "conf" in report["differing_arrays"]


def test_tracking_reads_ground_truth_flags_label_access(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text(
        "import numpy as np\n\ndef update(xyxy):\n    return xyxy\n", encoding="utf-8"
    )
    (tmp_path / "leak.py").write_text(
        "from pathlib import Path\n"
        "def cheat(video_id):\n"
        "    return Path('data/bdd100k/labels/box_track_20/val') / f'{video_id}.json'\n",
        encoding="utf-8",
    )
    hits = tracking_reads_ground_truth(tmp_path)
    assert [hit["file"] for hit in hits] == ["leak.py"]
    assert hits[0]["line"] == 3


def test_tracking_reads_ground_truth_is_clean_for_project_tracking_package() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert tracking_reads_ground_truth(project_root / "tracking") == []


def _sequence(offset: float, frames: int, score: float = 0.9) -> list[_DetSlice]:
    slices = []
    for index in range(frames):
        box = np.asarray([[offset + index, 10, offset + index + 20, 30]], np.float32)
        slices.append(
            _DetSlice(box, np.asarray([score], np.float32), np.asarray([2], np.float32))
        )
    return slices


def test_run_xywh_on_sequences_is_invariant_to_video_order() -> None:
    sequences = {"a": _sequence(0.0, 5), "b": _sequence(100.0, 5)}
    forward = run_xywh_on_sequences(sequences, order=["a", "b"])
    backward = run_xywh_on_sequences(sequences, order=["b", "a"])
    for video_id in sequences:
        for left, right in zip(forward[video_id], backward[video_id]):
            np.testing.assert_array_equal(left, right)


def test_run_xywh_on_sequences_restarts_ids_for_every_video() -> None:
    sequences = {"a": _sequence(0.0, 3), "b": _sequence(100.0, 3)}
    outputs = run_xywh_on_sequences(sequences, order=["a", "b"])
    assert outputs["a"][-1][0, 4] == 1.0
    assert outputs["b"][-1][0, 4] == 1.0


def test_build_manifest_is_deterministic_and_lists_required_sections(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    video = tmp_path / "v.mov"
    video.write_bytes(b"video")
    first = build_manifest(weights=weights, videos={"v": video}, cache_files={"v": [video]})
    second = build_manifest(weights=weights, videos={"v": video}, cache_files={"v": [video]})
    assert first == second
    assert len(first["weights"]["sha256"]) == 64
    assert first["videos"]["v"]["sha256"] != first["weights"]["sha256"]
    for section in ("weights", "videos", "caches", "libraries", "tracker_parameters", "evaluator"):
        assert section in first
    assert first["libraries"]["ultralytics"] == "8.4.84"
