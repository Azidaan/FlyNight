"""Цель 0: заморозить воспроизводимый стенд трекера на семи исходных видео.

Манифест хэшей и версий, повторный прогон неизменённых трекеров по сохранённым
кэшам и побайтовое сравнение с опубликованными выходами. Обучение, YOLO26 и сеть
FlyVis не запускаются: проверяется воспроизводимость ассоциации, не детектора.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import sys
import time
import tokenize
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hota_eval import evaluate_hota  # noqa: E402
from scripts.run_bdd_video_rate_experiment import (  # noqa: E402
    DEFAULT_GAINS,
    detections_for_frame,
    load_detection_cache,
    run_tracker,
    save_json,
    scalabel_frame,
)
from scripts.sweep_bytetrack_video_rate import run_configuration  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402

VIDEO_IDS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
    "b232c7c9-d251d9ee",
    "b1d22449-15fb948f",
    "b251064f-e7a165fd",
)
XYWH_CONFIG = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "track_buffer": 30,
    "match_thresh": 0.8,
}
XYWH_CONFIG_NAME = "h0p25_l0p1_b30_m0p8"
XYWH_RUN_KW = {"high": 0.25, "low": 0.1, "buffer": 30, "match": 0.8}
FLYVIS_TRACKERS = ("bytetrack", "flyvis_full_calibrated", "flyvis_tiled_calibrated")
DETECTION_ARRAYS = ("xyxy", "conf", "cls", "frame_idx", "frame_count")
METRIC_KEYS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSw", "CLR_FP", "CLR_FN", "Frag")
LIBRARIES = ("ultralytics", "numpy", "torch", "opencv-python", "trackeval", "scipy", "flyvis")
GROUND_TRUTH_PATTERNS = (
    r"box_track_20",
    r"bdd100k[\\/]+labels",
    r"ground[_ ]?truth",
    r"\btruth\b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-rate-root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument(
        "--stress-root", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_stress_gate"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "bench_freeze")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--skip-order-check", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(rows: list[dict]) -> list[dict]:
    return json.loads(json.dumps(rows))


def compare_scalabel(published: list[dict], rerun: list[dict]) -> dict:
    """Сравнить два scalabel-выхода покадрово: ID, класс, балл и координаты."""
    published, rerun = _normalise(published), _normalise(rerun)
    mismatches: list[dict] = []
    labels = 0
    if len(published) != len(rerun):
        mismatches.append({"frameIndex": None, "field": "frames", "published": len(published), "rerun": len(rerun)})
    for left, right in zip(published, rerun):
        frame_index = left.get("frameIndex")
        if left.get("frameIndex") != right.get("frameIndex"):
            mismatches.append({"frameIndex": frame_index, "field": "frameIndex", "published": frame_index, "rerun": right.get("frameIndex")})
            continue
        if len(left["labels"]) != len(right["labels"]):
            mismatches.append({"frameIndex": frame_index, "field": "labels", "published": len(left["labels"]), "rerun": len(right["labels"])})
            continue
        for position, (a, b) in enumerate(zip(left["labels"], right["labels"])):
            labels += 1
            for field in ("id", "category", "score"):
                if a.get(field) != b.get(field):
                    mismatches.append({"frameIndex": frame_index, "label": position, "field": field, "published": a.get(field), "rerun": b.get(field)})
            for coordinate in ("x1", "y1", "x2", "y2"):
                if a["box2d"].get(coordinate) != b["box2d"].get(coordinate):
                    mismatches.append({"frameIndex": frame_index, "label": position, "field": f"box2d.{coordinate}", "published": a["box2d"].get(coordinate), "rerun": b["box2d"].get(coordinate)})
    return {
        "identical": not mismatches,
        "frames": len(published),
        "labels": labels,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
    }


def detection_caches_identical(first: Path, second: Path) -> dict:
    """Сравнить содержательные массивы двух кэшей детекций; времена не сравниваются."""
    differing = []
    with np.load(first) as a, np.load(second) as b:
        for key in DETECTION_ARRAYS:
            if key not in a.files or key not in b.files:
                differing.append(key)
                continue
            if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
                differing.append(key)
        detections = int(a["xyxy"].shape[0]) if "xyxy" in a.files else 0
    return {"identical": not differing, "differing_arrays": differing, "detections": detections}


def tracking_reads_ground_truth(package_dir: Path) -> list[dict]:
    """Найти маркеры разметки в исполняемом коде, включая строковые пути."""
    pattern = re.compile("|".join(GROUND_TRUTH_PATTERNS), re.IGNORECASE)
    hits = []
    for path in sorted(package_dir.rglob("*.py")):
        with tokenize.open(path) as handle:
            source = handle.read()
        tree = ast.parse(source, filename=str(path))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node, clean=False) is not None:
                docstrings.add(node.body[0].value)
        lines = source.splitlines()
        matching_lines = set()
        for node in ast.walk(tree):
            if node in docstrings:
                continue
            for _, value in ast.iter_fields(node):
                if isinstance(value, str) and pattern.search(value):
                    matching_lines.add(node.lineno)
        for number in sorted(matching_lines):
            hits.append({"file": path.relative_to(package_dir).as_posix(), "line": number,
                         "text": lines[number - 1].strip()})
    return hits


def tracking_label_audit(package_dir: Path, loaded_files) -> dict:
    """Отделить загруженные стендом модули от остальных, сохранив все совпадения."""
    root = package_dir.resolve()
    loaded = set()
    for filename in loaded_files:
        path = Path(filename).resolve()
        if path.is_relative_to(root) and path.suffix == ".py":
            loaded.add(path.relative_to(root).as_posix())
    hits = tracking_reads_ground_truth(root)
    return {"loaded_files": sorted(loaded), "package_hits": hits,
            "loaded_hits": [hit for hit in hits if hit["file"] in loaded]}


def run_xywh_on_sequences(sequences: dict[str, list], order: list[str]) -> dict[str, list[np.ndarray]]:
    """Прогнать XYWHByteTrack по нескольким «видео» в заданном порядке, новый трекер на каждое."""
    from tracking.xywh_bytetrack import XYWHByteTrack

    outputs: dict[str, list[np.ndarray]] = {}
    for video_id in order:
        tracker = XYWHByteTrack(make_tracker_args(**XYWH_CONFIG))
        outputs[video_id] = [np.array(tracker.update(frame), copy=True) for frame in sequences[video_id]]
    return outputs


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if versions.get("flyvis") is None:
        # FlyVis живёт в отдельном каталоге tools/flyvis-runtime, не в .venv.
        runtime = PROJECT_ROOT / "tools" / "flyvis-runtime"
        for metadata in sorted(runtime.glob("flyvis-*.dist-info/METADATA")):
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version:"):
                    versions["flyvis"] = line.split(":", 1)[1].strip() + " (tools/flyvis-runtime)"
                    break
    return versions


def build_manifest(*, weights: Path, videos: dict[str, Path], cache_files: dict[str, list[Path]], extra: dict | None = None) -> dict:
    """Собрать манифест стенда: хэши по фиксированному списку файлов, версии, параметры."""
    args = vars(make_tracker_args(**XYWH_CONFIG))
    manifest = {
        "weights": {"path": str(weights), "sha256": file_sha256(weights), "bytes": weights.stat().st_size},
        "videos": {
            video_id: {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for video_id, path in sorted(videos.items())
        },
        "caches": {
            video_id: [
                {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
                for path in files
            ]
            for video_id, files in sorted(cache_files.items())
        },
        "libraries": _library_versions(),
        "python": platform.python_version(),
        "tracker_parameters": {
            "bytetrack_and_xywh": {key: args[key] for key in sorted(args)},
            "flyvis_gains": dict(DEFAULT_GAINS),
            "flyvis_trackers": list(FLYVIS_TRACKERS),
        },
        "evaluator": {
            "package": "trackeval",
            "version": _library_versions().get("trackeval"),
            "entry": "evaluation.hota_eval.evaluate_hota",
            "averaging": "OVERALL (по объектам) и *_cls_av (по классам) хранятся раздельно",
        },
    }
    if extra:
        manifest.update(extra)
    return manifest


def _truth(video_id: str) -> list[dict]:
    path = PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val" / f"{video_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_delta(published: dict, rerun: dict) -> dict:
    return {
        key: {"published": published.get(key), "rerun": rerun.get(key), "delta": (rerun.get(key) - published.get(key)) if key in published and key in rerun else None}
        for key in METRIC_KEYS
    }


def _video_table(video_rate_root: Path, video_ids: list[str]) -> dict:
    table = {}
    for video_id in video_ids:
        experiment = video_rate_root / video_id
        alignment = json.loads((experiment / "alignment.json").read_text(encoding="utf-8"))
        detections = json.loads((experiment / "detections" / "raw_video.json").read_text(encoding="utf-8"))
        motion_json = next(iter(sorted((experiment / "motion").glob("*.json"))), None)
        motion = json.loads(motion_json.read_text(encoding="utf-8")) if motion_json else {}
        table[video_id] = {
            "label_file": f"data/bdd100k/labels/box_track_20/val/{video_id}.json",
            "raw_frames": detections["frame_count"],
            "labeled_frames": len(alignment["matches"]),
            "fps_from_container": motion.get("fps"),
            "fps_source": str(motion_json.relative_to(PROJECT_ROOT)) if motion_json else None,
            "raw_step_counts": alignment.get("raw_step_counts"),
            "detections": detections["detections"],
        }
    return table


def main() -> int:
    args = parse_args()
    video_ids = args.video_id or list(VIDEO_IDS)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    weights = PROJECT_ROOT / "weights" / "baseline_seed42" / "best.pt"
    videos = {video_id: PROJECT_ROOT / "data" / "bdd100k" / "videos" / "val" / f"{video_id}.mov" for video_id in video_ids}
    cache_files = {}
    for video_id in video_ids:
        files = []
        for root in (args.video_rate_root, args.stress_root):
            experiment = root / video_id
            files += sorted((experiment / "detections").glob("raw_video.*"))
            files += sorted((experiment / "motion").glob("*.npz"))
            files += [experiment / "alignment.json"]
        files.append(PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val" / f"{video_id}.json")
        cache_files[video_id] = [path for path in files if path.exists()]
    print("[0] манифест", flush=True)
    manifest = build_manifest(weights=weights, videos=videos, cache_files=cache_files, extra={"videos_table": _video_table(args.video_rate_root, video_ids)})
    save_json(output / "manifest.json", manifest)

    print("[1] тождество кэшей детекций", flush=True)
    cache_identity = {
        video_id: detection_caches_identical(
            args.video_rate_root / video_id / "detections" / "raw_video.npz",
            args.stress_root / video_id / "detections" / "raw_video.npz",
        )
        for video_id in video_ids
    }

    print("[2] XYWHByteTrack: повторный прогон", flush=True)
    xywh_dir = output / "xywh" / XYWH_CONFIG_NAME
    xywh = run_configuration(args.video_rate_root, video_ids, xywh_dir, tracker_class="xywh", **XYWH_RUN_KW)
    published_dir = args.video_rate_root / "xywh_validation7" / XYWH_CONFIG_NAME
    published_metrics = json.loads((args.video_rate_root / "xywh_validation7.json").read_text(encoding="utf-8"))["results"][XYWH_CONFIG_NAME]["metrics"]
    xywh_report = {
        "per_video": {video_id: compare_scalabel(_load_rows(published_dir / f"{video_id}.json"), _load_rows(xywh_dir / f"{video_id}.json")) for video_id in video_ids},
        "metrics": _metric_delta(published_metrics, xywh["metrics"]),
    }

    print("[3] FlyVisTracker и ByteTrack по кэшу полей: повторный прогон", flush=True)
    flyvis_report = {}
    stress_summary = json.loads((args.stress_root / "combined" / "summary.json").read_text(encoding="utf-8"))["metrics"]
    for name in FLYVIS_TRACKERS:
        result_dir = output / "stress" / name
        per_video = {}
        for video_id in video_ids:
            experiment = args.stress_root / video_id
            detections = load_detection_cache(experiment / "detections" / "raw_video.npz")
            alignment = json.loads((experiment / "alignment.json").read_text(encoding="utf-8"))
            rows, _ = run_tracker(name, detections, _truth(video_id), alignment["matches"], experiment / "motion")
            save_json(result_dir / f"{video_id}.json", rows)
            per_video[video_id] = compare_scalabel(_load_rows(experiment / "results" / name / f"{video_id}.json"), rows)
        flyvis_report[name] = {
            "per_video": per_video,
            "metrics": _metric_delta(stress_summary.get(name, {}), evaluate_hota(result_dir, video_ids)),
        }

    order_report = None
    if not args.skip_order_check:
        print("[4] инвариантность к порядку видео (XYWHByteTrack, обратный порядок)", flush=True)
        reversed_dir = output / "xywh_reversed" / XYWH_CONFIG_NAME
        run_configuration(args.video_rate_root, list(reversed(video_ids)), reversed_dir, tracker_class="xywh", **XYWH_RUN_KW)
        order_report = {video_id: compare_scalabel(_load_rows(xywh_dir / f"{video_id}.json"), _load_rows(reversed_dir / f"{video_id}.json")) for video_id in video_ids}

    print("[5] статическая проверка доступа к разметке в tracking/", flush=True)
    audit = tracking_label_audit(PROJECT_ROOT / "tracking", [
        mod.__file__ for mod in tuple(sys.modules.values()) if getattr(mod, "__file__", None)
    ])
    required_modules = {"flyvis_tracker.py", "offline_association.py", "xywh_bytetrack.py"}

    def all_identical(section: dict) -> bool:
        return all(report["identical"] for report in section.values())

    def metrics_match(delta: dict) -> bool:
        return all(row["delta"] is not None and abs(row["delta"]) < 1e-9 for row in delta.values())

    acceptance = {
        "detection_caches_identical": all_identical(cache_identity),
        "xywh_rows_identical": all_identical(xywh_report["per_video"]),
        "stress_rows_identical": all(all_identical(report["per_video"]) for report in flyvis_report.values()),
        "xywh_metrics_match": metrics_match(xywh_report["metrics"]),
        "stress_metrics_match": all(metrics_match(report["metrics"]) for report in flyvis_report.values()),
        "order_invariant": None if order_report is None else all_identical(order_report),
        "tracking_free_of_ground_truth": required_modules.issubset(audit["loaded_files"]) and not audit["loaded_hits"],
    }
    reproduction = {
        "video_ids": video_ids,
        "elapsed_seconds": time.perf_counter() - started,
        "cache_identity": cache_identity,
        "xywh": xywh_report,
        "stress": flyvis_report,
        "order_invariance": order_report,
        "ground_truth_hits": audit["loaded_hits"],
        "tracking_label_audit": audit,
        "acceptance": acceptance,
    }
    save_json(output / "reproduction.json", reproduction)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    print(f"[OK] {output}")
    return 0 if all(value in (True, None) for value in acceptance.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
