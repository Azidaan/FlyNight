"""Сравнить ByteTrack и отдельный FlyVisTracker на исходном видео BDD100K.

Трекеры получают одни и те же обнаружения YOLO26 на каждом исходном кадре.
Результаты прореживаются до кадров с разметкой только после ассоциации.
Поля движения кэшируются при единичном масштабе, чтобы разные коэффициенты
FlyVis сравнивались без повторного запуска рекуррентной сети.
"""

from __future__ import annotations

import argparse
from functools import cache
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES, DATA_ROOT  # noqa: E402
from evaluation.hota_eval import evaluate_hota  # noqa: E402
from tracking.classical_motion import OpenCVMotionEstimator  # noqa: E402
from tracking.flyvis_motion import (  # noqa: E402
    FlyVisMotionEstimator, TileLayout, GEOMETRY_VERSION, GEOMETRY_VERSIONS,
)
from tracking.flyvis_tracker import (  # noqa: E402
    FlyVisTracker,
    FlyVisTrackerConfig,
    MotionField,
    shift_box,
)
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402
from tracking.video_alignment import (  # noqa: E402
    align_reference_frames,
    decode_compact_video,
    load_compact_images,
)


DEFAULT_VIDEO_ID = "b1c66a42-6f7d68ca"
DEFAULT_GAINS = {"flyvis_full": 16.00775, "flyvis_tiled": 38.84157}
LEGACY_GAIN = 102.596


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "baseline_seed42" / "best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-dt", type=float, default=0.02)
    parser.add_argument("--flyvis-geometry-version", "--motion-geometry-version", choices=GEOMETRY_VERSIONS,
                        default=GEOMETRY_VERSION)
    parser.add_argument("--search-radius", type=int, default=24)
    parser.add_argument(
        "--motion-sources",
        nargs="+",
        choices=(
            "dis",
            "dis_half",
            "dis_quarter",
            "farneback",
            "flyvis_full",
            "flyvis_tiled",
        ),
        default=("dis", "flyvis_full", "flyvis_tiled"),
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=(
            "bytetrack",
            "constant_velocity",
            "dis",
            "flyvis_full_calibrated",
            "flyvis_tiled_calibrated",
            "flyvis_full_legacy",
            "flyvis_tiled_legacy",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "video_rate" / DEFAULT_VIDEO_ID,
    )
    parser.add_argument("--force-detections", action="store_true")
    parser.add_argument("--force-motion", action="store_true")
    parser.add_argument("--motion-only", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def distribution(values: list[float] | np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"samples": 0}
    return {
        "samples": int(len(values)),
        "mean_ms": float(values.mean() * 1000.0),
        "median_ms": float(np.median(values) * 1000.0),
        "p90_ms": float(np.quantile(values, 0.90) * 1000.0),
        "p99_ms": float(np.quantile(values, 0.99) * 1000.0),
        "fps_from_mean": float(1.0 / values.mean()),
    }


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    video = args.video or (
        DATA_ROOT / "videos" / "val" / f"{args.video_id}.mov"
    )
    truth = DATA_ROOT / "labels" / "box_track_20" / "val" / f"{args.video_id}.json"
    images = sorted(
        (DATA_ROOT / "images" / "track" / "val" / args.video_id).glob("*.jpg")
    )
    for path in (video, truth, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not images:
        raise FileNotFoundError(f"Не найдены размеченные кадры: {args.video_id}")
    return video, truth, images


def build_alignment(
    video: Path,
    images: list[Path],
    output: Path,
    search_radius: int,
) -> tuple[list[dict], dict]:
    compact_video, metadata = decode_compact_video(video)
    references = load_compact_images(images)
    matches = align_reference_frames(
        compact_video,
        references,
        [path.name for path in images],
        search_radius=search_radius,
    )
    rows = [match.to_dict() for match in matches]
    errors = np.asarray([row["mean_absolute_error"] for row in rows])
    margins = np.asarray(
        [row["second_best_error"] - row["mean_absolute_error"] for row in rows]
    )
    steps = np.diff([row["video_frame_index"] for row in rows])
    audit = {
        "video": metadata,
        "reference_frames": len(images),
        "search_radius": search_radius,
        "maximum_match_mae": float(errors.max()),
        "median_match_mae": float(np.median(errors)),
        "minimum_second_best_margin": float(margins.min()),
        "median_second_best_margin": float(np.median(margins)),
        "raw_step_counts": {
            str(int(step)): int(np.sum(steps == step)) for step in np.unique(steps)
        },
        "matches": rows,
    }
    save_json(output, audit)
    return rows, metadata


def _empty_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.int16),
    )


def build_detection_cache(
    video: Path,
    weights: Path,
    cache_path: Path,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    force: bool,
    resolve_device=None,
) -> dict:
    meta_path = cache_path.with_suffix(".json")
    identity = {
        "video_path": str(video.resolve()),
        "video_sha256": sha256(video),
        "weights_path": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "device": str(device),
    }
    if cache_path.exists() and meta_path.exists() and not force:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise RuntimeError("Кэш обнаружений существует, но его протокол отличается")
        print(f"[CACHE] обнаружения: {cache_path}", flush=True)
        return existing

    import torch
    from ultralytics import YOLO

    # Keep the original CLI selector in cache identity. The shared resolver
    # supplies the logical Torch device after Ultralytics maps visible GPUs.
    if resolve_device is not None:
        device = resolve_device()
    model = YOLO(str(weights))
    warm = cv2.VideoCapture(str(video))
    ok, warm_frame = warm.read()
    warm.release()
    if not ok:
        raise RuntimeError(f"Не удалось прочитать первый кадр {video}")
    for _ in range(2):
        model.predict(
            warm_frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
    prediction_device = torch.device(model.predictor.device)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть {video}")
    xyxy_all, conf_all, cls_all, frame_idx_all = [], [], [], []
    decode_seconds, detector_seconds = [], []
    frame_count = 0
    total_started = time.perf_counter()
    while True:
        decode_started = time.perf_counter()
        ok, frame = capture.read()
        decode_seconds.append(time.perf_counter() - decode_started)
        if not ok:
            decode_seconds.pop()
            break
        if prediction_device.type == "cuda":
            torch.cuda.synchronize(prediction_device)
        detector_started = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            xyxy, scores, classes = _empty_arrays()
        else:
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
            classes = boxes.cls.detach().cpu().numpy().astype(np.int16)
        if prediction_device.type == "cuda":
            torch.cuda.synchronize(prediction_device)
        detector_seconds.append(time.perf_counter() - detector_started)
        if len(scores):
            xyxy_all.append(xyxy)
            conf_all.append(scores)
            cls_all.append(classes)
            frame_idx_all.append(np.full(len(scores), frame_count, np.int32))
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"[YOLO] {frame_count} кадров", flush=True)
    capture.release()
    total_seconds = time.perf_counter() - total_started

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        xyxy=np.concatenate(xyxy_all) if xyxy_all else _empty_arrays()[0],
        conf=np.concatenate(conf_all) if conf_all else _empty_arrays()[1],
        cls=np.concatenate(cls_all) if cls_all else _empty_arrays()[2],
        frame_idx=(
            np.concatenate(frame_idx_all)
            if frame_idx_all
            else np.zeros(0, dtype=np.int32)
        ),
        frame_count=np.asarray(frame_count, dtype=np.int32),
        decode_seconds=np.asarray(decode_seconds, dtype=np.float64),
        detector_seconds=np.asarray(detector_seconds, dtype=np.float64),
    )
    metadata = {
        "identity": identity,
        "frame_count": frame_count,
        "detections": int(sum(len(values) for values in conf_all)),
        "decode": distribution(decode_seconds),
        "detector": distribution(detector_seconds),
        "wall_seconds_including_decode": total_seconds,
        "pipeline_fps_including_decode": frame_count / total_seconds,
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "prediction_device": str(prediction_device),
            "gpu": torch.cuda.get_device_name(prediction_device) if prediction_device.type == "cuda" else None,
        },
    }
    save_json(meta_path, metadata)
    return metadata


def load_detection_cache(path: Path) -> dict:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def detections_for_frame(cache: dict, frame_index: int):
    mask = cache["frame_idx"] == frame_index
    return cache["xyxy"][mask], cache["conf"][mask], cache["cls"][mask]


def make_motion_estimator(source: str, fps: float, max_dt: float, model: str = "flow/0000/000",
                          *, geometry_version: str = GEOMETRY_VERSION, device=None):
    if source in {"dis", "dis_half", "dis_quarter", "farneback"}:
        method = "dis" if source.startswith("dis") else source
        scale = {"dis_half": 0.5, "dis_quarter": 0.25}.get(source, 1.0)
        return OpenCVMotionEstimator(method, processing_scale=scale, sample_step=8,
                                    geometry_version=geometry_version)
    mode = source.removeprefix("flyvis_")
    return FlyVisMotionEstimator(
        mode=mode,
        layout=TileLayout(2, 3, 0.20),
        source_fps=fps,
        dt=max_dt,
        flow_gain=1.0,
        temporal_resampling="linear",
        flow_aggregation="mean",
        model=model,
        geometry_version=geometry_version,
        device=device,
    )


def motion_video_identity(video: Path) -> dict:
    """Привязать новое поле к конкретному файлу и его содержимому."""
    return {"video_path": str(video.resolve()), "video_sha256": sha256(video)}


def build_motion_cache(
    source: str,
    video: Path,
    output: Path,
    *,
    fps: float,
    max_dt: float,
    force: bool,
    geometry_version: str = GEOMETRY_VERSION,
    device=None,
    resolve_device=None,
) -> dict:
    """Новый кэш проверяет видео; исторический без identity читается как есть.

    Прежние файлы без video_identity не получают подтверждения происхождения.
    Force пересчитывает то же видео; для смены видео нужен новый output.
    """
    data_path = output / f"{source}.npz"
    meta_path = output / f"{source}.json"
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_dt) or max_dt <= 0:
        raise ValueError("fps и max_dt должны быть конечными положительными числами")
    if geometry_version not in GEOMETRY_VERSIONS:
        raise ValueError(f"Неизвестная геометрия поля: {geometry_version}")
    # Старые файлы сохраняются: --force-motion не разрешает превратить
    # прежний эксперимент в новый под тем же именем.
    if data_path.exists() and not meta_path.exists():
        raise RuntimeError(f"У {data_path} нет метаданных; выберите новый output")
    video_identity = None
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        previous_geometry = previous.get("geometry_version", "legacy_v1")
        if previous_geometry != geometry_version:
            raise RuntimeError(
                f"{data_path}: геометрия {previous_geometry}, запрошена {geometry_version}; "
                "выберите новый output либо явно прежнюю geometry_version"
            )
        if "video_identity" in previous:
            video_identity = motion_video_identity(video)
            if previous["video_identity"] != video_identity:
                raise RuntimeError(f"{data_path}: видео или его содержимое изменилось; "
                                   "выберите новый output, force не меняет происхождение кэша")
    if data_path.exists() and meta_path.exists() and not force:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("source") != source or not np.isclose(
            metadata.get("fps", float("nan")), fps, rtol=1e-9, atol=0
        ):
            raise RuntimeError(f"{data_path}: источник или fps не совпадает с запросом; выберите новый output")
        previous_dt = metadata.get("integration", {}).get("requested_max_dt_seconds")
        if source.startswith("flyvis_") and previous_dt is not None and not np.isclose(
            previous_dt, max_dt, rtol=1e-9, atol=0
        ):
            raise RuntimeError(f"{data_path}: max_dt не совпадает с запросом; выберите новый output")
        print(f"[CACHE] движение {source}: {data_path}", flush=True)
        if "video_identity" not in metadata:
            print("[CACHE] исторический кэш: происхождение видео не подтверждено", flush=True)
        metadata.setdefault("geometry_version", "legacy_v1")
        return metadata

    if video_identity is None:
        video_identity = motion_video_identity(video)
    initialized = time.perf_counter()
    if resolve_device is not None and source.startswith("flyvis_"):
        device = str(resolve_device())
    estimator = make_motion_estimator(source, fps, max_dt, geometry_version=geometry_version, device=device)
    initialization_seconds = time.perf_counter() - initialized
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть {video}")
    vectors, observation_seconds = [], []
    points = None
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        started = time.perf_counter()
        field = estimator.observe(frame)
        observation_seconds.append(time.perf_counter() - started)
        if points is None:
            points = field.points_xy.copy()
        elif not np.allclose(points, field.points_xy):
            raise RuntimeError(f"Сетка {source} изменилась внутри видео")
        vectors.append(field.vectors_xy.copy())
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"[{source}] {frame_count} кадров", flush=True)
    capture.release()
    if points is None:
        raise RuntimeError("Видео не содержит кадров")
    if motion_video_identity(video) != video_identity:
        raise RuntimeError("Содержимое видео изменилось во время расчёта; кэш не записан")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        points=points.astype(np.float32),
        vectors=np.stack(vectors).astype(np.float32),
        observation_seconds=np.asarray(observation_seconds, dtype=np.float64),
    )
    metadata = {
        "source": source,
        "video_identity": video_identity,
        "frame_count": frame_count,
        "fps": fps,
        "flow_gain_in_cache": 1.0,
        "initialization_seconds_excluded": initialization_seconds,
        "observation": distribution(observation_seconds),
        "estimator_stats": estimator.stats,
        "geometry_version": estimator.geometry_version,
    }
    if isinstance(estimator, FlyVisMotionEstimator):
        metadata["flow_gain_status"] = ("raw_unvalidated_resize_diagnostic" if estimator.geometry_version == "resize_v2"
                                        else "raw_legacy_empirical_vector_convention")
        metadata["integration"] = {
            "steps_per_video_frame": estimator.integration_steps,
            "integration_dt_seconds": estimator.dt,
            "requested_max_dt_seconds": estimator.requested_max_dt,
            "temporal_resampling": estimator.temporal_resampling,
            "flow_aggregation": estimator.flow_aggregation,
        }
    save_json(meta_path, metadata)
    return metadata


def load_motion_cache(path: Path, gain: float, *, allow_unvalidated_resize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    # Старые потребители передают фиксированные эмпирические gains без версии.
    # Для новой гипотезы декодера такой перенос не разрешён автоматически.
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    version = metadata.get("geometry_version", "legacy_v1")
    if version not in GEOMETRY_VERSIONS:
        raise ValueError(f"Неизвестная геометрия кэша {path}: {version}")
    source = metadata.get("source", path.stem)
    if source.startswith("flyvis") and version == "resize_v2" and not allow_unvalidated_resize:
        raise RuntimeError(f"{path}: resize_v2 не совместим с прежней калибровкой; "
                           "для явной диагностики задайте allow_unvalidated_resize=True")
    with np.load(path) as data:
        return data["points"], data["vectors"] * float(gain)


def scalabel_frame(video_id: str, truth_frame: dict, output: np.ndarray) -> dict:
    labels = []
    for row in output:
        x1, y1, x2, y2, track_id, score, class_id = row[:7]
        labels.append(
            {
                "id": str(int(track_id)),
                "category": CLASS_NAMES[int(class_id)],
                "score": float(score),
                "box2d": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
            }
        )
    return {
        "videoName": video_id,
        "name": truth_frame["name"],
        "frameIndex": truth_frame["frameIndex"],
        "labels": labels,
    }


def tracker_spec(name: str) -> tuple[str | None, float, bool]:
    if name == "constant_velocity":
        return None, 0.0, True
    if name in {"dis", "farneback"}:
        return name, 1.0, False
    if name.endswith("_calibrated"):
        source = name.removesuffix("_calibrated")
        return source, DEFAULT_GAINS[source], False
    if name.endswith("_legacy"):
        return name.removesuffix("_legacy"), LEGACY_GAIN, False
    raise ValueError(f"Неизвестный трекер: {name}")


def run_tracker(
    name: str,
    detections: dict,
    truth: list[dict],
    matches: list[dict],
    motion_dir: Path,
) -> tuple[list[dict], dict]:
    from ultralytics.trackers.byte_tracker import BYTETracker

    anchor_by_raw = {
        int(row["video_frame_index"]): index for index, row in enumerate(matches)
    }
    frame_count = int(detections["frame_count"])
    association_seconds = []
    rows = []
    if name == "bytetrack":
        tracker = BYTETracker(make_tracker_args())
        points = vectors = None
    else:
        source, gain, fused = tracker_spec(name)
        config = FlyVisTrackerConfig(update_velocity_from_matches=fused)
        tracker = FlyVisTracker(config)
        if source is None:
            points = vectors = None
        else:
            points, vectors = load_motion_cache(motion_dir / f"{source}.npz", gain)
            if len(vectors) != frame_count:
                raise RuntimeError(f"{source}: число полей не совпало с видео")

    for frame_index in range(frame_count):
        xyxy, confidence, classes = detections_for_frame(detections, frame_index)
        started = time.perf_counter()
        if name == "bytetrack":
            output = tracker.update(_DetSlice(xyxy, confidence, classes))
        else:
            field = (
                MotionField.empty()
                if vectors is None
                else MotionField(points, vectors[frame_index])
            )
            output = tracker.update(
                xyxy, confidence, classes, motion_field=field
            )
        association_seconds.append(time.perf_counter() - started)
        if frame_index in anchor_by_raw:
            reference_index = anchor_by_raw[frame_index]
            rows.append(scalabel_frame(truth[0]["videoName"], truth[reference_index], output))
    if len(rows) != len(truth):
        raise RuntimeError(f"{name}: получено {len(rows)} размеченных кадров")
    return rows, {"association": distribution(association_seconds)}


def truth_tracks(frame: dict) -> dict[tuple[str, str], np.ndarray]:
    tracks = {}
    for label in frame.get("labels") or []:
        box = label.get("box2d")
        if box is None:
            continue
        tracks[(label["category"], str(label["id"]))] = np.asarray(
            [box["x1"], box["y1"], box["x2"], box["y2"]], np.float32
        )
    return tracks


def motion_confidence_reliability(
    confidence: list[float] | np.ndarray,
    errors: list[float] | np.ndarray,
    normalized_errors: list[float] | np.ndarray,
) -> dict:
    """Проверить, ранжирует ли эвристика уверенности точные движения выше.

    Значение confidence не трактуется как вероятность. Для проверки сохраняются
    фиксированные интервалы, корреляция с ошибкой и риск при выборе наиболее
    уверенных 25/50/75/100 процентов сегментов.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    normalized = np.asarray(normalized_errors, dtype=np.float64)
    if not (len(confidence) == len(errors) == len(normalized)):
        raise ValueError("Уверенность и ошибки должны иметь одинаковую длину")
    if not len(confidence):
        return {
            "definition": "segment mean confidence versus open-loop endpoint error",
            "segments": 0,
            "confidence_error_pearson": None,
            "confidence_normalized_error_pearson": None,
            "bins": [],
            "top_confidence_fractions": {},
        }
    if (
        not np.isfinite(confidence).all()
        or not np.isfinite(errors).all()
        or not np.isfinite(normalized).all()
    ):
        raise ValueError("Уверенность и ошибки должны быть конечными")

    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    bins = []
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & (
            confidence <= upper if index == len(edges) - 2 else confidence < upper
        )
        if not np.any(mask):
            continue
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "mean_endpoint_error_pixels": float(errors[mask].mean()),
                "mean_normalized_error": float(normalized[mask].mean()),
                "acceptable_rate_norm_0.10": float(np.mean(normalized[mask] <= 0.10)),
                "acceptable_rate_norm_0.25": float(np.mean(normalized[mask] <= 0.25)),
                "acceptable_rate_norm_0.50": float(np.mean(normalized[mask] <= 0.50)),
            }
        )

    order = np.argsort(-confidence, kind="stable")
    selective = {}
    for fraction in (0.25, 0.50, 0.75, 1.00):
        count = max(1, int(np.ceil(len(order) * fraction)))
        chosen = order[:count]
        selective[f"{fraction:.2f}"] = {
            "count": count,
            "minimum_confidence": float(confidence[chosen].min()),
            "mean_endpoint_error_pixels": float(errors[chosen].mean()),
            "mean_normalized_error": float(normalized[chosen].mean()),
            "acceptable_rate_norm_0.25": float(np.mean(normalized[chosen] <= 0.25)),
        }

    return {
        "definition": "segment mean confidence versus open-loop endpoint error",
        "segments": int(len(confidence)),
        "confidence_error_pearson": correlation(confidence, errors),
        "confidence_normalized_error_pearson": correlation(confidence, normalized),
        "bins": bins,
        "top_confidence_fractions": selective,
    }


def endpoint_motion_audit(
    truth: list[dict],
    matches: list[dict],
    points: np.ndarray,
    vectors: np.ndarray,
) -> dict:
    errors, normalized, target_lengths, predicted_lengths, directions = [], [], [], [], []
    confidences = []
    segment_confidences = []
    for reference_index in range(1, len(truth)):
        previous = truth_tracks(truth[reference_index - 1])
        current = truth_tracks(truth[reference_index])
        common = sorted(previous.keys() & current.keys())
        raw_start = int(matches[reference_index - 1]["video_frame_index"])
        raw_stop = int(matches[reference_index]["video_frame_index"])
        for key in common:
            box = previous[key].copy()
            start_center = (box[:2] + box[2:]) * 0.5
            velocity = np.zeros(2, dtype=np.float32)
            interval_confidences = []
            for raw_index in range(raw_start + 1, raw_stop + 1):
                estimate = MotionField(points, vectors[raw_index]).estimate_box(
                    box, context_scale=2.0, min_samples=3
                )
                confidence = float(np.clip(estimate.confidence, 0.0, 1.0))
                velocity = confidence * estimate.vector_xy + (1.0 - confidence) * velocity
                box = shift_box(box, velocity)
                confidences.append(confidence)
                interval_confidences.append(confidence)
            predicted_center = (box[:2] + box[2:]) * 0.5
            target_center = (current[key][:2] + current[key][2:]) * 0.5
            predicted_displacement = predicted_center - start_center
            target_displacement = target_center - start_center
            error = float(np.linalg.norm(predicted_center - target_center))
            diagonal = float(np.linalg.norm(previous[key][2:] - previous[key][:2]))
            errors.append(error)
            normalized.append(error / max(diagonal, 1.0))
            segment_confidences.append(
                float(np.mean(interval_confidences)) if interval_confidences else 0.0
            )
            target_lengths.append(float(np.linalg.norm(target_displacement)))
            predicted_lengths.append(float(np.linalg.norm(predicted_displacement)))
            if np.linalg.norm(target_displacement) >= 0.5:
                directions.append(float(np.dot(predicted_displacement, target_displacement) > 0))
    values = np.asarray(errors, dtype=np.float64)
    return {
        "definition": (
            "open-loop endpoint error accumulated over raw 30 Hz frames between "
            "two labeled BDD100K frames; ground truth is not used inside inference"
        ),
        "samples": len(errors),
        "mean_endpoint_error_pixels": float(values.mean()),
        "median_endpoint_error_pixels": float(np.median(values)),
        "p90_endpoint_error_pixels": float(np.quantile(values, 0.90)),
        "mean_error_over_start_box_diagonal": float(np.mean(normalized)),
        "direction_accuracy": float(np.mean(directions)) if directions else None,
        "mean_target_displacement_pixels": float(np.mean(target_lengths)),
        "mean_predicted_displacement_pixels": float(np.mean(predicted_lengths)),
        "mean_motion_confidence": float(np.mean(confidences)),
        "confidence_reliability": motion_confidence_reliability(
            segment_confidences, errors, normalized
        ),
    }


def main() -> int:
    args = parse_args()
    if args.flyvis_geometry_version == "resize_v2" and not args.motion_only:
        if any(name.startswith("flyvis") for name in args.trackers):
            raise ValueError("Трекеры FlyVis используют прежние эмпирические gains: resize_v2 "
                             "допустим только как --motion-only диагностика; "
                             "для стенда используйте pixel_centers_v2 либо явный legacy_v1")
    video, truth_path, images = resolve_paths(args)
    args.output.mkdir(parents=True, exist_ok=True)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    matches, video_metadata = build_alignment(
        video, images, args.output / "alignment.json", args.search_radius
    )
    if len(truth) != len(matches):
        raise RuntimeError("Число кадров разметки и сопоставлений не совпало")
    fps = float(video_metadata["fps"])

    detection_path = args.output / "detections" / "raw_video.npz"
    @cache
    def inference_device():
        from ultralytics.utils.torch_utils import select_device
        return select_device(args.device, verbose=False)

    detector_meta = build_detection_cache(
        video,
        args.weights,
        detection_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        force=args.force_detections,
        resolve_device=inference_device,
    )
    detections = load_detection_cache(detection_path)
    if int(detections["frame_count"]) != int(video_metadata["decoded_frame_count"]):
        raise RuntimeError("Число декодированных кадров изменилось между этапами")

    motion_meta = {}
    for source in args.motion_sources:
        motion_meta[source] = build_motion_cache(
            source,
            video,
            args.output / "motion",
            fps=fps,
            max_dt=args.max_dt,
            force=args.force_motion,
            geometry_version=args.flyvis_geometry_version,
            device=args.device,
            resolve_device=inference_device,
        )

    if args.motion_only:
        print(json.dumps(motion_meta, ensure_ascii=False, indent=2), flush=True)
        return 0

    tracker_results = {}
    required_sources = set()
    for name in args.trackers:
        if name != "bytetrack":
            source, _, _ = tracker_spec(name)
            if source is not None:
                required_sources.add(source)
    missing = required_sources - set(args.motion_sources)
    if missing:
        raise ValueError(f"Не построены поля движения: {sorted(missing)}")

    for name in args.trackers:
        print(f"[TRACKER] {name}", flush=True)
        rows, timing = run_tracker(
            name, detections, truth, matches, args.output / "motion"
        )
        result_dir = args.output / "results" / name
        save_json(result_dir / f"{args.video_id}.json", rows)
        metrics = None
        if not args.skip_evaluation:
            metrics = evaluate_hota(result_dir, [args.video_id])
        tracker_results[name] = {"timing": timing, "metrics": metrics}

    motion_audits = {}
    for name in args.trackers:
        if name == "bytetrack":
            continue
        source, gain, _ = tracker_spec(name)
        if source is None:
            continue
        points, vectors = load_motion_cache(
            args.output / "motion" / f"{source}.npz", gain
        )
        motion_audits[name] = endpoint_motion_audit(truth, matches, points, vectors)

    summary = {
        "protocol": {
            "video_id": args.video_id,
            "video_path": str(video.resolve()),
            "video_sha256": sha256(video),
            "source_fps": fps,
            "raw_frames": int(video_metadata["decoded_frame_count"]),
            "labeled_frames": len(truth),
            "detector": "YOLO26 baseline seed42 best.pt",
            "weights": str(args.weights.resolve()),
            "imgsz": args.imgsz,
            "detector_conf": args.conf,
            "detector_iou": args.iou,
            "note": (
                "all trackers consume identical detections on every raw frame; "
                "outputs are sampled at content-aligned labeled frames only"
            ),
        },
        "alignment": json.loads(
            (args.output / "alignment.json").read_text(encoding="utf-8")
        ),
        "detector": detector_meta,
        "motion_sources": motion_meta,
        "trackers": tracker_results,
        "motion_endpoint_audits": motion_audits,
    }
    save_json(args.output / "summary.json", summary)
    compact = {
        name: {
            **(result["metrics"] or {}),
            "association_fps": result["timing"]["association"]["fps_from_mean"],
        }
        for name, result in tracker_results.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] {args.output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
