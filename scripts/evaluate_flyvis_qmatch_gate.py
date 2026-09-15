"""Дешёвый гейт условной полезности FlyVis для вероятности правильной пары.

GT используется только для построения целевой метки: принадлежат ли две
детекции одному постоянному идентификатору. Все признаки доступны во время
инференса и не содержат координат или атрибутов GT.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from scripts.run_bdd_video_rate_experiment import (  # noqa: E402
    DEFAULT_GAINS,
    detections_for_frame,
    load_detection_cache,
    load_motion_cache,
    save_json,
)
from tracking.flyvis_tracker import MotionField, box_iou, shift_box  # noqa: E402


TRAIN_VIDEO_IDS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
)
TEST_VIDEO_IDS = (
    "b232c7c9-d251d9ee",
    "b1d22449-15fb948f",
    "b251064f-e7a165fd",
)
CLASS_TO_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}
BASE_FEATURES = (
    "previous_confidence",
    "current_confidence",
    "pair_iou",
    "pair_center_distance",
    "pair_size_log_change",
    *(f"class_{class_id}" for class_id in sorted(CLASS_NAMES)),
)
MOTION_SOURCES = {
    "dis": ("dis_quarter", 1.0),
    "flyvis_full": ("flyvis_full", DEFAULT_GAINS["flyvis_full"]),
    "flyvis_tiled": ("flyvis_tiled", DEFAULT_GAINS["flyvis_tiled"]),
}
MOTION_FEATURES = tuple(
    feature
    for source in MOTION_SOURCES
    for feature in (
        f"{source}_predicted_iou",
        f"{source}_center_distance",
        f"{source}_mean_internal_weight",
    )
)
FEATURE_NAMES = BASE_FEATURES + MOTION_FEATURES
FEATURE_GROUPS = {
    "base": BASE_FEATURES,
    "base_dis": BASE_FEATURES + MOTION_FEATURES[0:3],
    "base_flyvis_full": BASE_FEATURES + MOTION_FEATURES[3:6],
    "base_flyvis_tiled": BASE_FEATURES + MOTION_FEATURES[6:9],
    "base_dis_flyvis_tiled": (
        BASE_FEATURES + MOTION_FEATURES[0:3] + MOTION_FEATURES[6:9]
    ),
}


@dataclass(frozen=True)
class MatchedDetection:
    detection_index: int
    gt_id: str | None
    class_id: int
    box: np.ndarray
    confidence: float
    occluded: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate"
    )
    parser.add_argument(
        "--flyvis-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "flyvis_stress_gate",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "qmatch_flyvis_gate",
    )
    parser.add_argument("--detection-iou", type=float, default=0.5)
    parser.add_argument(
        "--pairs-only",
        action="store_true",
        help="только построить и сохранить pairs.npz без обучения моделей",
    )
    parser.add_argument("--candidate-distance", type=float, default=2.5)
    parser.add_argument(
        "--candidate-set",
        choices=("matched_only", "all_current"),
        default="matched_only",
        help=(
            "matched_only воспроизводит условный гейт среди GT-сопоставленных "
            "детекций; all_current добавляет несопоставленные текущие детекции "
            "как отрицательные кандидаты"
        ),
    )
    parser.add_argument(
        "--temporal-mode",
        choices=("reference", "adjacent_pseudo"),
        default="reference",
        help=(
            "reference соединяет размеченные кадры; adjacent_pseudo строит "
            "соседние 30-герцовые переходы по консервативной интерполяции GT"
        ),
    )
    return parser.parse_args()


def truth_arrays(frame: dict) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    boxes, classes, identities, occluded = [], [], [], []
    for label in frame.get("labels", []):
        attributes = label.get("attributes", {})
        if attributes.get("crowd", False) or label.get("category") not in CLASS_TO_ID:
            continue
        box = label.get("box2d")
        if box is None:
            continue
        boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
        classes.append(CLASS_TO_ID[label["category"]])
        identities.append(str(label["id"]))
        occluded.append(bool(attributes.get("occluded", False)))
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64),
        identities,
        np.asarray(occluded, dtype=bool),
    )


def interpolation_label_is_usable(label: dict) -> bool:
    attributes = label.get("attributes", {})
    return (
        label.get("category") in CLASS_TO_ID
        and label.get("box2d") is not None
        and not any(
            bool(attributes.get(name, False))
            for name in ("crowd", "occluded", "truncated")
        )
    )


def interpolate_truth_frame(left_frame: dict, right_frame: dict, alpha: float) -> dict:
    """Линейная псевдоразметка общих чистых объектов между опорными кадрами."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Коэффициент интерполяции вне [0, 1]: {alpha}")
    left = {
        str(label["id"]): label
        for label in left_frame.get("labels", [])
        if interpolation_label_is_usable(label)
    }
    right = {
        str(label["id"]): label
        for label in right_frame.get("labels", [])
        if interpolation_label_is_usable(label)
    }
    labels = []
    for identity in sorted(set(left) & set(right)):
        left_label = left[identity]
        right_label = right[identity]
        if left_label["category"] != right_label["category"]:
            continue
        left_box = left_label["box2d"]
        right_box = right_label["box2d"]
        box = {
            key: (1.0 - alpha) * float(left_box[key])
            + alpha * float(right_box[key])
            for key in ("x1", "y1", "x2", "y2")
        }
        labels.append(
            {
                "id": identity,
                "category": left_label["category"],
                "attributes": {
                    "crowd": False,
                    "occluded": False,
                    "truncated": False,
                    "pseudo_interpolated": 0.0 < alpha < 1.0,
                },
                "box2d": box,
            }
        )
    return {"labels": labels}


def temporal_transitions(
    truth: list[dict], alignment: list[dict], temporal_mode: str
):
    """Выдаёт индексы исходных кадров и соответствующие целевые рамки."""
    if temporal_mode not in {"reference", "adjacent_pseudo"}:
        raise ValueError(f"Неизвестный временной режим: {temporal_mode}")
    for reference_index in range(1, len(truth)):
        raw_start = int(alignment[reference_index - 1]["video_frame_index"])
        raw_stop = int(alignment[reference_index]["video_frame_index"])
        if raw_stop <= raw_start:
            continue
        if temporal_mode == "reference":
            yield raw_start, raw_stop, truth[reference_index - 1], truth[reference_index]
            continue
        raw_span = raw_stop - raw_start
        for adjacent_stop in range(raw_start + 1, raw_stop + 1):
            previous_alpha = (adjacent_stop - 1 - raw_start) / raw_span
            current_alpha = (adjacent_stop - raw_start) / raw_span
            yield (
                adjacent_stop - 1,
                adjacent_stop,
                interpolate_truth_frame(
                    truth[reference_index - 1],
                    truth[reference_index],
                    previous_alpha,
                ),
                interpolate_truth_frame(
                    truth[reference_index - 1],
                    truth[reference_index],
                    current_alpha,
                ),
            )


def match_detections_to_truth(
    boxes: np.ndarray,
    confidence: np.ndarray,
    classes: np.ndarray,
    truth_frame: dict,
    *,
    min_iou: float,
) -> list[MatchedDetection]:
    truth_boxes, truth_classes, truth_ids, truth_occluded = truth_arrays(truth_frame)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    if not len(boxes) or not len(truth_boxes):
        return []
    overlaps = box_iou(boxes, truth_boxes)
    compatible = classes[:, None] == truth_classes[None, :]
    costs = np.where(compatible, 1.0 - overlaps, 1e6)
    detection_indices, truth_indices = linear_sum_assignment(costs)
    matched = []
    for detection_index, truth_index in zip(detection_indices, truth_indices):
        if not compatible[detection_index, truth_index]:
            continue
        if overlaps[detection_index, truth_index] < min_iou:
            continue
        matched.append(
            MatchedDetection(
                detection_index=int(detection_index),
                gt_id=truth_ids[truth_index],
                class_id=int(classes[detection_index]),
                box=boxes[detection_index].copy(),
                confidence=float(confidence[detection_index]),
                occluded=bool(truth_occluded[truth_index]),
            )
        )
    return matched


def detections_with_truth_status(
    boxes: np.ndarray,
    confidence: np.ndarray,
    classes: np.ndarray,
    truth_frame: dict,
    *,
    min_iou: float,
) -> list[MatchedDetection]:
    """Возвращает все детекции, помечая несопоставленные как кандидаты без GT."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.int64).reshape(-1)
    matched = {
        detection.detection_index: detection
        for detection in match_detections_to_truth(
            boxes,
            confidence,
            classes,
            truth_frame,
            min_iou=min_iou,
        )
    }
    detections = []
    for detection_index, (box, score, class_id) in enumerate(
        zip(boxes, confidence, classes)
    ):
        if detection_index in matched:
            detections.append(matched[detection_index])
            continue
        detections.append(
            MatchedDetection(
                detection_index=detection_index,
                gt_id=None,
                class_id=int(class_id),
                box=box.copy(),
                confidence=float(score),
                occluded=False,
            )
        )
    return detections


def box_pair_geometry(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    left = np.asarray(left, dtype=np.float32).reshape(4)
    right = np.asarray(right, dtype=np.float32).reshape(4)
    overlap = float(box_iou(left[None, :], right[None, :])[0, 0])
    left_center = (left[:2] + left[2:]) * 0.5
    right_center = (right[:2] + right[2:]) * 0.5
    left_wh = np.maximum(left[2:] - left[:2], 1.0)
    right_wh = np.maximum(right[2:] - right[:2], 1.0)
    scale = max(float(np.sqrt(np.prod(left_wh))), 1.0)
    center_distance = float(np.linalg.norm(right_center - left_center) / scale)
    size_log_change = float(np.mean(np.abs(np.log(right_wh / left_wh))))
    return overlap, center_distance, size_log_change


def propagate_box(
    box: np.ndarray,
    points: np.ndarray,
    vectors: np.ndarray,
    raw_start: int,
    raw_stop: int,
) -> tuple[np.ndarray, float]:
    predicted = np.asarray(box, dtype=np.float32).reshape(4).copy()
    velocity = np.zeros(2, dtype=np.float32)
    weights = []
    for raw_index in range(raw_start + 1, raw_stop + 1):
        estimate = MotionField(points, vectors[raw_index]).estimate_box(
            predicted, context_scale=2.0, min_samples=3
        )
        weight = float(np.clip(estimate.confidence, 0.0, 1.0))
        velocity = weight * estimate.vector_xy + (1.0 - weight) * velocity
        predicted = shift_box(predicted, velocity)
        weights.append(weight)
    return predicted, float(np.mean(weights)) if weights else 0.0


def motion_predictions(
    detections: list[MatchedDetection],
    caches: dict[str, tuple[np.ndarray, np.ndarray]],
    raw_start: int,
    raw_stop: int,
) -> dict[tuple[int, str], tuple[np.ndarray, float]]:
    predictions = {}
    for detection in detections:
        for source, (points, vectors) in caches.items():
            predictions[(detection.detection_index, source)] = propagate_box(
                detection.box, points, vectors, raw_start, raw_stop
            )
    return predictions


def pair_features(
    previous: MatchedDetection,
    current: MatchedDetection,
    predictions: dict[tuple[int, str], tuple[np.ndarray, float]],
) -> np.ndarray:
    overlap, center_distance, size_change = box_pair_geometry(
        previous.box, current.box
    )
    values = [
        previous.confidence,
        current.confidence,
        overlap,
        center_distance,
        size_change,
    ]
    values.extend(float(previous.class_id == class_id) for class_id in sorted(CLASS_NAMES))
    for source in MOTION_SOURCES:
        predicted, internal_weight = predictions[(previous.detection_index, source)]
        predicted_iou, predicted_distance, _ = box_pair_geometry(predicted, current.box)
        values.extend((predicted_iou, predicted_distance, internal_weight))
    return np.asarray(values, dtype=np.float32)


def pair_target(previous: MatchedDetection, current: MatchedDetection) -> int:
    """Правильная пара требует известного и совпадающего GT-идентификатора."""
    return int(
        previous.gt_id is not None
        and current.gt_id is not None
        and previous.gt_id == current.gt_id
    )


def build_video_pairs(
    video_id: str,
    root: Path,
    flyvis_root: Path,
    labels_root: Path,
    *,
    detection_iou: float,
    candidate_distance: float,
    candidate_set: str = "matched_only",
    temporal_mode: str = "reference",
) -> dict:
    experiment = root / video_id
    detections = load_detection_cache(experiment / "detections" / "raw_video.npz")
    truth = json.loads((labels_root / f"{video_id}.json").read_text(encoding="utf-8"))
    alignment = json.loads(
        (experiment / "alignment.json").read_text(encoding="utf-8")
    )["matches"]
    caches = {}
    for source, (cache_name, gain) in MOTION_SOURCES.items():
        motion_root = root if source == "dis" else flyvis_root
        caches[source] = load_motion_cache(
            motion_root / video_id / "motion" / f"{cache_name}.npz", gain
        )

    if candidate_set not in {"matched_only", "all_current"}:
        raise ValueError(f"Неизвестный набор кандидатов: {candidate_set}")
    features, targets, previous_occluded, current_occluded = [], [], [], []
    current_truth_matched = []
    pair_frames, previous_boxes, current_boxes = [], [], []
    previous_index, current_index = [], []
    positive_eligible = 0
    positive_in_gate = 0
    transition_count = 0
    for raw_start, raw_stop, previous_truth, current_truth in temporal_transitions(
        truth, alignment, temporal_mode
    ):
        transition_count += 1
        previous_arrays = detections_for_frame(detections, raw_start)
        current_arrays = detections_for_frame(detections, raw_stop)
        previous = match_detections_to_truth(
            *previous_arrays, previous_truth, min_iou=detection_iou
        )
        if candidate_set == "all_current":
            current = detections_with_truth_status(
                *current_arrays, current_truth, min_iou=detection_iou
            )
        else:
            current = match_detections_to_truth(
                *current_arrays, current_truth, min_iou=detection_iou
            )
        if not previous or not current:
            continue
        current_ids = {
            detection.gt_id for detection in current if detection.gt_id is not None
        }
        positive_eligible += sum(detection.gt_id in current_ids for detection in previous)
        predictions = motion_predictions(
            previous, caches, raw_start, raw_stop
        )
        for left in previous:
            for right in current:
                if left.class_id != right.class_id:
                    continue
                overlap, distance, _ = box_pair_geometry(left.box, right.box)
                if distance > candidate_distance and overlap < 0.01:
                    continue
                target = pair_target(left, right)
                positive_in_gate += target
                features.append(pair_features(left, right, predictions))
                targets.append(target)
                previous_occluded.append(left.occluded)
                current_occluded.append(right.occluded)
                current_truth_matched.append(right.gt_id is not None)
                pair_frames.append((raw_start, raw_stop))
                previous_boxes.append(left.box)
                current_boxes.append(right.box)
                previous_index.append(left.detection_index)
                current_index.append(right.detection_index)
    if not features:
        raise RuntimeError(f"Для {video_id} не построено ни одной пары")
    return {
        "features": np.stack(features),
        "targets": np.asarray(targets, dtype=np.int8),
        "previous_occluded": np.asarray(previous_occluded, dtype=bool),
        "current_occluded": np.asarray(current_occluded, dtype=bool),
        "current_truth_matched": np.asarray(current_truth_matched, dtype=bool),
        "pair_frames": np.asarray(pair_frames, dtype=np.int32),
        "previous_boxes": np.stack(previous_boxes).astype(np.float32),
        "current_boxes": np.stack(current_boxes).astype(np.float32),
        "previous_index": np.asarray(previous_index, dtype=np.int32),
        "current_index": np.asarray(current_index, dtype=np.int32),
        "positive_eligible": int(positive_eligible),
        "positive_in_gate": int(positive_in_gate),
        "transitions": transition_count,
    }


def f1_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probability)
    if not len(thresholds):
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.argmax(f1))])


def metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    true_positive = int(np.sum(predicted & (target == 1)))
    false_positive = int(np.sum(predicted & (target == 0)))
    false_negative = int(np.sum(~predicted & (target == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "samples": int(len(target)),
        "positives": int(np.sum(target)),
        "average_precision": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "threshold_from_train": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
    }


def decision_stratum_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float
) -> dict:
    """Описывает допуск внутри заранее определённой группы пар."""
    target = np.asarray(target, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    accepted = probability >= threshold
    return {
        "samples": int(len(target)),
        "positives": int(np.sum(target)),
        "mean_probability": float(np.mean(probability)) if len(target) else None,
        "brier": (
            float(np.mean((probability - target) ** 2)) if len(target) else None
        ),
        "accepted": int(np.sum(accepted)),
        "acceptance_rate": float(np.mean(accepted)) if len(target) else None,
    }


def serialize_linear_probability_model(model, feature_names: tuple[str, ...]) -> dict:
    """Сохраняет точные параметры StandardScaler и логистической регрессии."""
    scaler = model.steps[0][1]
    classifier = model.steps[1][1]
    return {
        "kind": "standardized_logistic_regression",
        "features": list(feature_names),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coefficient": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
    }


def fit_and_evaluate(
    features: np.ndarray,
    targets: np.ndarray,
    video_indices: np.ndarray,
    current_truth_matched: np.ndarray,
    feature_names: tuple[str, ...],
    train_count: int,
    video_ids: tuple[str, ...],
    all_feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict:
    indices = [all_feature_names.index(name) for name in feature_names]
    train = video_indices < train_count
    test = ~train
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    )
    model.fit(features[train][:, indices], targets[train])
    train_probability = model.predict_proba(features[train][:, indices])[:, 1]
    threshold = f1_threshold(targets[train], train_probability)
    test_probability = model.predict_proba(features[test][:, indices])[:, 1]
    result = {
        "features": list(feature_names),
        "model": serialize_linear_probability_model(model, feature_names),
        "train": metrics(targets[train], train_probability, threshold),
        "test": metrics(targets[test], test_probability, threshold),
        "test_strata": {},
        "test_by_video": {},
    }
    strata = {
        "positive_pairs": targets[test] == 1,
        "wrong_identity_negative_pairs": (targets[test] == 0)
        & current_truth_matched[test],
        "unmatched_detection_negative_pairs": ~current_truth_matched[test],
    }
    for name, mask in strata.items():
        result["test_strata"][name] = decision_stratum_metrics(
            targets[test][mask], test_probability[mask], threshold
        )
    for video_index in range(train_count, len(video_ids)):
        mask = video_indices == video_index
        result["test_by_video"][video_ids[video_index]] = metrics(
            targets[mask],
            model.predict_proba(features[mask][:, indices])[:, 1],
            threshold,
        )
    return result


def paired_video_bootstrap(
    reference: dict,
    candidate: dict,
    metric_name: str,
    *,
    repetitions: int = 5000,
    seed: int = 42,
) -> dict:
    """Парный бутстрап среднего изменения метрики с единицей «видео»."""
    video_ids = tuple(reference["test_by_video"])
    if tuple(candidate["test_by_video"]) != video_ids:
        raise ValueError("Наборы и порядок тестовых видео не совпали")
    differences = np.asarray(
        [
            candidate["test_by_video"][video_id][metric_name]
            - reference["test_by_video"][video_id][metric_name]
            for video_id in video_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = differences[
        rng.integers(0, len(differences), size=(repetitions, len(differences)))
    ].mean(axis=1)
    return {
        "metric": metric_name,
        "unit": "video",
        "video_count": len(video_ids),
        "repetitions": repetitions,
        "per_video_differences": {
            video_id: float(difference)
            for video_id, difference in zip(video_ids, differences)
        },
        "mean_difference": float(differences.mean()),
        "ci95_percentile": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "fraction_positive": float(np.mean(samples > 0.0)),
    }


def main() -> int:
    args = parse_args()
    video_ids = TRAIN_VIDEO_IDS + TEST_VIDEO_IDS
    built = []
    for index, video_id in enumerate(video_ids, 1):
        print(f"[PAIR] {index}/{len(video_ids)} {video_id}", flush=True)
        built.append(
            build_video_pairs(
                video_id,
                args.root,
                args.flyvis_root,
                args.labels,
                detection_iou=args.detection_iou,
                candidate_distance=args.candidate_distance,
                candidate_set=args.candidate_set,
                temporal_mode=args.temporal_mode,
            )
        )

    features = np.concatenate([row["features"] for row in built])
    targets = np.concatenate([row["targets"] for row in built])
    video_indices = np.concatenate(
        [np.full(len(row["targets"]), index, dtype=np.int16) for index, row in enumerate(built)]
    )
    previous_occluded = np.concatenate([row["previous_occluded"] for row in built])
    current_occluded = np.concatenate([row["current_occluded"] for row in built])
    current_truth_matched = np.concatenate(
        [row["current_truth_matched"] for row in built]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "pairs.npz",
        features=features.astype(np.float32),
        targets=targets,
        video_indices=video_indices,
        previous_occluded=previous_occluded,
        current_occluded=current_occluded,
        current_truth_matched=current_truth_matched,
        pair_frames=np.concatenate([row["pair_frames"] for row in built]),
        previous_boxes=np.concatenate([row["previous_boxes"] for row in built]),
        current_boxes=np.concatenate([row["current_boxes"] for row in built]),
        previous_index=np.concatenate([row["previous_index"] for row in built]),
        current_index=np.concatenate([row["current_index"] for row in built]),
        feature_names=np.asarray(FEATURE_NAMES),
        video_ids=np.asarray(video_ids),
    )
    if args.pairs_only:
        print(f"[OK] pairs only: {args.output / 'pairs.npz'}", flush=True)
        return 0

    results = {
        name: fit_and_evaluate(
            features,
            targets,
            video_indices,
            current_truth_matched,
            feature_names,
            len(TRAIN_VIDEO_IDS),
            video_ids,
        )
        for name, feature_names in FEATURE_GROUPS.items()
    }
    bootstrap = {
        metric_name: paired_video_bootstrap(
            results["base_dis"],
            results["base_dis_flyvis_tiled"],
            metric_name,
        )
        for metric_name in ("average_precision", "brier")
    }
    report = {
        "protocol": {
            "target": "same persistent GT identity for two detector boxes",
            "gt_usage": "target construction and occlusion strata only; no GT field is a model feature",
            "detector": "fixed YOLO26 baseline seed42 best.pt at confidence 0.01",
            "train_video_ids": list(TRAIN_VIDEO_IDS),
            "test_video_ids": list(TEST_VIDEO_IDS),
            "detection_truth_iou": args.detection_iou,
            "candidate_center_distance": args.candidate_distance,
            "candidate_set": args.candidate_set,
            "temporal_mode": args.temporal_mode,
            "pseudo_label_policy": (
                "linear interpolation of identities present at both endpoints; "
                "crowd, occluded, truncated, missing, and class-changing tracks excluded"
                if args.temporal_mode == "adjacent_pseudo"
                else None
            ),
            "dis_source": "DIS at quarter-resolution processing scale",
            "detection_and_dis_root": str(args.root),
            "flyvis_root": str(args.flyvis_root),
            "flyvis_gains": DEFAULT_GAINS,
        },
        "dataset": {
            "samples": int(len(targets)),
            "positives": int(np.sum(targets)),
            "negatives": int(np.sum(targets == 0)),
            "per_video": {
                video_id: {
                    "samples": int(len(row["targets"])),
                    "transitions": int(row["transitions"]),
                    "positives": int(np.sum(row["targets"])),
                    "wrong_identity_negative_pairs": int(
                        np.sum((row["targets"] == 0) & row["current_truth_matched"])
                    ),
                    "unmatched_detection_negative_pairs": int(
                        np.sum(~row["current_truth_matched"])
                    ),
                    "positive_candidate_coverage": float(
                        row["positive_in_gate"] / max(row["positive_eligible"], 1)
                    ),
                }
                for video_id, row in zip(video_ids, built)
            },
        },
        "feature_groups": results,
        "decision": {
            "primary_comparison": "base_dis_flyvis_tiled versus base_dis on held-out videos",
            "acceptance": (
                "FlyVis passes only if held-out average precision increases and Brier score does not worsen; "
                "tracker-level HOTA/AssA/IDSw confirmation remains mandatory"
            ),
            "paired_video_bootstrap": bootstrap,
        },
    }
    save_json(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] {args.output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
