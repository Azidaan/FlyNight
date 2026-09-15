"""Гейт считывания внутренней активности FlyVis для q_match (карточка flyvis_internal_readout_v1).

Берёт пары «прошлая детекция — текущая детекция» из pairs.npz гейта q_match
(те же пары, метки и разбиение), добавляет признаки считывания из кэша
активности (T4/T5 и все 34 типа), контроль shuffled и сравнивает вложенные
логистические модели на отложенных видео парным бутстрапом по видео.
Разметка не читается: метки уже лежат в pairs.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_flyvis_qmatch_gate import (  # noqa: E402
    BASE_FEATURES,
    FEATURE_NAMES,
    MOTION_FEATURES,
    TEST_VIDEO_IDS,
    TRAIN_VIDEO_IDS,
    fit_and_evaluate,
    paired_video_bootstrap,
)
from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402
from tracking.flyvis_activity import (  # noqa: E402
    DIRECTION_SUFFIXES,
    OUTPUT_CELL_TYPES,
    box_readout_all_types,
    box_readout_t4t5,
    column_spacing,
    load_activity_cache,
)

DIS_FEATURES = MOTION_FEATURES[0:3]
DECODED_TILED_FEATURES = MOTION_FEATURES[6:9]

_BOX_T4T5 = tuple(
    f"{prefix}_{name}"
    for prefix in ("inside", "ring")
    for name in (
        *(f"energy_{s}" for s in DIRECTION_SUFFIXES),
        "energy_on", "energy_off", "energy_total", "dominance", "entropy", "net_x", "net_y",
    )
) + ("contrast", "coverage")
READOUT_T4T5_FEATURES = tuple(f"prev_{n}" for n in _BOX_T4T5) + tuple(f"cur_{n}" for n in _BOX_T4T5)
READOUT_PAIR_TERMS = ("pair_net_cosine", "pair_entropy_delta", "pair_energy_delta")
READOUT_PAIR_FEATURES = READOUT_T4T5_FEATURES + READOUT_PAIR_TERMS
_BOX_ALL34 = tuple(f"{prefix}_{t}" for prefix in ("inside", "ring") for t in OUTPUT_CELL_TYPES) + ("coverage",)
READOUT_ALL34_FEATURES = tuple(f"prev_{n}" for n in _BOX_ALL34) + tuple(f"cur_{n}" for n in _BOX_ALL34)
SHUFFLED_FEATURES = tuple(f"shuffled_{n}" for n in READOUT_PAIR_FEATURES)
ALL_FEATURE_NAMES = FEATURE_NAMES + READOUT_PAIR_FEATURES + READOUT_ALL34_FEATURES + SHUFFLED_FEATURES

FEATURE_GROUPS = {
    "base_dis": BASE_FEATURES + DIS_FEATURES,
    "base_dis_decoded_tiled": BASE_FEATURES + DIS_FEATURES + DECODED_TILED_FEATURES,
    "base_dis_t4t5": BASE_FEATURES + DIS_FEATURES + READOUT_PAIR_FEATURES,
    "base_dis_all34": BASE_FEATURES + DIS_FEATURES + READOUT_ALL34_FEATURES,
    "base_dis_shuffled_t4t5": BASE_FEATURES + DIS_FEATURES + SHUFFLED_FEATURES,
    "base_dis_decoded_t4t5": BASE_FEATURES + DIS_FEATURES + DECODED_TILED_FEATURES + READOUT_PAIR_FEATURES,
    "base_t4t5": BASE_FEATURES + READOUT_PAIR_FEATURES,
}
# Заранее записанные сравнения (карточка flyvis_internal_readout_v1).
COMPARISONS = {
    "primary_t4t5_vs_base_dis": ("base_dis", "base_dis_t4t5"),
    "t4t5_vs_shuffled": ("base_dis_shuffled_t4t5", "base_dis_t4t5"),
    "t4t5_vs_decoded": ("base_dis_decoded_tiled", "base_dis_t4t5"),
    "all34_vs_base_dis": ("base_dis", "base_dis_all34"),
    "shuffled_vs_base_dis": ("base_dis", "base_dis_shuffled_t4t5"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="pairs.npz гейта q_match с рамками и кадрами пар")
    parser.add_argument("--reference-pairs", type=Path, default=None, help="pairs.npz прежнего прогона для проверки тождества пар и меток")
    parser.add_argument("--activity-root", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_activity")
    parser.add_argument("--activity-source", default="flyvis_tiled")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ring-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fovea-root", type=Path, default=None, help="runs/flyvis_fovea: режим фовеации (карточка flyvis_foveation_readout_v1)")
    parser.add_argument("--tiled-readout", type=Path, default=None, help="readout_features.npz tiled-гейта для прямого сравнения охватов")
    return parser.parse_args()


def frame_activity_matrix(frame: np.ndarray) -> np.ndarray:
    """(tiles, types, columns) -> (types, tiles*columns) в порядке points кэша."""
    frame = np.asarray(frame, dtype=np.float32)
    tiles, types, columns = frame.shape
    return frame.transpose(1, 0, 2).reshape(types, tiles * columns)


def readout_pair_features(previous_activity, current_activity, baseline, points, *, previous_box, current_box, min_margin: float, ring_scale: float = 1.5) -> np.ndarray:
    prev = box_readout_t4t5(previous_activity, baseline, points, previous_box, ring_scale=ring_scale, min_margin=min_margin)
    cur = box_readout_t4t5(current_activity, baseline, points, current_box, ring_scale=ring_scale, min_margin=min_margin)
    values = [prev[name] for name in _BOX_T4T5] + [cur[name] for name in _BOX_T4T5]
    a = np.asarray([prev["inside_net_x"], prev["inside_net_y"]])
    b = np.asarray([cur["inside_net_x"], cur["inside_net_y"]])
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    values += [cosine, abs(prev["inside_entropy"] - cur["inside_entropy"]), abs(prev["inside_energy_total"] - cur["inside_energy_total"])]
    return np.asarray(values, dtype=np.float32)


def readout_all34_features(previous_activity, current_activity, baseline, points, *, previous_box, current_box, min_margin: float, ring_scale: float = 1.5) -> np.ndarray:
    prev = box_readout_all_types(previous_activity, baseline, points, previous_box, ring_scale=ring_scale, min_margin=min_margin)
    cur = box_readout_all_types(current_activity, baseline, points, current_box, ring_scale=ring_scale, min_margin=min_margin)
    return np.asarray([prev[n] for n in _BOX_ALL34] + [cur[n] for n in _BOX_ALL34], dtype=np.float32)


def shuffle_within_video(matrix: np.ndarray, video_indices: np.ndarray, seed: int) -> np.ndarray:
    """Переставить строки внутри каждого видео (контроль shuffled), детерминированно."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(matrix, copy=True)
    for video in np.unique(video_indices):
        rows = np.flatnonzero(video_indices == video)
        shuffled[rows] = matrix[rng.permutation(rows)]
    return shuffled


def join_fovea_rows(pairs: dict, rows: dict, *, key: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Сопоставить парам (кадр t, prev i, cur j) фовеальные строки (frame, prev_detection_index, detection_index)."""
    lookup = {}
    for position, (frame, prev_index, det_index) in enumerate(zip(rows["frame"], rows["prev_detection_index"], rows["detection_index"])):
        if det_index >= 0:
            lookup[(int(frame), int(prev_index), int(det_index))] = position
    matrix = np.zeros((len(pairs["previous_index"]), width), np.float32)
    has_eye = np.zeros(len(pairs["previous_index"]), np.int8)
    for position, (frames, prev_index, cur_index) in enumerate(zip(pairs["pair_frames"], pairs["previous_index"], pairs["current_index"])):
        hit = lookup.get((int(frames[1]), int(prev_index), int(cur_index)))
        if hit is not None:
            matrix[position] = rows[key][hit]
            has_eye[position] = 1
    return matrix, has_eye


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def compute_readouts(pairs: dict, video_ids: list[str], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    t4t5 = np.zeros((len(pairs["targets"]), len(READOUT_PAIR_FEATURES)), np.float32)
    all34 = np.zeros((len(pairs["targets"]), len(READOUT_ALL34_FEATURES)), np.float32)
    stats = {}
    for video_index, video_id in enumerate(video_ids):
        rows = np.flatnonzero(pairs["video_indices"] == video_index)
        if not rows.size:
            continue
        cache = load_activity_cache(args.activity_root / video_id / f"{args.activity_source}_activity.npz")
        assert list(cache["cell_types"]) == list(OUTPUT_CELL_TYPES)
        frames = cache["frames"]
        points = cache["points"]
        baseline = frame_activity_matrix(cache["baseline"])
        spacing = column_spacing(points)
        matrices: dict[int, np.ndarray] = {}

        def matrix(frame_index: int) -> np.ndarray:
            if frame_index not in matrices:
                matrices[frame_index] = frame_activity_matrix(frames[frame_index])
                if len(matrices) > 8:
                    matrices.pop(next(iter(matrices)))
            return matrices[frame_index]

        order = rows[np.argsort(pairs["pair_frames"][rows, 1], kind="stable")]
        for count, row in enumerate(order, 1):
            start, stop = (int(v) for v in pairs["pair_frames"][row])
            if stop >= len(frames) or start >= len(frames):
                raise RuntimeError(f"{video_id}: кадр пары {start}->{stop} вне кэша активности ({len(frames)})")
            kwargs = dict(previous_box=pairs["previous_boxes"][row], current_box=pairs["current_boxes"][row], min_margin=spacing, ring_scale=args.ring_scale)
            t4t5[row] = readout_pair_features(matrix(start), matrix(stop), baseline, points, **kwargs)
            all34[row] = readout_all34_features(matrix(start), matrix(stop), baseline, points, **kwargs)
            if count % 20000 == 0:
                print(f"[READOUT] {video_id}: {count}/{len(rows)}", flush=True)
        stats[video_id] = {"pairs": int(rows.size), "frames": int(len(frames)), "column_spacing_px": spacing, "max_pair_frame": int(pairs["pair_frames"][rows].max())}
        print(f"[READOUT] {video_id}: {rows.size} пар готово", flush=True)
    return t4t5, all34, stats

FOVEA_T4T5 = tuple(f"fovea_{n}" for n in READOUT_PAIR_FEATURES)
FOVEA_ALL34 = tuple(f"fovea_{n}" for n in READOUT_ALL34_FEATURES)
FOVEA_SHUFFLED = tuple(f"shuffled_fovea_{n}" for n in READOUT_PAIR_FEATURES)
TILED_T4T5 = tuple(f"tiled_{n}" for n in READOUT_PAIR_FEATURES)
EYE_FLAG = ("has_eye",)
FOVEA_FEATURE_NAMES = FEATURE_NAMES + EYE_FLAG + FOVEA_T4T5 + FOVEA_ALL34 + FOVEA_SHUFFLED + TILED_T4T5
FOVEA_GROUPS = {
    "base_dis_eye": BASE_FEATURES + DIS_FEATURES + EYE_FLAG,
    "base_dis_eye_decoded": BASE_FEATURES + DIS_FEATURES + EYE_FLAG + DECODED_TILED_FEATURES,
    "base_dis_eye_fovea_t4t5": BASE_FEATURES + DIS_FEATURES + EYE_FLAG + FOVEA_T4T5,
    "base_dis_eye_fovea_all34": BASE_FEATURES + DIS_FEATURES + EYE_FLAG + FOVEA_ALL34,
    "base_dis_eye_shuffled_fovea": BASE_FEATURES + DIS_FEATURES + EYE_FLAG + FOVEA_SHUFFLED,
    "base_dis_eye_tiled_t4t5": BASE_FEATURES + DIS_FEATURES + EYE_FLAG + TILED_T4T5,
    "base_dis_eye_fovea_tiled": BASE_FEATURES + DIS_FEATURES + EYE_FLAG + FOVEA_T4T5 + TILED_T4T5,
}
FOVEA_COMPARISONS = {
    "primary_fovea_vs_base_dis_eye": ("base_dis_eye", "base_dis_eye_fovea_t4t5"),
    "fovea_vs_shuffled": ("base_dis_eye_shuffled_fovea", "base_dis_eye_fovea_t4t5"),
    "fovea_vs_tiled": ("base_dis_eye_tiled_t4t5", "base_dis_eye_fovea_t4t5"),
    "fovea_vs_decoded": ("base_dis_eye_decoded", "base_dis_eye_fovea_t4t5"),
    "all34_vs_base_dis_eye": ("base_dis_eye", "base_dis_eye_fovea_all34"),
}


def run_fovea_mode(args: argparse.Namespace, pairs: dict, video_ids: list[str], identity: dict) -> int:
    n = len(pairs["targets"])
    t4t5 = np.zeros((n, len(READOUT_PAIR_FEATURES)), np.float32)
    all34 = np.zeros((n, len(READOUT_ALL34_FEATURES)), np.float32)
    has_eye = np.zeros(n, np.int8)
    coverage = {}
    for video_index, video_id in enumerate(video_ids):
        rows = np.flatnonzero(pairs["video_indices"] == video_index)
        with np.load(args.fovea_root / video_id / "fovea_readout.npz") as data:
            fovea = {key: data[key] for key in data.files}
        assert [str(x) for x in fovea["t4t5_names"]] == list(READOUT_PAIR_FEATURES)
        sub = {key: pairs[key][rows] for key in ("pair_frames", "previous_index", "current_index")}
        t4t5[rows], flag = join_fovea_rows(sub, fovea, key="t4t5", width=len(READOUT_PAIR_FEATURES))
        all34[rows], _ = join_fovea_rows(sub, fovea, key="all34", width=len(READOUT_ALL34_FEATURES))
        has_eye[rows] = flag
        targets = pairs["targets"][rows]
        coverage[video_id] = {
            "pairs": int(rows.size),
            "with_eye": float(flag.mean()),
            "positives_with_eye": float(flag[targets == 1].mean()) if (targets == 1).any() else None,
            "negatives_with_eye": float(flag[targets == 0].mean()) if (targets == 0).any() else None,
            "fovea_rows": int(len(fovea["frame"])),
        }
        print(f"[FOVEA] {video_id}: глаз у {flag.mean():.3f} пар, у положительных {coverage[video_id]['positives_with_eye']}", flush=True)
    shuffled = np.zeros_like(t4t5)
    rng = np.random.default_rng(args.seed)
    for video in np.unique(pairs["video_indices"]):
        rows = np.flatnonzero((pairs["video_indices"] == video) & (has_eye == 1))
        shuffled[rows] = t4t5[rng.permutation(rows)]
    tiled = np.zeros_like(t4t5)
    if args.tiled_readout is not None:
        with np.load(args.tiled_readout) as data:
            assert sha256_arrays(data["targets"], data["video_indices"]) == sha256_arrays(pairs["targets"], pairs["video_indices"])
            tiled = data["t4t5"].astype(np.float32)
    features = np.concatenate([pairs["features"].astype(np.float32), has_eye[:, None].astype(np.float32), t4t5, all34, shuffled, tiled], axis=1)
    assert features.shape[1] == len(FOVEA_FEATURE_NAMES)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "fovea_features.npz", t4t5=t4t5, all34=all34, has_eye=has_eye, shuffled=shuffled, video_indices=pairs["video_indices"], targets=pairs["targets"])

    groups = dict(FOVEA_GROUPS)
    if args.tiled_readout is None:
        groups = {k: v for k, v in groups.items() if "tiled" not in k}
    results = {}
    for name, feature_names in groups.items():
        print(f"[FIT] {name} ({len(feature_names)} признаков)", flush=True)
        results[name] = fit_and_evaluate(features, pairs["targets"], pairs["video_indices"], pairs["current_truth_matched"], feature_names, len(TRAIN_VIDEO_IDS), tuple(video_ids), all_feature_names=FOVEA_FEATURE_NAMES)
    comparisons = {k: v for k, v in FOVEA_COMPARISONS.items() if v[0] in results and v[1] in results}
    bootstrap = {c: {m: paired_video_bootstrap(results[a], results[b], m) for m in ("average_precision", "brier")} for c, (a, b) in comparisons.items()}

    def wins(c: str, metric: str, better: str):
        if c not in comparisons:
            return None
        a, b = comparisons[c]
        pa = {v: r[metric] for v, r in results[a]["test_by_video"].items()}
        pb = {v: r[metric] for v, r in results[b]["test_by_video"].items()}
        return all((pb[v] > pa[v]) if better == "higher" else (pb[v] <= pa[v]) for v in pa)

    def ap(name: str) -> float:
        return results[name]["test"]["average_precision"]

    acceptance = {
        "positives_with_eye_at_least_0_8_every_video": all((c["positives_with_eye"] or 0) >= 0.8 for c in coverage.values()),
        "primary_ap_up_every_video": wins("primary_fovea_vs_base_dis_eye", "average_precision", "higher"),
        "primary_brier_not_worse_every_video": wins("primary_fovea_vs_base_dis_eye", "brier", "lower"),
        "fovea_beats_shuffled_every_video": wins("fovea_vs_shuffled", "average_precision", "higher"),
        "fovea_not_worse_than_tiled_ap": (ap("base_dis_eye_fovea_t4t5") >= ap("base_dis_eye_tiled_t4t5")) if "base_dis_eye_tiled_t4t5" in results else None,
        "fovea_not_worse_than_decoded_ap": ap("base_dis_eye_fovea_t4t5") >= ap("base_dis_eye_decoded"),
        "same_pairs_and_labels": identity.get("same_pairs_and_labels"),
    }
    report = {
        "protocol": {
            "mode": "fovea", "pairs": str(args.pairs), "fovea_root": str(args.fovea_root),
            "tiled_readout": str(args.tiled_readout) if args.tiled_readout else None,
            "readout": "карточка flyvis_foveation_readout_v1: глаз на трек XYWHByteTrack, self = предсказанная рамка, candidate = детекция в вырезке; те же признаки T4T5/all34, что у tiled, в координатах глаза",
            "control": "base + DIS + has_eye (флаг — состояние трекера, доступное на инференсе)",
            "shuffled": f"перестановка строк fovea T4T5 среди пар с глазом внутри видео, сид {args.seed}",
            "train_video_ids": list(TRAIN_VIDEO_IDS), "test_video_ids": list(TEST_VIDEO_IDS),
            "model": "StandardScaler + LogisticRegression(C=1, max_iter=2000), порог по F1 на обучающих видео",
        },
        "identity": identity, "eye_coverage": coverage, "feature_counts": {k: len(v) for k, v in groups.items()},
        "feature_groups": results, "comparisons": bootstrap, "acceptance": acceptance,
    }
    save_json(args.output / "report.json", report)
    summary = {name: {"AP": round(r["test"]["average_precision"], 6), "Brier": round(r["test"]["brier"], 6), "by_video_AP": {k[:8]: round(v["average_precision"], 5) for k, v in r["test_by_video"].items()}} for name, r in results.items()}
    print(json.dumps({"summary": summary, "acceptance": acceptance}, ensure_ascii=False, indent=1))
    print(f"[OK] {args.output / 'report.json'}")
    return 0


def main() -> int:
    args = parse_args()
    with np.load(args.pairs) as data:
        pairs = {key: data[key] for key in data.files}
    video_ids = [str(v) for v in pairs["video_ids"]]
    assert tuple(video_ids) == TRAIN_VIDEO_IDS + TEST_VIDEO_IDS
    assert [str(n) for n in pairs["feature_names"]] == list(FEATURE_NAMES)
    for key in ("pair_frames", "previous_boxes", "current_boxes"):
        if key not in pairs:
            raise RuntimeError(f"{args.pairs}: нет {key}; пересоберите пары гейтом с поддержкой метаданных")

    identity = {"pairs_sha256": sha256_arrays(pairs["features"], pairs["targets"], pairs["video_indices"])}
    if args.reference_pairs is not None:
        with np.load(args.reference_pairs) as data:
            identity["reference_sha256"] = sha256_arrays(data["features"], data["targets"], data["video_indices"])
        identity["same_pairs_and_labels"] = identity["pairs_sha256"] == identity["reference_sha256"]
        print(f"[PAIRS] тождество с {args.reference_pairs}: {identity['same_pairs_and_labels']}", flush=True)
    if args.fovea_root is not None:
        return run_fovea_mode(args, pairs, video_ids, identity)

    t4t5, all34, readout_stats = compute_readouts(pairs, video_ids, args)
    shuffled = shuffle_within_video(t4t5, pairs["video_indices"], args.seed)
    features = np.concatenate([pairs["features"].astype(np.float32), t4t5, all34, shuffled], axis=1)
    assert features.shape[1] == len(ALL_FEATURE_NAMES)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "readout_features.npz", t4t5=t4t5, all34=all34, shuffled=shuffled, video_indices=pairs["video_indices"], targets=pairs["targets"], feature_names=np.asarray(READOUT_PAIR_FEATURES + READOUT_ALL34_FEATURES))

    results = {}
    for name, feature_names in FEATURE_GROUPS.items():
        print(f"[FIT] {name} ({len(feature_names)} признаков)", flush=True)
        results[name] = fit_and_evaluate(features, pairs["targets"], pairs["video_indices"], pairs["current_truth_matched"], feature_names, len(TRAIN_VIDEO_IDS), tuple(video_ids), all_feature_names=ALL_FEATURE_NAMES)
    bootstrap = {
        comparison: {metric: paired_video_bootstrap(results[control], results[variant], metric) for metric in ("average_precision", "brier")}
        for comparison, (control, variant) in COMPARISONS.items()
    }

    def per_video(name: str, metric: str) -> dict:
        return {video: float(row[metric]) for video, row in results[name]["test_by_video"].items()}

    def wins_every_video(comparison: str, metric: str, better: str) -> bool:
        control, variant = COMPARISONS[comparison]
        a, b = per_video(control, metric), per_video(variant, metric)
        return all((b[v] > a[v]) if better == "higher" else (b[v] <= a[v]) for v in a)

    acceptance = {
        "primary_ap_up_every_video": wins_every_video("primary_t4t5_vs_base_dis", "average_precision", "higher"),
        "primary_brier_not_worse_every_video": wins_every_video("primary_t4t5_vs_base_dis", "brier", "lower"),
        "t4t5_beats_shuffled_every_video": wins_every_video("t4t5_vs_shuffled", "average_precision", "higher"),
        "t4t5_not_worse_than_decoded_ap": results["base_dis_t4t5"]["test"]["average_precision"] >= results["base_dis_decoded_tiled"]["test"]["average_precision"],
        "same_pairs_and_labels": identity.get("same_pairs_and_labels"),
    }
    report = {
        "protocol": {
            "pairs": str(args.pairs), "reference_pairs": str(args.reference_pairs) if args.reference_pairs else None,
            "activity_root": str(args.activity_root), "activity_source": args.activity_source,
            "readout": "карточка flyvis_internal_readout_v1: inside/ring по колонкам, T4/T5 энергии направлений, ON/OFF, доминирование, энтропия, чистый вектор, контраст, coverage; all34 — средняя активность 34 типов; baseline после fade-in вычитается",
            "ring_scale": args.ring_scale, "min_margin": "шаг сетки колонок (медиана расстояния до соседа)",
            "shuffled": f"перестановка строк T4T5-признаков внутри видео, сид {args.seed}",
            "train_video_ids": list(TRAIN_VIDEO_IDS), "test_video_ids": list(TEST_VIDEO_IDS),
            "model": "StandardScaler + LogisticRegression(C=1, max_iter=2000), порог по F1 на обучающих видео",
        },
        "identity": identity,
        "readout_stats": readout_stats,
        "feature_counts": {name: len(group) for name, group in FEATURE_GROUPS.items()},
        "feature_groups": results,
        "comparisons": bootstrap,
        "acceptance": acceptance,
    }
    save_json(args.output / "report.json", report)
    summary = {name: {"AP": round(r["test"]["average_precision"], 6), "Brier": round(r["test"]["brier"], 6), "by_video_AP": {k[:8]: round(v["average_precision"], 5) for k, v in r["test_by_video"].items()}} for name, r in results.items()}
    print(json.dumps({"summary": summary, "acceptance": acceptance}, ensure_ascii=False, indent=1))
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
