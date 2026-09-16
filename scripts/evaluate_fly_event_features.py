"""Н3: считывание FlyVis как признак событий без пары — q_alive и q_birth (карточка flyvis_event_features_v1).

q_alive: self-строка фовеального глаза трека на кадре перехода в lost (runs/flyvis_fovea).
q_birth: тайловое считывание рамки рождающей детекции на кадре рождения (runs/flyvis_activity).
Контроль — base + DIS (+ has_eye), как в event_heads_v1; плюс shuffled. Разметка не читается.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_event_heads import fit_group, load_event  # noqa: E402
from scripts.evaluate_flyvis_qmatch_gate import TEST_VIDEO_IDS, TRAIN_VIDEO_IDS, paired_video_bootstrap  # noqa: E402
from scripts.evaluate_flyvis_readout_gate import _BOX_ALL34, _BOX_T4T5, frame_activity_matrix  # noqa: E402
from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, save_json  # noqa: E402
from tracking.flyvis_activity import box_readout_all_types, box_readout_t4t5, column_spacing, load_activity_cache  # noqa: E402

FLY_T4T5 = tuple(f"fly_{n}" for n in _BOX_T4T5)          # 24 признака self/birth-рамки
FLY_ALL34 = tuple(f"fly_{n}" for n in _BOX_ALL34)        # 69
FLY_SHUFFLED = tuple(f"shuffled_{n}" for n in FLY_T4T5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "runs" / "event_dataset")
    parser.add_argument("--fovea-root", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_fovea")
    parser.add_argument("--activity-root", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_activity")
    parser.add_argument("--detections-root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_event_features")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def alive_fly_features(data: dict, video_ids: list[str], fovea_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Для каждой единицы q_alive — self-строка глаза трека на кадре потери."""
    n = len(data["meta"]); mn = data["meta_names"]
    t4 = np.zeros((n, len(FLY_T4T5)), np.float32); a34 = np.zeros((n, len(FLY_ALL34)), np.float32); has = np.zeros(n, np.int8)
    coverage = {}
    for vi, video_id in enumerate(video_ids):
        rows = np.flatnonzero(data["video"] == vi)
        with np.load(fovea_root / video_id / "fovea_readout.npz") as f:
            frames, tracks, dets = f["frame"], f["track_id"], f["detection_index"]
            self_rows = np.flatnonzero(dets == -1)
            fov_t4, fov_34 = f["t4t5"][self_rows], f["all34"][self_rows]  # читаем массивы один раз
        lookup = {(int(frames[i]), int(tracks[i])): k for k, i in enumerate(self_rows)}
        for r in rows:
            key = (int(data["meta"][r, mn.index("frame")]), int(data["meta"][r, mn.index("track_id")]))
            hit = lookup.get(key)
            if hit is not None:
                t4[r] = fov_t4[hit][: len(FLY_T4T5)]      # первые 24 — self-рамка
                a34[r] = fov_34[hit][: len(FLY_ALL34)]
                has[r] = 1
        coverage[video_id] = float(has[rows].mean()) if rows.size else None
    return t4, a34, has, coverage


def birth_fly_features(data: dict, video_ids: list[str], activity_root: Path, detections_root: Path) -> tuple[np.ndarray, np.ndarray]:
    n = len(data["meta"]); mn = data["meta_names"]
    t4 = np.zeros((n, len(FLY_T4T5)), np.float32); a34 = np.zeros((n, len(FLY_ALL34)), np.float32)
    for vi, video_id in enumerate(video_ids):
        rows = np.flatnonzero(data["video"] == vi)
        if not rows.size:
            continue
        cache = load_activity_cache(activity_root / video_id / "flyvis_tiled_activity.npz")
        points = cache["points"]; baseline = frame_activity_matrix(cache["baseline"]); spacing = column_spacing(points)
        frames_all = cache["frames"]
        detections = load_detection_cache(detections_root / video_id / "detections" / "raw_video.npz")
        cached: dict[int, np.ndarray] = {}
        for r in rows[np.argsort(data["meta"][rows, mn.index("frame")], kind="stable")]:
            frame = int(data["meta"][r, mn.index("frame")]); det = int(data["meta"][r, mn.index("det_index")])
            if frame not in cached:
                cached = {frame: frame_activity_matrix(frames_all[frame])}
            box = detections_for_frame(detections, frame)[0][det]
            f1 = box_readout_t4t5(cached[frame], baseline, points, box, min_margin=spacing)
            f2 = box_readout_all_types(cached[frame], baseline, points, box, min_margin=spacing)
            t4[r] = [f1[k] for k in _BOX_T4T5]; a34[r] = [f2[k] for k in _BOX_ALL34]
        print(f"[BIRTH] {video_id}: {rows.size} рождений считано", flush=True)
    return t4, a34


def shuffle_rows(matrix: np.ndarray, video: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); out = np.zeros_like(matrix)
    for v in np.unique(video):
        rows = np.flatnonzero((video == v) & (mask == 1))
        out[rows] = matrix[rng.permutation(rows)]
    return out


def evaluate(event: str, data: dict, extra: dict[str, np.ndarray], base_names: tuple[str, ...], video_ids: list[str], seed: int) -> dict:
    names = list(data["names"]) + [n for block in extra.values() for n in block[0]]
    matrix = np.concatenate([data["features"]] + [block[1] for block in extra.values()], axis=1)
    data = dict(data); data["features"] = matrix; data["names"] = names; data["event"] = event
    groups = {"control": base_names, "fly_t4t5": base_names + FLY_T4T5, "fly_all34": base_names + FLY_ALL34, "shuffled_t4t5": base_names + FLY_SHUFFLED}
    results = {}
    for g, feats in groups.items():
        print(f"[FIT] {event}/{g} ({len(feats)})", flush=True)
        results[g] = fit_group(data, feats, len(TRAIN_VIDEO_IDS), video_ids, min_class_support=20)
    comps = {"fly_t4t5_vs_control": ("control", "fly_t4t5"), "fly_all34_vs_control": ("control", "fly_all34"), "fly_t4t5_vs_shuffled": ("shuffled_t4t5", "fly_t4t5")}
    boot = {c: {m: paired_video_bootstrap(results[a], results[b], m) for m in ("average_precision", "brier")} for c, (a, b) in comps.items()}

    def wins(c, metric, better):
        a, b = comps[c]
        pa = {v: r[metric] for v, r in results[a]["test_by_video"].items()}; pb = {v: r[metric] for v, r in results[b]["test_by_video"].items()}
        return all((pb[v] > pa[v]) if better == "higher" else (pb[v] <= pa[v]) for v in pa)

    acceptance = {"ap_up_every_video": wins("fly_t4t5_vs_control", "average_precision", "higher"), "brier_not_worse_every_video": wins("fly_t4t5_vs_control", "brier", "lower"), "beats_shuffled_every_video": wins("fly_t4t5_vs_shuffled", "average_precision", "higher"),
                  "all34_ap_up_every_video": wins("fly_all34_vs_control", "average_precision", "higher")}
    line = " | ".join(f"{g}: AP={r['test']['average_precision']:.4f} Brier={r['test']['brier']:.4f} by_video={[round(v['average_precision'], 4) for v in r['test_by_video'].values()]}" for g, r in results.items())
    print(f"[{event}] {line}", flush=True)
    return {"groups": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}, "comparisons": boot, "acceptance": acceptance}


def main() -> int:
    args = parse_args()
    video_ids = list(TRAIN_VIDEO_IDS + TEST_VIDEO_IDS)
    summary = json.loads((args.dataset / "summary.json").read_text(encoding="utf-8"))
    report = {"protocol": {"control": "base + DIS (+ has_eye для q_alive) — спецификация event_heads_v1", "readout": "tracking/flyvis_activity.py без изменений; q_alive — self-строка фовеального глаза на кадре потери; q_birth — тайловое считывание рамки рождения", "shuffled": f"перестановка внутри видео среди единиц с признаком, сид {args.seed}"}, "events": {}}
    # q_alive
    data = load_event(args.dataset, "q_alive", video_ids)
    t4, a34, has, coverage = alive_fly_features(data, video_ids, args.fovea_root)
    print(f"[ALIVE] покрытие глазом: { {k[:8]: round(v, 3) for k, v in coverage.items()} }", flush=True)
    sh = shuffle_rows(t4, data["video"], has, args.seed)
    base = tuple(data["names"]) + ("has_eye",)
    report["events"]["q_alive"] = {"eye_coverage": coverage, **evaluate("q_alive", data, {"eye": (("has_eye",), has[:, None].astype(np.float32)), "t4": (FLY_T4T5, t4), "a34": (FLY_ALL34, a34), "sh": (FLY_SHUFFLED, sh)}, base, video_ids, args.seed)}
    # q_birth
    data = load_event(args.dataset, "q_birth", video_ids)
    t4, a34 = birth_fly_features(data, video_ids, args.activity_root, args.detections_root)
    sh = shuffle_rows(t4, data["video"], np.ones(len(t4), np.int8), args.seed)
    report["events"]["q_birth"] = evaluate("q_birth", data, {"t4": (FLY_T4T5, t4), "a34": (FLY_ALL34, a34), "sh": (FLY_SHUFFLED, sh)}, tuple(data["names"]), video_ids, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "report.json", report)
    print(json.dumps({e: v["acceptance"] for e, v in report["events"].items()}, ensure_ascii=False))
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
