"""Эталонный допуск слабых детекций, реальные события и метрики трекинга.

Запуск не обучает детектор. Все вмешательства помечены oracle.
По умолчанию 6 дневных и 6 ночных видео, выбранных воспроизводимо.
Возраст 0 на первом кадре видео учитывается отдельной группой initial.
birth/mature сравниваются с одинаковым числом кандидатов в каждой ячейке
класс × диапазон score × размер × окклюзия внутри одного видео.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.bdd100k_loader import CLASS_NAMES
from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
from evaluation.hota_eval import evaluate_hota
from evaluation.tracker_lifecycle import (LifecycleAudit, ignored_unmatched,
                                         label_candidates, read_truth)
from scripts.tracker_zoo import load_args
from tracking.confidence_audit import AuditedBoTSORT, IndexedDetections
from tracking.detection_cache import iter_frames, load_video_cache

GROUPS = ("birth", "mature", "continuation", "recovery", "initial")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def seed_for(seed, video):
    return int.from_bytes(hashlib.sha256(f"{seed}:{video}".encode()).digest()[:8], "little")


def stratum(row):
    area = row["area"]
    return (row["class"], int(row["score"] >= .1),
            0 if area < 32 ** 2 else 1 if area < 96 ** 2 else 2, row["occluded"])


def select_cohorts(candidates, seed):
    """Списки (frame, index), один и тот же случайный порядок для всех долей."""
    rng = np.random.default_rng(seed)
    birth, mature = defaultdict(list), defaultdict(list)
    others = {k: [] for k in GROUPS if k not in ("birth", "mature")}
    for row in candidates:
        key = (row["frame"], row["index"])
        if row["age"] == 0 and row["frame"] > 0:
            birth[stratum(row)].append(key)
        if row["age"] >= 10:
            mature[stratum(row)].append(key)
        if row["age"] == 0 and row["frame"] == 0:
            others["initial"].append(key)
        if row["context"] in ("continuation", "recovery"):
            others[row["context"]].append(key)
    pairs = []
    for cell in sorted(birth):
        b, m = birth[cell], mature[cell]
        rng.shuffle(b)
        rng.shuffle(m)
        pairs.extend(zip(b[:min(len(b), len(m))], m[:min(len(b), len(m))]))
    rng.shuffle(pairs)
    groups = {"birth": [x[0] for x in pairs], "mature": [x[1] for x in pairs]}
    for k, v in others.items():
        rng.shuffle(v)
        groups[k] = v
    counts = {"birth_available": sum(map(len, birth.values())),
              "mature_available": sum(map(len, mature.values())),
              "birth_mature_matched": len(pairs)}
    counts.update({f"{k}_available": len(v) for k, v in others.items()})
    return groups, counts


def chosen_fraction(ordered, fraction):
    # Явное округление вниз, чтобы не выдавать единственный объект за 25%.
    return ordered[:int(len(ordered) * fraction)]


def calculate_warps(frames, video, cfg):
    import cv2
    from ultralytics.trackers.utils.gmc import GMC
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    gmc = GMC(method=cfg.gmc_method)
    warps = []
    for fi, name, boxes, scores, _ in frames:
        image = cv2.imread(str(ROOT / "data/bdd100k/images/track/val" / video / name))
        if image is None:
            raise FileNotFoundError(f"{video}/{name}: отсутствует кадр для GMC")
        warp = gmc.apply(image, boxes[scores >= cfg.track_high_thresh])
        if not np.isfinite(warp).all():
            raise ValueError(f"{video}/{fi}: нечисловая матрица GMC")
        warps.append(warp)
    return warps


def run_variant(frames, truth, first, warps, cfg, *, fixed, selected, action,
                directory, video, build_candidates=False):
    tracker = AuditedBoTSORT(load_args("botsort", vars(cfg)), second_iou_only=fixed)
    audit = LifecycleAudit(first)
    by_frame = defaultdict(list)
    for fi, idx in selected:
        by_frame[fi].append(idx)
    outputs, candidates = [], []
    stats = Counter()
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"{video}.events.jsonl.gz", "wt", encoding="utf-8") as log:
        for position, (fi, name, boxes, scores, classes) in enumerate(frames):
            gt = truth[fi]
            frame_candidates = []
            if gt.name and gt.name != name:
                raise ValueError(f"{video}/{fi}: имя кадра в GT и кэше не совпадает")
            mapping = label_candidates(boxes, classes, scores, gt, cfg.track_high_thresh)
            if build_candidates:
                stats["gt_observations"] += len(gt.ids)
                stats["candidate_high_tp"] += int(((mapping >= 0) & (scores >= cfg.track_high_thresh)).sum())
                stats["candidate_low_tp"] += int(((mapping >= 0) & (scores < cfg.track_high_thresh)).sum())
                stats["empty_detection_frames"] += int(len(boxes) == 0)
                stats["gt_birth_observations"] += int(((gt.ages == 0) & (fi > 0)).sum())
                stats["gt_initial_observations"] += int(((gt.ages == 0) & (fi == 0)).sum())
                stats["gt_mature_observations"] += int((gt.ages >= 10).sum())
                for oid in gt.ids:
                    last = audit.last_correct.get(oid)
                    if last is not None:
                        stats["gt_continuation_observations" if last == fi - 1
                              else "gt_recovery_observations"] += 1
                for idx in np.flatnonzero((mapping >= 0) & (scores < cfg.track_high_thresh)):
                    gi = mapping[idx]
                    oid = gt.ids[gi]
                    last = audit.last_correct.get(oid)
                    context = ("untracked" if last is None else
                               "continuation" if last == fi - 1 else "recovery")
                    b = gt.boxes[gi]
                    frame_candidates.append({"frame": fi, "index": int(idx), "gt_id": oid,
                                       "class": int(classes[idx]), "score": float(scores[idx]),
                                       "age": int(gt.ages[gi]), "occluded": bool(gt.occluded[gi]),
                                       "area": float((b[2]-b[0])*(b[3]-b[1])), "context": context})
            admitted = by_frame[fi]
            changed_scores = scores.copy()
            if action == "boost":
                changed_scores[admitted] = 1.0
            detections = IndexedDetections(boxes, changed_scores, classes)
            out = tracker.update(detections, allowed=admitted if action == "gate" else (),
                                 warp=warps[position])
            consumed = {e["det_index"] for e in tracker.events
                        if "det_index" in e and e["event"] != "rejected"}
            # Вмешиваемся только в реально отклонённые кандидаты контроля.
            # Уже использованный второй проход нельзя выдавать за резерв.
            rejected_candidates = [r for r in frame_candidates if r["index"] not in consumed]
            candidates.extend(rejected_candidates)
            if build_candidates:
                stats["rejected_true_low"] += len(rejected_candidates)
            # События содержат сырой балл и GT только ПОСЛЕ решения трекера.
            for event in tracker.events:
                event = dict(event, frame=fi)
                idx = event.get("det_index")
                if idx is not None:
                    gi = mapping[idx]
                    event.update(raw_score=float(scores[idx]), oracle_selected=idx in admitted,
                                 gt_id=gt.ids[gi] if gi >= 0 else None)
                log.write(json.dumps(event) + "\n")
                stats["event_" + event["event"]] += 1
                if "stage" in event:
                    stats["matches_" + event["stage"]] += 1
            audit.observe(fi, out, gt)
            for event in audit.events:
                log.write(json.dumps(dict(event, source="output_audit")) + "\n")
            audit.events.clear()
            labels = [{"id": str(int(row[4])), "category": CLASS_NAMES[int(row[6])],
                       "score": float(row[5]),
                       "box2d": dict(zip(("x1", "y1", "x2", "y2"), map(float, row[:4])))}
                      for row in out]
            outputs.append({"videoName": video, "name": name, "frameIndex": fi, "labels": labels})
    write_json(directory / f"{video}.json", outputs)
    summary = {**audit.summary(), **dict(stats), "selected": len(selected)}
    write_json(directory / f"{video}.audit.json", summary)
    return candidates, summary


def main():
    import ultralytics
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=ROOT / "runs/det_cache_ctl_960")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/tracker1_rescue_pilot")
    ap.add_argument("--per-condition", type=int, default=6)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--groups", nargs="*", choices=GROUPS,
                    default=["birth", "mature", "continuation", "recovery"])
    ap.add_argument("--fractions", nargs="+", type=float, default=[.25, .5, 1.])
    ap.add_argument("--action", choices=["gate", "boost"], default="gate")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.per_condition <= 0 or any(not 0 < f <= 1 for f in a.fractions):
        ap.error("per-condition > 0, fractions в (0,1]")
    a.out.mkdir(parents=True, exist_ok=True)
    attrs = load_video_attributes(str(ROOT / "data/bdd100k/labels/bdd100k_labels_images_val.json"))
    available = sorted(p.stem for p in (a.cache / "val").glob("*.npz"))
    conditions = {k: filter_videos_by_subset(attrs, available, k) for k in ("normal", "night")}
    rng = np.random.default_rng(a.seed)
    if a.all:
        videos = available
    else:
        videos = sorted(v for k in ("normal", "night")
                        for v in rng.choice(conditions[k], min(a.per_condition, len(conditions[k])), replace=False))
    cfg = load_args("botsort", {"track_high_thresh": .25, "new_track_thresh": .25,
                               "track_low_thresh": .1, "with_reid": False})
    modes = [("installed", False, None, 0.), ("iou_second", True, None, 0.)]
    modes += [(f"oracle_{group}_{a.action}_{fraction:g}", True, group, fraction)
              for group in a.groups for fraction in a.fractions]
    source_paths = [Path(__file__), ROOT / "tracking/confidence_audit.py",
                    ROOT / "evaluation/tracker_lifecycle.py", ROOT / "evaluation/hota_eval.py"]
    from ultralytics.trackers import byte_tracker, bot_sort
    from ultralytics.trackers.utils import matching, gmc, stracks
    source_paths += [Path(m.__file__) for m in (byte_tracker, bot_sort, matching, gmc, stracks)]
    import cv2
    import scipy
    import importlib.metadata
    manifest = {"protocol": 2, "ultralytics": ultralytics.__version__, "seed": a.seed,
                "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                "opencv": cv2.__version__, "trackeval": importlib.metadata.version("trackeval"),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "cache_meta": json.loads((a.cache / "meta.json").read_text(encoding="utf-8")),
                "cache": str(a.cache.resolve()), "videos": videos,
                "conditions": {k: sorted(set(videos) & set(v)) for k, v in conditions.items()},
                "tracker": vars(cfg), "modes": modes, "action": a.action,
                "sources": {str(p): digest(p) for p in source_paths},
                "cache_hashes": {v: digest(a.cache / "val" / f"{v}.npz") for v in videos},
                "gt_hashes": {v: digest(ROOT / "data/bdd100k/labels/box_track_20/val" / f"{v}.json")
                              for v in videos}}
    # JSON преобразует tuple в list — канонизируем до сравнения.
    manifest = json.loads(json.dumps(manifest))
    mp = a.out / "manifest.json"
    if mp.exists():
        if not a.resume or json.loads(mp.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Каталог занят другим/старым протоколом; задайте новый --out")
    else:
        write_json(mp, manifest)
    print(f"[START] videos={len(videos)} modes={len(modes)} output={a.out}", flush=True)
    for vi, video in enumerate(videos, 1):
        done = a.out / "completed" / f"{video}.json"
        if done.exists():
            print(f"[RESUME] {vi}/{len(videos)} {video}", flush=True)
            continue
        t0 = time.monotonic()
        cache = load_video_cache(a.cache, "val", video)
        frames = list(iter_frames(cache))
        truth, first, observations = read_truth(ROOT / "data/bdd100k/labels/box_track_20/val" / f"{video}.json")
        if sorted(truth) != list(range(len(frames))):
            raise ValueError(f"{video}: несовпадение кадров GT и кэша")
        warps = calculate_warps(frames, video, cfg)
        np.save(a.out / f"{video}.gmc.npy", np.asarray(warps))
        run_variant(frames, truth, first, warps, cfg, fixed=False, selected=[], action="gate",
                    directory=a.out / "installed", video=video)
        candidates, _ = run_variant(frames, truth, first, warps, cfg, fixed=True, selected=[], action="gate",
                                    directory=a.out / "iou_second", video=video, build_candidates=True)
        cohorts, counts = select_cohorts(candidates, seed_for(a.seed, video))
        write_json(a.out / "cohorts" / f"{video}.json",
                   {"counts": counts, "ordered": cohorts, "candidates": candidates})
        for name, fixed, group, fraction in modes[2:]:
            selected = chosen_fraction(cohorts[group], fraction)
            run_variant(frames, truth, first, warps, cfg, fixed=fixed, selected=selected, action=a.action,
                        directory=a.out / name, video=video)
        write_json(done, {"seconds": time.monotonic() - t0, "counts": counts})
        write_json(a.out / "progress.json", {"stage": "tracking", "completed": vi,
                                           "total": len(videos), "last_video": video})
        print(f"[VIDEO] {vi}/{len(videos)} {video} {time.monotonic()-t0:.1f}s {counts}", flush=True)
    rows = []
    subsets = {"all": videos, **manifest["conditions"]}
    for name, _, group, fraction in modes:
        for subset, ids in subsets.items():
            if not ids:
                continue
            total = Counter()
            for video in ids:
                audit = json.loads((a.out / name / f"{video}.audit.json").read_text())
                total.update({k: v for k, v in audit.items() if isinstance(v, int)})
            row = {"mode": name, "subset": subset, "videos": len(ids),
                   "group": group or "", "fraction": fraction, "action": a.action, **dict(total)}
            if group:
                denominator = sum(json.loads((a.out / "iou_second" / f"{v}.audit.json").read_text())
                                  .get(f"gt_{group}_observations", 0) for v in ids)
                row["target_gt_observations"] = denominator
                row["admission_recall_gain_pp"] = 100 * total["selected"] / denominator if denominator else None
            count = row.get("gt_tracks", 0) - row.get("never_tracked", 0)
            row["first_output_delay_mean_found_only"] = (
                row.get("first_output_delay_sum_found_only", 0) / count if count else None)
            metric_path = a.out / name / f"_metrics_{subset}.json"
            if not a.skip_metrics:
                if metric_path.exists() and a.resume:
                    metrics = json.loads(metric_path.read_text())
                else:
                    metrics = evaluate_hota(a.out / name, ids, tracker_name=name,
                                            details_path=a.out / name / f"_details_{subset}.json.gz")
                    write_json(metric_path, metrics)
                row.update(metrics)
            rows.append(row)
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with (a.out / "results.csv").open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            write_json(a.out / "progress.json", {"stage": "metrics", "mode": name, "subset": subset})
            print(f"[METRIC] {name}/{subset} HOTA={row.get('HOTA_cls_av')} "
                  f"FP={row.get('CLR_FP')} IDSw={row.get('IDSw')} "
                  f"selected={row.get('selected', 0)}", flush=True)
    write_json(a.out / "progress.json", {"stage": "complete", "rows": len(rows)})
    print("[DONE]", a.out / "results.csv", flush=True)


if __name__ == "__main__":
    main()
