"""Сквозная проверка рычагов надёжности предсказания по HOTA (карточка e2e_motion_source_v1).

Конфигурации `MotionSourceByteTrack` (источник движения, буфер IoU, голова надёжности)
прогоняются на dev/test обучающих видео (кэши ступени 1, разметка BDD100K train) и на
val-семёрке. HOTA/IDF1/AssA — по каждому видео отдельно (для парного бутстрепа) и по
сплиту целиком. Параметры трекера — замороженного стенда (bench_freeze_v1).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, load_motion_cache, save_json, scalabel_frame  # noqa: E402

XYWH = {"track_high_thresh": 0.25, "track_low_thresh": 0.1, "track_buffer": 30, "match_thresh": 0.8}
VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSw", "CLR_FP", "CLR_FN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True, help="JSON {имя: kwargs MotionSourceByteTrack (+ head: путь к голове)}")
    parser.add_argument("--splits", default="dev,test,val7")
    parser.add_argument("--train-cache", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--val-root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def split_videos(args: argparse.Namespace) -> dict[str, tuple[list[str], Path, str]]:
    selection = json.loads((args.train_cache / "selection.json").read_text(encoding="utf-8"))["videos"]
    out = {}
    for split in args.splits.split(","):
        if split == "val7":
            out[split] = (VAL7, args.val_root, "val")
        else:
            out[split] = (selection[split], args.train_cache, "train")
    return out


def make_tracker(config: dict, root: Path, video_id: str):
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box
    from tracking.motion_source_bytetrack import MotionSourceByteTrack
    from tracking.offline_association import make_tracker_args

    kwargs = {k: v for k, v in config.items() if k not in {"head"}}
    needs_dis = kwargs.get("motion", "kalman") != "kalman" or config.get("head")
    if needs_dis:
        points, vectors = load_motion_cache(root / video_id / "motion" / "dis_quarter.npz", 1.0)
        kwargs["dis_propagate"] = lambda box, raw: propagate_box(box, points, vectors, raw - 1, raw)[0] if raw > 0 else np.full(4, np.nan)
    if config.get("head"):
        from tracking.reliability_head import load_reliability_head
        kwargs["reliability"] = load_reliability_head(Path(config["head"]))
    return MotionSourceByteTrack(make_tracker_args(**XYWH), **kwargs)


def run_video(task: tuple) -> dict:
    name, config, split, video_id, root, label_split, out_dir = task
    from evaluation.hota_eval import evaluate_hota
    from tracking.offline_association import _DetSlice

    result_dir = Path(out_dir) / name / split
    result_dir.mkdir(parents=True, exist_ok=True)
    detections = load_detection_cache(root / video_id / "detections" / "raw_video.npz")
    alignment = json.loads((root / video_id / "alignment.json").read_text(encoding="utf-8"))
    truth = json.loads((PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / label_split / f"{video_id}.json").read_text(encoding="utf-8"))
    anchor = {int(r["video_frame_index"]): i for i, r in enumerate(alignment["matches"])}
    tracker = make_tracker(config, root, video_id)
    rows = []
    for frame_index in range(int(detections["frame_count"])):
        xyxy, conf, classes = detections_for_frame(detections, frame_index)
        output = tracker.update(_DetSlice(xyxy, conf, classes))
        if frame_index in anchor:
            rows.append(scalabel_frame(video_id, truth[anchor[frame_index]], output))
    save_json(result_dir / f"{video_id}.json", rows)
    metrics = evaluate_hota(result_dir, [video_id], split=label_split)
    return {"name": name, "split": split, "video_id": video_id, "metrics": {k: metrics.get(k) for k in METRICS}, "stats": dict(tracker.stats)}


def main() -> int:
    args = parse_args()
    configs = json.loads(args.configs.read_text(encoding="utf-8"))
    splits = split_videos(args)
    tasks = [(name, cfg, split, v, root, label_split, str(args.output)) for name, cfg in configs.items() for split, (vids, root, label_split) in splits.items() for v in vids]
    print(f"[PLAN] {len(configs)} конфигураций × {sum(len(v[0]) for v in splits.values())} видео = {len(tasks)} прогонов", flush=True)
    per_video: dict = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, res in enumerate(pool.map(run_video, tasks, chunksize=1), 1):
            per_video.setdefault(res["name"], {}).setdefault(res["split"], {})[res["video_id"]] = {**res["metrics"], "stats": res["stats"]}
            if k % 50 == 0:
                print(f"[RUN] {k}/{len(tasks)}", flush=True)
    from evaluation.hota_eval import evaluate_hota

    summary = {}
    for name in configs:
        summary[name] = {}
        for split, (vids, root, label_split) in splits.items():
            agg = evaluate_hota(args.output / name / split, vids, split=label_split)
            summary[name][split] = {k: agg.get(k) for k in METRICS}
        line = " | ".join(f"{s}: HOTA {summary[name][s]['HOTA']:.3f} IDF1 {summary[name][s]['IDF1']:.3f} AssA {summary[name][s]['AssA']:.3f} IDSw {summary[name][s]['IDSw']:.0f}" for s in splits)
        print(f"[SUM] {name:28s} {line}", flush=True)
    save_json(args.output / "report.json", {"configs": configs, "splits": {s: v[0] for s, v in splits.items()}, "summary": summary, "per_video": per_video})
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
