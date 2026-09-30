"""Отложенное рождение по q_birth на обучающих видео и его совмещение с буфером IoU (карточка birth_deferred_train_v1).

Голова q_birth (StandardScaler + LogisticRegression, признаки base + DIS, как birth_policy_deferred_v1)
учится на q_birth train-сплита кэшей ступени 1 (130 роликов); τ — максимум F1 на train. Число
подтверждений N выбирается на dev (максимум HOTA), отчёт — test (35) и val7. Варианты:
deferred_nN и deferred_nN + буфер IoU 0,5 (tracking.motion_source_bytetrack.make_deferred_buffered).
Эталоны «стенд» и «буфер 0,5» берутся из отчётов eval_motion_source.py (тот же трекер стенда).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402

VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSw", "CLR_FP", "CLR_FN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "runs" / "event_dataset_train")
    parser.add_argument("--train-cache", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "birth_train")
    parser.add_argument("--confirm-hits", nargs="+", type=int, default=(2, 3, 5))
    parser.add_argument("--buffer", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stage", choices=("dev", "final"), default="dev")
    parser.add_argument("--selected", default="", help="final: имя варианта, выбранного на dev (например deferred_n3)")
    return parser.parse_args()


def train_head(dataset: Path, video_ids: list[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tracking.birth_policy import BirthHead

    features, labels, names = [], [], None
    for v in video_ids:
        path = dataset / v / "q_birth.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            names = [str(n) for n in data["feature_names"]]
            if len(data["features"]):
                features.append(data["features"]); labels.append(data["meta"][:, list(data["meta_names"]).index("label")])
    X, y = np.concatenate(features), np.concatenate(labels).astype(np.int8)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42)).fit(X, y)
    q = pipeline.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, q)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    tau = float(thresholds[int(np.argmax(f1))])
    head = BirthHead.from_pipeline(pipeline, names)
    return head, {"train_videos": len(features), "train_units": int(len(y)), "train_positives": int(y.sum()), "tau_f1": tau, "f1_train": float(f1.max())}


def run_one(task: tuple) -> dict:
    name, spec, video_id, root, videos, labels, label_split, head_path, out_dir = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import scripts.evaluate_birth_policy as ebp
    from evaluation.hota_eval import evaluate_hota
    from tracking.birth_policy import BirthHead
    from tracking.motion_source_bytetrack import make_deferred_buffered
    from tracking.offline_association import make_tracker_args

    base_make = ebp.make_tracker

    def make_tracker(s, h):
        if s.get("buffer"):
            return make_deferred_buffered(make_tracker_args(**ebp.XYWH_CONFIG), buffer=s["buffer"], head=h, tau=s["tau"], confirm_hits=s["confirm_hits"], class_ids=ebp.CLASS_IDS)
        return base_make(s, h)

    ebp.make_tracker = make_tracker
    head = BirthHead.from_json(Path(head_path))
    args = argparse.Namespace(root=Path(root), videos=Path(videos), labels=Path(labels))
    result_dir = Path(out_dir) / name / label_split
    run = ebp.run_variant(name, spec, head, args, [video_id], result_dir)
    metrics = evaluate_hota(result_dir, [video_id], split=label_split)
    return {"name": name, "video_id": video_id, "label_split": label_split, "metrics": {k: metrics.get(k) for k in METRICS}, "births": run["per_video"][video_id]}


def main() -> int:
    args = parse_args()
    selection = json.loads((args.train_cache / "selection.json").read_text(encoding="utf-8"))["videos"]
    args.output.mkdir(parents=True, exist_ok=True)
    head_path = args.output / "head.json"
    if not head_path.exists():
        head, info = train_head(args.dataset, selection["train"])
        save_json(head_path, head.to_json()); save_json(args.output / "head_info.json", info)
    info = json.loads((args.output / "head_info.json").read_text(encoding="utf-8"))
    print(f"[HEAD] {info}", flush=True)
    variants = {}
    for n in args.confirm_hits:
        variants[f"deferred_n{n}"] = {"kind": "deferred", "tau": info["tau_f1"], "confirm_hits": n}
        variants[f"deferred_n{n}+buf{args.buffer}"] = {"kind": "deferred", "tau": info["tau_f1"], "confirm_hits": n, "buffer": args.buffer}
    if args.stage == "final":
        keep = {args.selected, f"{args.selected}+buf{args.buffer}"}
        variants = {k: v for k, v in variants.items() if k in keep}
        splits = {"test": (selection["test"], args.train_cache, PROJECT_ROOT / "data/bdd100k/videos/train", PROJECT_ROOT / "data/bdd100k/labels/box_track_20/train", "train"),
                  "val7": (VAL7, PROJECT_ROOT / "runs/video_rate", PROJECT_ROOT / "data/bdd100k/videos/val", PROJECT_ROOT / "data/bdd100k/labels/box_track_20/val", "val")}
    else:
        splits = {"dev": (selection["dev"], args.train_cache, PROJECT_ROOT / "data/bdd100k/videos/train", PROJECT_ROOT / "data/bdd100k/labels/box_track_20/train", "train")}
    tasks = [(name, spec, v, str(root), str(videos), str(labels), lsplit, str(head_path), str(args.output)) for name, spec in variants.items() for split, (vids, root, videos, labels, lsplit) in splits.items() for v in vids]
    print(f"[PLAN] {len(variants)} вариантов × {sum(len(s[0]) for s in splits.values())} видео = {len(tasks)}", flush=True)
    per_video: dict = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, res in enumerate(pool.map(run_one, tasks, chunksize=1), 1):
            split = next(s for s, (vids, *_rest) in splits.items() if res["video_id"] in vids)
            per_video.setdefault(res["name"], {}).setdefault(split, {})[res["video_id"]] = {**res["metrics"], "births": res["births"]}
            if k % 20 == 0:
                print(f"[RUN] {k}/{len(tasks)}", flush=True)
    from evaluation.hota_eval import evaluate_hota

    summary = {}
    for name in variants:
        summary[name] = {}
        for split, (vids, root, videos, labels, lsplit) in splits.items():
            agg = evaluate_hota(args.output / name / lsplit, vids, split=lsplit)
            summary[name][split] = {k: agg.get(k) for k in METRICS}
            fb = sum(per_video[name][split][v]["births"]["false_births"] for v in vids); tb = sum(per_video[name][split][v]["births"]["true_births"] for v in vids)
            summary[name][split].update({"false_births": fb, "true_births": tb})
            m = summary[name][split]
            print(f"[SUM] {name:24s} {split}: HOTA {m['HOTA']:.3f} IDF1 {m['IDF1']:.3f} AssA {m['AssA']:.3f} IDSw {m['IDSw']:.0f} FP {m['CLR_FP']} FN {m['CLR_FN']} | births true {tb} false {fb}", flush=True)
    save_json(args.output / f"report_{args.stage}.json", {"head": info, "variants": variants, "splits": {s: v[0] for s, v in splits.items()}, "summary": summary, "per_video": per_video})
    print(f"[OK] {args.output / f'report_{args.stage}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
