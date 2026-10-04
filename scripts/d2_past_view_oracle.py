"""Д2: оракул «в прошлом был лучший вид» — потолок любого временного модуля (карточка past_view_oracle_v1).

Для эталонного объекта на размеченном кадре r (5 Гц), потерянного на пороге (есть кандидат того же класса
IoU ≥ 0,5, но лучшая уверенность < 0,25) и НЕ удержанного трекером-стендом (нет выхода трекера с IoU ≥ 0,5 на r):
есть ли в сырых кадрах r−1 … r−5 (30 Гц) уверенная детекция (≥ 0,25) того же класса с IoU ≥ 0,5 к положению
объекта, линейно интерполированному между его рамками на предыдущем размеченном кадре и на r?
Доля таких объектов среди реальных потерь на пороге — потолок того, что может вернуть временной модуль
по прошлому (без заглядывания вперёд). Рождение (у объекта нет предыдущего размеченного кадра) считается отдельно.

Правило (записано до прогона): доля ≥ 15 % ночью — временной модуль имеет мишень; < 15 % — закрыть.

Наборы: dev+test 70 (train_cache) и val7; выходы трекера — runs/line_b_combined/outputs/stand.
Запуск: python scripts/d2_past_view_oracle.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.oracle_detection_ladder import CATS, VAL7, iou_mat  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "d2_past_view"
LOOKBACK = 5


def video_counts(vid, root, split):
    from scripts.run_bdd_video_rate_experiment import load_detection_cache
    det = load_detection_cache(Path(root) / vid / "detections" / "raw_video.npz")
    outs = np.load(PROJECT_ROOT / "runs/line_b_combined/outputs/stand" / f"{vid}.npz")["outputs"]
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    fidx, xyxy, conf, cls = det["frame_idx"], det["xyxy"], det["conf"], det["cls"].astype(int)
    last = {}  # id -> (raw frame, box)
    c = {"loss": 0, "loss_tracked": 0, "loss_real": 0, "loss_real_birth": 0, "loss_real_past_conf": 0, "loss_real_with_past": 0}
    for m, fr in zip(matches, truth):
        r = int(m["video_frame_index"])
        labs = [(l["id"], CATS[l["category"]], np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32))
                for l in fr.get("labels") or [] if l.get("box2d") and l.get("category") in CATS]
        sel = np.flatnonzero(fidx == r)
        tr = outs[outs[:, 0] == r]
        for tid, k, box in labs:
            if len(sel):
                ious = np.where(cls[sel] == k, iou_mat(box[None], xyxy[sel])[0], 0.0)
                hit = ious >= 0.5
            else:
                hit = np.zeros(0, bool)
            is_loss = hit.any() and conf[sel][hit].max() < 0.25
            if is_loss:
                c["loss"] += 1
                held = len(tr) and (iou_mat(box[None], tr[:, 1:5])[0] >= 0.5).any()
                if held:
                    c["loss_tracked"] += 1
                else:
                    c["loss_real"] += 1
                    if tid not in last:
                        c["loss_real_birth"] += 1
                    else:
                        c["loss_real_with_past"] += 1
                        r0, b0 = last[tid]
                        found = False
                        for q in range(max(r - LOOKBACK, r0), r):
                            w = (q - r0) / max(r - r0, 1); bq = (1 - w) * b0 + w * box
                            s = np.flatnonzero((fidx == q) & (conf >= 0.25) & (cls == k))
                            if len(s) and (iou_mat(bq[None], xyxy[s])[0] >= 0.5).any():
                                found = True; break
                        c["loss_real_past_conf"] += int(found)
            last[tid] = (r, box)
    return c


def main() -> int:
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    sets = {"devtest": (sel["dev"] + sel["test"], PROJECT_ROOT / "runs/train_cache", "train"), "val7": (VAL7, PROJECT_ROOT / "runs/video_rate", "val")}
    res = {}
    for s, (vids, root, split) in sets.items():
        attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
        cond = {}
        for cnd in ("normal", "night"):
            for v in filter_videos_by_subset(attrs, vids, cnd):
                cond.setdefault(v, cnd)
        per = {v: video_counts(v, root, split) for v in vids}
        groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
        res[s] = {}
        rng = np.random.default_rng(0)
        for g, vs in groups.items():
            if not vs:
                continue
            tot = {k: sum(per[v][k] for v in vs) for k in per[vs[0]]}
            share = tot["loss_real_past_conf"] / max(tot["loss_real"], 1)
            boots = []
            for _ in range(2000):
                pick = [vs[i] for i in rng.integers(0, len(vs), len(vs))]
                a = sum(per[v]["loss_real_past_conf"] for v in pick); b = sum(per[v]["loss_real"] for v in pick)
                boots.append(a / max(b, 1))
            res[s][g] = {**tot, "videos": len(vs), "share_past_confident": share, "share_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "share_tracked": tot["loss_tracked"] / max(tot["loss"], 1), "share_birth": tot["loss_real_birth"] / max(tot["loss_real"], 1)}
            print(f"{s} [{g}] потерь на пороге {tot['loss']}, удержано трекером {res[s][g]['share_tracked']:.1%}, реальных {tot['loss_real']} "
                  f"(рождение {res[s][g]['share_birth']:.1%}); с уверенной детекцией в прошлых {LOOKBACK} кадрах: {share:.1%} "
                  f"[{res[s][g]['share_ci95'][0]:.1%}; {res[s][g]['share_ci95'][1]:.1%}]", flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
