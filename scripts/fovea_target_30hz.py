"""Мишень фовеального входа мухи на 30 Гц (карточка fovea_target_30hz_v1): оракул поправки положения в области глаза.

Трекер стенда (XYWHByteTrack, h0,25 / l0,1 / буфер 30 / m0,8) на сохранённых детекциях dev 35, процессор. Перед первым
сопоставлением (хук _pre_first_associate — после прогноза Калмана) центр прогноза трека заменяется центром уверенной
детекции (≥ 0,25) его собственного объекта разметки, если центр детекции внутри области глаза — квадрат
tracking/foveated_eyes.crop_box (сторона max(2 · max(w, h), 64 px), кадр 1280 × 720). Объект трека — объект разметки его
последнего наблюдения (детекция последнего обновления → объект: венгерское внутри класса, IoU ≥ 0,5, разметка линейно
интерполирована между размеченными кадрами — association_log.gt_index / label). Разметка — только диагностика.
  stand — без оракула (сверка: HOTA без сшивки 61,336);  O1 — поправка только ведомых треков (нынешний глаз);
  O2 — также потерянных (глаз сохраняется при потере).
Счётчики: применённых поправок; поправок, менявших исход (стоимость пары «прогноз — своя детекция» до поправки ≥ 0,8 —
без поправки пара не прошла бы порог); сдвиг в долях стороны области глаза.
Выход — runs/fovea_target_30hz/report.json.

Запуск: python scripts/fovea_target_30hz.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/fovea_target_30hz"
VARIANTS = {"stand": None, "O1": ("tracked",), "O2": ("tracked", "lost")}
FRAME = (1280, 720)
KEYS = ("HOTA", "DetA", "AssA", "IDF1", "IDSw")


def new_stats() -> dict:
    return {"applied": 0, "changed": 0, "applied_tracked": 0, "applied_lost": 0, "changed_tracked": 0, "changed_lost": 0, "shift_frac": []}


def oracle_tracker(scope, frame_ctx: dict, stats: dict):
    """Трекер стенда с оракулом области глаза (scope None — без оракула); frame_ctx заполняет oracle_pre перед update.
    Используется также в scripts/build_fly_signatures.py --oracle (o2_masa_residual_v1)."""
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.bot_sort import BOTrack, parse_bboxes
    from scripts.output_filter_oracle import _iou
    from tracking.foveated_eyes import crop_box
    from tracking.xywh_bytetrack import XYWHByteTrack

    class T(BOTrack):
        own = None

    class BT(XYWHByteTrack):
        def init_track(self, results, img=None):
            if len(results) == 0:
                return []
            return [T(x, s, c) for x, s, c in zip(parse_bboxes(results), results.conf, results.cls)]

        def _pre_first_associate(self, pool, unconfirmed, img, results_high):
            super()._pre_first_associate(pool, unconfirmed, img, results_high)
            if scope is None:
                return
            hb, hl, hs = frame_ctx["high_boxes"], frame_ctx["high_labels"], frame_ctx["high_scores"]
            for t in pool:
                st = "tracked" if t.state == TrackState.Tracked else "lost"
                if st not in scope or t.own is None or not len(hb):
                    continue
                k = [i for i, g in enumerate(hl) if g == t.own]
                if not k:
                    continue
                i = k[0]; b = hb[i]; c = (b[:2] + b[2:]) / 2
                cx, cy, w, h = (float(v) for v in t.mean[:4])
                x1, y1, x2, y2 = crop_box((cx, cy, w, h), FRAME)
                if not (x1 <= c[0] <= x2 and y1 <= c[1] <= y2):
                    continue
                pb = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                cost = 1.0 - float(_iou([pb], [b])[0, 0]) * float(hs[i])
                stats["applied"] += 1; stats[f"applied_{st}"] += 1
                if cost >= 0.8:
                    stats["changed"] += 1; stats[f"changed_{st}"] += 1
                    stats["shift_frac"].append(float(np.linalg.norm(c - np.array([cx, cy])) / max(x2 - x1, 1)))
                t.mean[0], t.mean[1] = float(c[0]), float(c[1])

    return BT


def oracle_pre(tr, frame_ctx: dict, G: dict, f: int, xyxy, conf, cls) -> list:
    """Метки разметки детекций кадра f и уверенные детекции для оракула; вызывать до tr.update."""
    from datasets.bdd100k_loader import CLASS_NAMES
    from scripts.association_log import label
    names = [CLASS_NAMES[int(c)] for c in cls]
    objs = G.get(f, [])
    lab = label(list(xyxy), names, objs) if (objs and len(xyxy)) else [None] * len(xyxy)
    hi = np.asarray(conf) >= tr.args.track_high_thresh
    frame_ctx.update(high_boxes=np.asarray(xyxy)[hi] if len(xyxy) else np.zeros((0, 4)), high_labels=[l for l, m in zip(lab, hi) if m], high_scores=np.asarray(conf)[hi] if len(xyxy) else np.zeros(0))
    return lab


def oracle_post(tr, xyxy, lab: list) -> None:
    """Объект трека — объект детекции его последнего обновления; вызывать после tr.update."""
    from scripts.output_filter_oracle import _iou
    boxes = np.asarray(xyxy)
    for t in tr.tracked_stracks:
        if t.frame_id == tr.frame_id and len(boxes):
            j = int(np.argmax(_iou([t.xyxy], boxes)[0]))
            t.own = lab[j]


def run(task):
    from ultralytics.trackers.basetrack import BaseTrack
    from evaluation.hota_eval import evaluate_hota
    from scripts.association_log import gt_index
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.output_filter_oracle import VALID
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, save_json, scalabel_frame
    from tracking.offline_association import _DetSlice, make_tracker_args
    vid, root, split, name = task
    det = load_detection_cache(PROJECT_ROOT / "runs/train_cache" / vid / "detections" / "raw_video.npz")
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    vf = [int(m["video_frame_index"]) for m in matches]
    G = gt_index(truth, vf, VALID)
    n = int(np.load(PROJECT_ROOT / "runs/fly_signatures" / vid / "signatures.npz")["frame_count"])
    stats = new_stats(); frame_ctx = {}
    BaseTrack.reset_id()
    tr = oracle_tracker(VARIANTS[name], frame_ctx, stats)(make_tracker_args(**XYWH_CONFIG))
    by_frame = {}
    for f in range(n):
        xyxy, conf, cls = detections_for_frame(det, f)
        lab = oracle_pre(tr, frame_ctx, G, f, xyxy, conf, cls)
        o = tr.update(_DetSlice(xyxy, conf, cls))
        if len(o):
            by_frame[f] = np.asarray(o, np.float64)[:, :7]
        oracle_post(tr, xyxy, lab)
    rows = [scalabel_frame(vid, truth[i], by_frame.get(int(m["video_frame_index"]), np.zeros((0, 7)))) for i, m in enumerate(matches)]
    d = OUT / name; d.mkdir(parents=True, exist_ok=True); save_json(d / f"{vid}.json", rows)
    m = evaluate_hota(d, [vid], split=split)
    sf = stats.pop("shift_frac")
    return vid, name, {k: m.get(k) for k in KEYS}, stats, sf


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]]
    vids = [v for v, _, _ in dev]; split = dev[0][2]
    with ProcessPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(run, [(v, rt, sp, n) for v, rt, sp in dev for n in VARIANTS]))
    per = {n: {v: m for v, nn, m, _, _ in got if nn == n} for n in VARIANTS}
    st = {n: {k: int(sum(s[k] for v, nn, _, s, _ in got if nn == n)) for k in ("applied", "changed", "applied_tracked", "applied_lost", "changed_tracked", "changed_lost")} for n in ("O1", "O2")}
    sf = {n: [x for v, nn, _, _, f in got if nn == n for x in f] for n in ("O1", "O2")}
    night = set(filter_videos_by_subset(load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json")), vids, "night"))
    rep = {"videos": len(vids), "pooled": {}, "diff": {}, "events": st,
           "shift_frac_changed": {n: ({"median": float(np.median(sf[n])), "p90": float(np.quantile(sf[n], .9)), "n": len(sf[n])} if sf[n] else None) for n in sf}}
    rng = np.random.default_rng(1)
    for g, gv in (("all", vids), ("night", [v for v in vids if v in night])):
        for nme in VARIANTS:
            m = evaluate_hota(OUT / nme, gv, split=split); rep["pooled"][f"{nme}|{g}"] = {k: m.get(k) for k in KEYS}
        for a in ("O1", "O2"):
            d = np.array([per[a][v]["HOTA"] - per["stand"][v]["HOTA"] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rep["diff"][f"{a}-stand|{g}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d),
                                             "pooled_gain": rep["pooled"][f"{a}|{g}"]["HOTA"] - rep["pooled"][f"stand|{g}"]["HOTA"]}
    rep["criteria"] = {a: bool(rep["diff"][f"{a}-stand|all"]["pooled_gain"] >= 0.3 and rep["diff"][f"{a}-stand|all"]["ci95"][0] > 0) for a in ("O1", "O2")}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for g in ("all", "night"):
        print(f"[HOTA] {g}: " + "; ".join(f"{n} {rep['pooled'][f'{n}|{g}']['HOTA']:.3f} (AssA {rep['pooled'][f'{n}|{g}']['AssA']:.2f}, IDSw {rep['pooled'][f'{n}|{g}']['IDSw']:.0f})" for n in VARIANTS), flush=True)
        for a in ("O1", "O2"):
            x = rep["diff"][f"{a}-stand|{g}"]; print(f"   {a} − stand: общий {x['pooled_gain']:+.3f}; по видео {x['mean_per_video']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}] лучше {x['better']}/{x['n']}", flush=True)
    print(f"[EVENTS] {st}\n[SHIFT] {rep['shift_frac_changed']}\n[CRITERIA] {rep['criteria']}\n[OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
