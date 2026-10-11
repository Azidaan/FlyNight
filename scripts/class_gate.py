"""Запрет возврата потерянного трека на детекцию другого класса после потери > 30 кадров (карточка class_gate_v1).

ByteTrack стенда (XYWHByteTrack, h0,25 / l0,1 / m0,8) на сохранённых детекциях dev 35, на процессоре, без мухи и без
сшивки. Единственное отличие варианта: при первом сопоставлении стоимость пары «потерянный трек, потеря > 30 кадров —
детекция другого класса» = 1 (выше порога 0,8 — возврат запрещён). Варианты: b30, b150 (сверка с runs/buffer_extend/report.json:
61,336 и 61,114 без сшивки), b30_gate, b150_gate. Выход — scalabel на опорных кадрах (как eval_fly_reid.stitch_video без
сшивки), HOTA по TrackEval, разности по видео с бутстрэпом 5000.

Запуск: python scripts/class_gate.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/class_gate"
VARIANTS = {"b30": (30, False), "b150": (150, False), "b30_gate": (30, True), "b150_gate": (150, True)}
LOST_GATE = 30
KEYS = ("HOTA", "DetA", "AssA", "IDF1", "IDSw")


def make_tracker(buf, gate):
    from ultralytics.trackers.basetrack import TrackState
    from scripts.freeze_bench import XYWH_CONFIG
    from tracking.offline_association import make_tracker_args
    from tracking.xywh_bytetrack import XYWHByteTrack

    class GateBT(XYWHByteTrack):
        def get_dists(self, tracks, detections):
            d = super().get_dists(tracks, detections)
            if gate and len(tracks) and len(detections):
                dc = np.array([int(x.cls) for x in detections])
                for i, t in enumerate(tracks):
                    if t.state == TrackState.Lost and self.frame_id - t.end_frame > LOST_GATE:
                        d[i, dc != int(t.cls)] = 1.0
            return d
    return GateBT(make_tracker_args(**dict(XYWH_CONFIG, track_buffer=buf)))


def video(task):
    from ultralytics.trackers.basetrack import BaseTrack
    from evaluation.hota_eval import evaluate_hota
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, save_json, scalabel_frame
    from tracking.offline_association import _DetSlice
    vid, root, split, name = task
    buf, gate = VARIANTS[name]
    det = load_detection_cache(PROJECT_ROOT / "runs/train_cache" / vid / "detections" / "raw_video.npz")
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    n = int(np.load(PROJECT_ROOT / "runs/fly_signatures" / vid / "signatures.npz")["frame_count"])
    BaseTrack.reset_id()
    tr = make_tracker(buf, gate)
    by_frame = {}
    for f in range(n):
        xyxy, conf, cls = detections_for_frame(det, f)
        o = tr.update(_DetSlice(xyxy, conf, cls))
        if len(o):
            by_frame[f] = np.asarray(o, np.float64)[:, :7]
    rows = [scalabel_frame(vid, truth[i], by_frame.get(int(m["video_frame_index"]), np.zeros((0, 7)))) for i, m in enumerate(matches)]
    d = OUT / name; d.mkdir(parents=True, exist_ok=True); save_json(d / f"{vid}.json", rows)
    m = evaluate_hota(d, [vid], split=split)
    return vid, name, {k: m.get(k) for k in KEYS}


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]]
    vids = [v for v, _, _ in dev]; split = dev[0][2]
    with ProcessPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(video, [(v, rt, sp, n) for v, rt, sp in dev for n in VARIANTS]))
    per = {n: {v: m for v, nn, m in got if nn == n} for n in VARIANTS}
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    night = list(filter_videos_by_subset(attrs, vids, "night"))
    rep = {"videos": len(vids), "pooled": {}, "diff": {}}
    rng = np.random.default_rng(1)
    for g, gv in (("all", vids), ("night", night)):
        for n in VARIANTS:
            m = evaluate_hota(OUT / n, gv, split=split); rep["pooled"][f"{n}|{g}"] = {k: m.get(k) for k in KEYS}
        for a, b in (("b150_gate", "b150"), ("b30_gate", "b30"), ("b150_gate", "b30"), ("b150", "b30")):
            for met in ("HOTA", "AssA", "DetA"):
                dd = np.array([per[a][v][met] - per[b][v][met] for v in gv]); bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                rep["diff"][f"{a}-{b}|{g}|{met}"] = {"mean_per_video": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((dd > 0).sum()), "worse": int((dd < 0).sum()), "n": len(dd)}
    D = rep["diff"]; ok = lambda k: D[k]["mean_per_video"] > 0 and D[k]["ci95"][0] > 0
    rep["repro"] = {"b30": rep["pooled"]["b30|all"]["HOTA"], "b150": rep["pooled"]["b150|all"]["HOTA"], "expected": [61.336, 61.114]}
    rep["mechanism_met"] = bool(ok("b150_gate-b150|all|HOTA"))
    rep["useful_met"] = bool(ok("b150_gate-b30|all|HOTA") or ok("b30_gate-b30|all|HOTA"))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for g in ("all", "night"):
        print(f"[HOTA] {g}: " + "; ".join(f"{n} {rep['pooled'][f'{n}|{g}']['HOTA']:.3f} (AssA {rep['pooled'][f'{n}|{g}']['AssA']:.2f}, IDSw {rep['pooled'][f'{n}|{g}']['IDSw']:.0f})" for n in VARIANTS))
        for k, x in D.items():
            if f"|{g}|HOTA" in k:
                print(f"[DIFF] {k}: {x['mean_per_video']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}] лучше {x['better']}, хуже {x['worse']} из {x['n']}")
    print(f"[REPRO] {rep['repro']}\n[MECHANISM] {rep['mechanism_met']}\n[USEFUL] {rep['useful_met']}\n[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
