"""Подготовка З13 (карточка fly_object_readout_z13_v1): ориентир выбранного действия и объём обучающих примеров.

Действие З13 — поправка центра прогноза только у потерянного трека в пределах буфера (30 кадров); при отказе — штатный
прогноз. Ориентир для этого действия — оракул fovea_target_30hz только для потерянных треков (Olost; O2 трогал и ведомые):
dev 35, HOTA без сшивки против стенда (сшивку MASA для Olost не считаем — ≈ 2 ч GPU; справка, не критерий).
Объём примеров: на размеченных кадрах (не интерполированных) — сколько пар «потерянный трек в буфере × кадр»; у скольких
известен объект трека (own); у скольких центр своего объекта разметки внутри области глаза вокруг прогноза (положительный
пример: «найти»), у скольких — вне области или объекта нет в разметке кадра (отрицательный: «отказаться»); сколько из
положительных имеют уверенную свою детекцию (там действовал бы оракул). По train 130, dev 35. Разметка — только для
подготовки обучающих меток и справки; трекер её не видит.
Выход — runs/z13_prep/stats.json.

Запуск: python scripts/z13_prep_stats.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/z13_prep"
FRAME = (1280, 720)


def _init():
    import scripts.fovea_target_30hz as FT
    FT.VARIANTS["Olost"] = ("lost",)
    FT.OUT = OUT


def hota_task(task):
    import scripts.fovea_target_30hz as FT
    v, n, m, st, _ = FT.run(task)
    return v, n, m, st


def count_task(task):
    """Счёт пар «потерянный трек в буфере × размеченный кадр» на трекере стенда (без вмешательства)."""
    from ultralytics.trackers.basetrack import BaseTrack, TrackState
    import scripts.fovea_target_30hz as FT
    from scripts.association_log import gt_index
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.output_filter_oracle import VALID
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache
    from tracking.foveated_eyes import crop_box
    from tracking.offline_association import _DetSlice, make_tracker_args
    vid, root, split = task
    det = load_detection_cache(PROJECT_ROOT / "runs/train_cache" / vid / "detections" / "raw_video.npz")
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    vf = [int(m["video_frame_index"]) for m in matches]; annotated = set(vf)
    G = gt_index(truth, vf, VALID)
    n = int(np.load(PROJECT_ROOT / "runs/fly_signatures" / vid / "signatures.npz")["frame_count"])
    c = {"pairs": 0, "own_known": 0, "positive": 0, "negative_out_of_crop": 0, "negative_absent": 0, "positive_with_own_det": 0, "gap_hist": [0] * 31}
    ctx = {}; cur = {"f": -1}
    Base = FT.oracle_tracker(None, ctx, FT.new_stats())

    class Count(Base):
        def _pre_first_associate(self, pool, unconfirmed, img, results_high):
            super()._pre_first_associate(pool, unconfirmed, img, results_high)
            f = cur["f"]
            if f not in annotated:
                return
            objs = {g: b for g, _, b in G.get(f, [])}
            hb, hl = ctx["high_boxes"], ctx["high_labels"]
            for t in pool:
                if t.state == TrackState.Tracked:
                    continue
                c["pairs"] += 1; c["gap_hist"][min(max(self.frame_id - t.frame_id, 0), 30)] += 1
                if t.own is None:
                    continue
                c["own_known"] += 1
                if t.own not in objs:
                    c["negative_absent"] += 1; continue
                b = objs[t.own]; ctr = (b[:2] + b[2:]) / 2
                x1, y1, x2, y2 = crop_box(tuple(float(v) for v in t.mean[:4]), FRAME)
                if not (x1 <= ctr[0] <= x2 and y1 <= ctr[1] <= y2):
                    c["negative_out_of_crop"] += 1; continue
                c["positive"] += 1
                if any(g == t.own for g in hl):
                    c["positive_with_own_det"] += 1

    BaseTrack.reset_id()
    tr = Count(make_tracker_args(**XYWH_CONFIG))
    for f in range(n):
        xyxy, conf, cls = detections_for_frame(det, f)
        lab = FT.oracle_pre(tr, ctx, G, f, xyxy, conf, cls); cur["f"] = f
        tr.update(_DetSlice(xyxy, conf, cls))
        FT.oracle_post(tr, xyxy, lab)
    return vid, c


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]]; train = [(v, str(rt), sp) for v, sr, rt, sp in S["train"]]
    vids = [v for v, _, _ in dev]; split = dev[0][2]
    rep = {}
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as pool:
        got = list(pool.map(hota_task, [(v, rt, sp, n) for v, rt, sp in dev for n in ("stand", "Olost")]))
        cnt = {name: dict(pool.map(count_task, part)) for name, part in (("dev", dev), ("train", train))}
    per = {n: {v: m for v, nn, m, _ in got if nn == n} for n in ("stand", "Olost")}
    rep["events_Olost"] = {k: int(sum(s[k] for _, nn, _, s in got if nn == "Olost")) for k in ("applied", "changed", "applied_lost", "changed_lost")}
    rep["pooled"] = {n: evaluate_hota(OUT / n, vids, split=split)["HOTA"] for n in ("stand", "Olost")}
    d = np.array([per["Olost"][v]["HOTA"] - per["stand"][v]["HOTA"] for v in vids]); rng = np.random.default_rng(1)
    bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    rep["Olost-stand"] = {"pooled_gain": rep["pooled"]["Olost"] - rep["pooled"]["stand"], "mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
    rep["counts"] = {}
    for name, cc in cnt.items():
        tot = {k: int(sum(x[k] for x in cc.values())) for k in ("pairs", "own_known", "positive", "negative_out_of_crop", "negative_absent", "positive_with_own_det")}
        tot["gap_hist"] = [int(sum(x["gap_hist"][i] for x in cc.values())) for i in range(31)]; tot["videos"] = len(cc)
        rep["counts"][name] = tot
    (OUT / "stats.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    x = rep["Olost-stand"]
    print(f"[HOTA] stand {rep['pooled']['stand']:.3f}, Olost {rep['pooled']['Olost']:.3f}: общий {x['pooled_gain']:+.3f}, по видео {x['mean_per_video']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}] лучше {x['better']}/{x['n']}")
    print(f"[EVENTS] {rep['events_Olost']}")
    for name, t in rep["counts"].items():
        print(f"[COUNT] {name}: " + ", ".join(f"{k} {v}" for k, v in t.items() if k != "gap_hist"))
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
