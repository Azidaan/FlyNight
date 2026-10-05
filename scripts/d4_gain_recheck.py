"""Перепроверка Д4 (fly_decoder_refit_v1): штатная муха с усилением стенда.

Д4 брал поток мухи из кэша runs/video_rate/<id>/motion/flyvis_tiled.npz как есть, а кэш записан с
усилением 1,0 (flow_gain_in_cache); трекер и гейты стенда умножают его на DEFAULT_GAINS["flyvis_tiled"] = 38,84.
Здесь та же метрика Д4 (перенос рамки между соседними размеченными кадрами) для g = 1 (как в Д4), g = 38,84
(стенд) и g = 102,6 (синтетическая калибровка), плюс оптимальный общий g по сетке (верхняя оценка
«только масштаб», подобран на тех же видео — не для заявки).

Запуск: python scripts/d4_gain_recheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.d4_fly_decoder_refit import VAL7, CATS  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "d4_decoder_refit" / "gain_recheck.json"
GAINS = {"g1_as_d4": 1.0, "g_stand": 38.84157, "g_synthetic": 102.59663505027247}
GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]


def main() -> int:
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box
    attrs = load_video_attributes(str(PROJECT_ROOT / "data/bdd100k/labels/bdd100k_labels_images_val.json"))
    cond = {}
    for c in ("normal", "night"):
        for v in filter_videos_by_subset(attrs, VAL7, c):
            cond.setdefault(v, c)
    res = {}
    for v in VAL7:
        fo = np.load(PROJECT_ROOT / "runs/video_rate" / v / "motion/flyvis_tiled.npz")
        dq = np.load(PROJECT_ROOT / "runs/video_rate" / v / "motion/dis_quarter.npz")
        al = json.loads((PROJECT_ROOT / "runs/video_rate" / v / "alignment.json").read_text(encoding="utf-8"))["matches"]
        truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20/val" / f"{v}.json").read_text(encoding="utf-8"))
        nf = min(len(fo["vectors"]), len(dq["vectors"]))
        boxes = [(int(m["video_frame_index"]), {l["id"]: np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32)
                                                for l in fr.get("labels") or [] if l.get("box2d") and l.get("category") in CATS}) for m, fr in zip(al, truth)]
        srcs = {"zero": (fo["points"], np.zeros_like(fo["vectors"])), "dis": (dq["points"], dq["vectors"])}
        srcs.update({k: (fo["points"], fo["vectors"] * g) for k, g in GAINS.items()})
        srcs.update({f"grid_{g}": (fo["points"], fo["vectors"] * g) for g in GRID})
        errs = {k: [] for k in srcs}
        for (r0, b0), (r1, b1) in zip(boxes[:-1], boxes[1:]):
            if r1 <= r0 or r1 >= nf:
                continue
            for tid, box in b0.items():
                if tid in b1:
                    gt = (b1[tid][:2] + b1[tid][2:]) / 2
                    for k, (p, vec) in srcs.items():
                        pr = propagate_box(box, p, vec, r0, r1)[0]
                        errs[k].append(float(np.linalg.norm((pr[:2] + pr[2:]) / 2 - gt)))
        res[v] = {k: float(np.median(e)) for k, e in errs.items()}
        print(f"[ERR] {v} ({cond.get(v, '—')}): " + " | ".join(f"{k} {res[v][k]:.2f}" for k in ("zero", "dis", *GAINS)), flush=True)
    summary = {}
    for g, vs in {"all": VAL7, "night": [v for v in VAL7 if cond.get(v) == "night"], "day": [v for v in VAL7 if cond.get(v) == "normal"]}.items():
        if vs:
            summary[g] = {k: float(np.mean([res[v][k] for v in vs])) for k in res[vs[0]]}
            summary[g]["videos"] = len(vs)
    OUT.write_text(json.dumps({"per_video": res, "summary": summary, "cond": cond, "gains": GAINS}, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, s in summary.items():
        best = min(GRID, key=lambda x: s[f"grid_{x}"])
        print(f"== {g}: zero {s['zero']:.2f} dis {s['dis']:.2f} g1 {s['g1_as_d4']:.2f} стенд {s['g_stand']:.2f} синт {s['g_synthetic']:.2f} | лучший g по сетке {best}: {s[f'grid_{best}']:.2f}")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
