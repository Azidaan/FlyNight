"""Что мешает сшивке (карточка fly_stitch_oracle_v1): голова узнавания или окна поиска кандидатов.

Склейка обрывков даёт до +6,1 HOTA на test (fly_assoc_target_v1, decompose.json), сшивка по геометрии берёт +0,84
(stitch_appearance_v1). Здесь — идеальная сшивка: пары кандидатов «конец трека A → начало трека B» строятся как в
scripts/eval_fly_reid.py:video_pairs (разрыв ≤ G кадров, расстояние ≤ D размеров рамки, оба конца сопоставлены с эталоном),
сшиваются только настоящие (метка по эталону), тем же жадным сшивателем stitch_video. Окна: (60; 3) — нынешнее,
(150; 5), (300; 8). HOTA test 35 и val7 против «как есть» (стенд).

Запуск: python scripts/fly_stitch_oracle.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs" / "fly_stitch_oracle"
WINDOWS = [(60, 3.0), (150, 5.0), (300, 8.0)]


def _set(gap, dist):
    import scripts.eval_fly_reid as E
    E.MAX_GAP, E.MAX_DIST = gap, dist


def job(task):
    vid, sig, root, split, out_dir = task
    import scripts.eval_fly_reid as E
    d = E.video_pairs((vid, sig, root, split))
    pairs = d["pairs"][:, :2] if len(d["y"]) else np.zeros((0, 2), np.int32)
    probs = d["y"].astype(np.float64) if len(d["y"]) else np.zeros(0)
    r = E.stitch_video((vid, sig, root, split, pairs, probs, 0.5, out_dir))
    return vid, r["metrics"]["HOTA"], int(r["links"]), int(len(d["y"])), int(d["y"].sum()) if len(d["y"]) else 0


def main() -> int:
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    base = {}
    for s in ("test", "val7"):
        split = S[s][0][3]
        from evaluation.hota_eval import evaluate_hota
        base[s] = {v: evaluate_hota(PROJECT_ROOT / "runs/fly_assoc_target/relabel/asis" / split, [v], split=split)["HOTA"] for v, *_ in S[s]}
    res = {"windows": {}}
    rng = np.random.default_rng(1)
    for gap, dist in WINDOWS:
        key = f"gap{gap}_dist{dist:g}"; res["windows"][key] = {}
        for s in ("test", "val7"):
            tasks = [(v, str(sr), str(rt), sp, str(OUT / key / s)) for v, sr, rt, sp in S[s]]
            with ProcessPoolExecutor(max_workers=6, initializer=_set, initargs=(gap, dist)) as pool:
                rows = list(pool.map(job, tasks))
            dd = np.array([h - base[s][v] for v, h, *_ in rows]); b = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
            r = {"mean_hota_gain": float(dd.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "links": int(sum(x[2] for x in rows)),
                 "candidates": int(sum(x[3] for x in rows)), "true_candidates": int(sum(x[4] for x in rows)), "per_video": {v: h for v, h, *_ in rows}}
            res["windows"][key][s] = r
            print(f"[ORACLE] окно {key} {s}: идеальная сшивка Δ HOTA {r['mean_hota_gain']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}], связей {r['links']}, кандидатов {r['candidates']} (настоящих {r['true_candidates']})", flush=True)
            (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
