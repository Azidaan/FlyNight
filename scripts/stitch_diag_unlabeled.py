"""Диагностика сшивки (карточка stitch_unlabeled_diag_v1): сколько принятых связей задевают концы без эталона и сколько HOTA они стоят.

Гипотеза после stitch_global_assign_v1: головы обучаются на 2 % кандидатов (оба конца сопоставлены с эталоном), а сшивают все —
98 % пар задевают ложные или несопоставленные треки; AP по помеченным парам этого не видит. Здесь для голов base и base_masa
на их порогах из runs/stitch_v2/gap150_dist5/report.json (жадный сшиватель) считаются:
  * состав принятых связей: верные (y = 1), неверные (y = 0), без эталона у конца A / у начала B / у обоих;
  * HOTA сшивки при наборах кандидатов (все признаки и вероятности те же, меняется только допуск пар — разметкой, как оракул):
      all        — все кандидаты (воспроизведение report.json);
      known      — только пары с эталоном у обоих концов (прежняя постановка с утечкой);
      no_unk     — принятые связи без связей с концами без эталона (сколько стоят именно они);
      no_wrong   — принятые связи без неверных (y = 0);
      correct    — только верные принятые связи (потолок точности при этой полноте).
Парный бутстрэп по видео (5000) — разность с all. test 35 и val7.

Запуск: python scripts/stitch_diag_unlabeled.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stitch_eval_v2 as V  # noqa: E402

BASE = PROJECT_ROOT / "runs/stitch_v2/gap150_dist5"
OUT = PROJECT_ROOT / "runs/stitch_unlabeled_diag"
HEADS = ["base", "base_masa"]


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    thr_of = {h: json.loads((BASE / "report.json").read_text(encoding="utf-8"))["stitch"][h]["threshold"] for h in HEADS}
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    data = {}
    with ProcessPoolExecutor(max_workers=6, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        for s in ("train", "test", "val7"):
            data[s] = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]))
            print(f"[DATA] {s}: видео {len(data[s])}, кандидатов {sum(len(d['y']) for d in data[s])}", flush=True)
    meta = {s: (S[s][0][1], S[s][0][2], S[s][0][3]) for s in data}
    Xtr = np.concatenate([d["X"][d["y"] >= 0] for d in data["train"]]); ytr = np.concatenate([d["y"][d["y"] >= 0] for d in data["train"]])
    rep = {"thresholds": thr_of, "composition": {}, "hota": {}, "diff": {}}
    rng = np.random.default_rng(1)
    for h in HEADS:
        idx = [V.NAMES.index(c) for c in V.HEADS[h]]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr[:, idx], ytr)
        thr = thr_of[h]
        for s in ("test", "val7"):
            probs = {d["video_id"]: m.predict_proba(d["X"][:, idx])[:, 1] for d in data[s]}

            def run(name, select):
                d_out = OUT / h / name / s
                tasks = []
                for d in data[s]:
                    k = select(d)
                    tasks.append((d["video_id"], *meta[s], d["pairs"][k].reshape(-1, 2), probs[d["video_id"]][k], thr, str(d_out)))
                with ProcessPoolExecutor(max_workers=6) as pool:
                    per = {r["video_id"]: r for r in pool.map(stitch_video, tasks)}
                agg = evaluate_hota(d_out, [d["video_id"] for d in data[s]], split=meta[s][2])
                return per, {k: agg.get(k) for k in ("HOTA", "IDF1", "AssA", "IDSw")}

            allk = {d["video_id"]: np.arange(len(d["y"])) for d in data[s]}
            per_all, pooled_all = run("all", lambda d: allk[d["video_id"]])
            acc = {v: np.asarray(per_all[v]["accepted"], np.int64) for v in per_all}   # индексы в полном списке кандидатов
            comp = {"links": 0, "correct": 0, "wrong": 0, "unk_a": 0, "unk_b": 0, "unk_both": 0}
            for d in data[s]:
                k = acc[d["video_id"]]; y = d["y"][k]; kd = d["kind"][k]
                comp["links"] += len(k); comp["correct"] += int((y == 1).sum()); comp["wrong"] += int((y == 0).sum())
                comp["unk_a"] += int((kd == 1).sum()); comp["unk_b"] += int((kd == 2).sum()); comp["unk_both"] += int((kd == 3).sum())
            rep["composition"][f"{h}|{s}"] = comp
            print(f"[COMP] {h} {s} порог {thr}: связей {comp['links']} — верных {comp['correct']}, неверных {comp['wrong']}, без эталона: у A {comp['unk_a']}, у B {comp['unk_b']}, у обоих {comp['unk_both']}", flush=True)
            variants = {
                "known": lambda d: np.nonzero(d["y"] >= 0)[0],
                "no_unk": lambda d: acc[d["video_id"]][d["y"][acc[d["video_id"]]] >= 0],
                "no_wrong": lambda d: acc[d["video_id"]][d["y"][acc[d["video_id"]]] != 0],
                "correct": lambda d: acc[d["video_id"]][d["y"][acc[d["video_id"]]] == 1],
            }
            rep["hota"][f"{h}|{s}|all"] = pooled_all
            print(f"[HOTA] {h} {s} all: {pooled_all['HOTA']:.3f}", flush=True)
            for name, sel in variants.items():
                per, pooled = run(name, sel)
                rep["hota"][f"{h}|{s}|{name}"] = pooled
                dd = np.array([per[v]["metrics"]["HOTA"] - per_all[v]["metrics"]["HOTA"] for v in per])
                b = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                rep["diff"][f"{h}|{s}|{name}-all"] = {"mean": float(dd.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "better": int((dd > 0).sum()), "n": len(dd)}
                r = rep["diff"][f"{h}|{s}|{name}-all"]
                print(f"[HOTA] {h} {s} {name}: {pooled['HOTA']:.3f} (Δ {pooled['HOTA'] - pooled_all['HOTA']:+.3f}; по видео {r['mean']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']})", flush=True)
            (OUT / "summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
