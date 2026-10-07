"""Обучаемое доверие к движению мухи, исправленный протокол (карточка fly_motion_trust_learned_v2; правило 6.15).

Отличия от fly_motion_trust_learned_v1 (там — порог 0,5 и обучение только на треках с чистым Калманом):
  1) сбор данных в условиях применения: круг 0 — трекер с Калманом; круг 1 — трекер с модулем круга 0, данные объединяются,
     модуль переобучается (как в DAgger);
  2) порог принятия поправки выбирается по HOTA на 18 видео подбора из {0,5; 0,6; 0,7; 0,8}.
Оба модуля — с признаками мухи и без них — проходят одну и ту же процедуру. Проверка — 24 отложенных видео (17 dev + val7).
Критерий: с признаками мухи − без движения > 0, ДИ > 0; с признаками − без признаков > 0, ДИ > 0.

Запуск: python scripts/fly_trust_learned_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_trust_learned as L  # noqa: E402

OUT = PROJECT_ROOT / "runs/motion_scheme_search"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8]


def fit(X, y, cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[:, cols], y)


def gate_of(mdl, cols, thr):
    return lambda x: bool(mdl.predict_proba(x[cols].reshape(1, -1))[0, 1] > thr)


def main() -> int:
    split = json.loads((OUT / "trust_split.json").read_text(encoding="utf-8"))
    tc, vr = PROJECT_ROOT / "runs/train_cache", PROJECT_ROOT / "runs/video_rate"
    tune = [(v, tc, "train") for v in split["tune"]]
    tests = [(v, tc, "train") for v in split["test"]] + [(v, vr, "val") for v in L.M.VAL7]
    base = {**json.loads((OUT / "dev_trust_test.json").read_text(encoding="utf-8"))["r5_all_none"], **json.loads((OUT / "val7_trust_test.json").read_text(encoding="utf-8"))["r5_all_none"]}
    base_tune = json.loads((OUT / "dev_trust_tune.json").read_text(encoding="utf-8"))["r5_all_none"]
    d0 = []
    for v, r, sp in tune:
        L.run_video(v, r, sp, gate=None, collect=d0)
    print(f"[ROUND0] примеров {len(d0)}", flush=True)
    res = {"rounds": {}, "per_video": {}}
    final = {}
    for name, cols in (("with_fly", L.FLY_FEATS), ("no_fly", L.NOFLY_FEATS)):
        X0 = np.array([x for x, _ in d0]); y0 = np.array([t for _, t in d0])
        m0 = fit(X0, y0, cols)
        d1 = []
        for v, r, sp in tune:   # круг 1 — в условиях применения (порог 0,5)
            L.run_video(v, r, sp, gate=gate_of(m0, cols, 0.5), collect=d1)
        X = np.concatenate([X0, np.array([x for x, _ in d1])]); y = np.concatenate([y0, np.array([t for _, t in d1])])
        m1 = fit(X, y, cols)
        print(f"[ROUND1] {name}: примеров +{len(d1)}, всего {len(y)}", flush=True)
        scores = {}
        for thr in THRESHOLDS:
            d = [L.hota(L.run_video(v, r, sp, gate=gate_of(m1, cols, thr)), v, sp, f"{name}_v2_tune_t{thr}") - base_tune[v]["HOTA"] for v, r, sp in tune]
            scores[thr] = float(np.mean(d))
            print(f"[THR] {name} порог {thr}: подбор {scores[thr]:+.3f}", flush=True)
        best = max(THRESHOLDS, key=lambda t: scores[t])
        res["rounds"][name] = {"n": int(len(y)), "tune_by_threshold": scores, "best_threshold": best}
        final[name] = (m1, cols, best)
    for i, (v, r, sp) in enumerate(tests):
        row = {"none": base[v]["HOTA"]}
        for name, (m1, cols, thr) in final.items():
            row[name] = L.hota(L.run_video(v, r, sp, gate=gate_of(m1, cols, thr)), v, sp, f"{name}_v2")
        res["per_video"][v] = row
        print(f"[TEST] {i + 1}/{len(tests)} {v}: без движения {row['none']:.2f} | с мухой {row['with_fly']:.2f} | без признаков мухи {row['no_fly']:.2f}", flush=True)
    rng = np.random.default_rng(1)
    def boot(d):
        bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        return {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "wins": int((d > 0).sum()), "n": int(len(d))}
    P = res["per_video"]; vs = list(P); dev = [v for v in vs if v in split["test"]]; val = [v for v in vs if v not in split["test"]]
    for a_, b_ in (("with_fly", "none"), ("no_fly", "none"), ("with_fly", "no_fly")):
        for nm, sub in (("all24", vs), ("dev17", dev), ("val7", val)):
            res[f"{a_}-{b_}_{nm}"] = boot(np.array([P[v][a_] - P[v][b_] for v in sub]))
        r = res[f"{a_}-{b_}_all24"]
        print(f"== {a_} − {b_}: {r['mean']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}], лучше {r['wins']}/{r['n']} (dev17 {res[f'{a_}-{b_}_dev17']['mean']:+.3f}, val7 {res[f'{a_}-{b_}_val7']['mean']:+.3f})", flush=True)
    (OUT / "trust_learned_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
