"""Разбор после итога life_fly_history_v1 (один, на сохранённых прогнозах): различает ли голова с мухой ложные жизни лучше.

Прогнозы вне обучения на train 130 (runs/life_fly_history/probs.npz, oof_*) и итоговых голов на dev 35 (dev_*), метки v2
(runs/life_filter_head_v2/lives.pkl). AP (average precision, ложная = 1) головы без мухи и с мухой — на всех размеченных
жизнях и на целевой группе А2 (жизни с > 5 измерениями глаза, n = end − start); разность «с мухой − без мухи» — бутстрэп
по видео (5000). Выход — runs/life_fly_history/ap.json.

Запуск: python scripts/life_fly_history_ap.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "runs/life_fly_history"


def main() -> int:
    from sklearn.metrics import average_precision_score as aps
    L = pickle.load(open(PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl", "rb"))
    P = np.load(OUT / "probs.npz")
    rep = {}
    rng = np.random.default_rng(1)
    for part, key in (("train", "oof"), ("dev", "dev")):
        vids = L[part]
        per = []
        for c in vids:
            z = np.load(PROJECT_ROOT / "runs/fly_signatures_hist" / c["video_id"] / "signatures.npz"); n = z["end"] - z["start"]
            per.append((c["label"], n, P[f"{key}_nofly|{c['video_id']}"], P[f"{key}_fly|{c['video_id']}"]))
        for grp, f in (("all", lambda n: n >= 0), ("gt5", lambda n: n > 5), ("le5", lambda n: n <= 5)):
            def ap(idx, var):
                y = np.concatenate([per[i][0][f(per[i][1])] for i in idx]); p = np.concatenate([per[i][2 + var][f(per[i][1])] for i in idx]); k = y >= 0
                return aps(y[k], p[k]) if (y[k] == 1).any() else np.nan
            allv = np.arange(len(per)); a0, a1 = ap(allv, 0), ap(allv, 1)
            bs = []
            for _ in range(1000):
                s = rng.integers(0, len(per), len(per)); bs.append(ap(s, 1) - ap(s, 0))
            bs = np.array(bs); bs = bs[~np.isnan(bs)]
            nl = int(sum(((x[0] >= 0) & f(x[1])).sum() for x in per)); nf = int(sum(((x[0] == 1) & f(x[1])).sum() for x in per))
            rep[f"{part}|{grp}"] = {"lives": nl, "false": nf, "AP_nofly": float(a0), "AP_fly": float(a1), "diff": float(a1 - a0), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
            r = rep[f"{part}|{grp}"]
            print(f"[AP] {part} {grp}: жизней {nl}, ложных {nf}; без мухи {r['AP_nofly']:.4f}, с мухой {r['AP_fly']:.4f}, разность {r['diff']:+.4f} [{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}]", flush=True)
    (OUT / "ap.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
