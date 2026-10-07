"""К16-1 (fly_flywire_swap_v1): сравнение признаков мухи трёх вариантов связей на «объект ещё здесь?».

Варианты сигнатур: stock (runs/fly_signatures), flywire (runs/fly_signatures_k16_flywire), shuf (runs/fly_signatures_k16_shuf).
События и метки «жив через ≤ 30 кадров» — scripts/fly_lifecycle_alive.py:video_events (трекер от мухи не зависит, события
вариантов совпадают — проверяется). Головы (логистическая регрессия, как в fly_lifecycle_alive) обучаются на dev 35,
проверяются на test 35: geometry, geometry_fly, geometry_hexin, geometry_hexin_fly. Сравнения AP на test с бутстрэпом
по видео (2000): flywire − shuf и flywire − stock для geometry_fly и geometry_hexin_fly; «прибавка мухи» (geometry_fly −
geometry) для каждого варианта.

Запуск: python scripts/k16_compare_heads.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fly_lifecycle_alive import HEADS, ap, build_X, video_events  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_flywire_swap"
ROOTS = {"stock": PROJECT_ROOT / "runs/fly_signatures", "flywire": PROJECT_ROOT / "runs/fly_signatures_k16_flywire", "shuf": PROJECT_ROOT / "runs/fly_signatures_k16_shuf"}
USE = ["geometry", "geometry_fly", "geometry_hexin", "geometry_hexin_fly"]


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    tc = str(PROJECT_ROOT / "runs/train_cache")
    vids = [v for v in sel["dev"] + sel["test"] if all((r / v / "signatures.npz").exists() for r in ROOTS.values())]
    dev = [v for v in sel["dev"] if v in vids]; test = [v for v in sel["test"] if v in vids]
    print(f"видео со всеми тремя вариантами: dev {len(dev)}, test {len(test)}", flush=True)
    ev = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name, root in ROOTS.items():
            ev[name] = dict(pool.map(video_events, [(v, str(root), tc, "train") for v in vids]))
    key = lambda r: (r["tid"], r["end"], r["alive"])
    same = all([key(r) for r in ev["stock"][v]] == [key(r) for r in ev[n][v]] for n in ("flywire", "shuf") for v in vids)
    print(f"события вариантов совпадают: {same}", flush=True)
    preds = {}
    for name in ROOTS:
        tr = [r for v in dev for r in ev[name][v]]; y = np.array([r["alive"] for r in tr])
        for h in USE:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(build_X(tr, HEADS[h]), y)
            preds[(name, h)] = {v: (np.array([r["alive"] for r in ev[name][v]]), clf.predict_proba(build_X(ev[name][v], HEADS[h]))[:, 1]) for v in test if ev[name][v]}
    tv = [v for v in test if all(v in preds[(n, "geometry")] for n in ROOTS)]
    res = {"dev": len(dev), "test": len(tv), "events_identical": bool(same), "AP": {}, "diff": {}}
    for (name, h), pv in preds.items():
        yy = np.concatenate([pv[v][0] for v in tv]); pp = np.concatenate([pv[v][1] for v in tv])
        res["AP"][f"{name}|{h}"] = ap(yy, pp)
    rng = np.random.default_rng(0)
    comps = [(("flywire", "geometry_fly"), ("shuf", "geometry_fly")), (("flywire", "geometry_fly"), ("stock", "geometry_fly")), (("stock", "geometry_fly"), ("shuf", "geometry_fly")),
             (("flywire", "geometry_hexin_fly"), ("shuf", "geometry_hexin_fly")), (("flywire", "geometry_hexin_fly"), ("stock", "geometry_hexin_fly")),
             (("stock", "geometry_fly"), ("stock", "geometry")), (("flywire", "geometry_fly"), ("flywire", "geometry")), (("shuf", "geometry_fly"), ("shuf", "geometry")),
             (("stock", "geometry_hexin_fly"), ("stock", "geometry_hexin")), (("flywire", "geometry_hexin_fly"), ("flywire", "geometry_hexin")), (("shuf", "geometry_hexin_fly"), ("shuf", "geometry_hexin"))]
    for a, b in comps:
        bs = []
        for _ in range(2000):
            pick = [tv[i] for i in rng.integers(0, len(tv), len(tv))]
            ya = np.concatenate([preds[a][v][0] for v in pick])
            bs.append(ap(ya, np.concatenate([preds[a][v][1] for v in pick])) - ap(ya, np.concatenate([preds[b][v][1] for v in pick])))
        bs = np.array([x for x in bs if np.isfinite(x)])
        k = f"{a[0]}:{a[1]} − {b[0]}:{b[1]}"
        res["diff"][k] = {"mean": res["AP"][f"{a[0]}|{a[1]}"] - res["AP"][f"{b[0]}|{b[1]}"], "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        r = res["diff"][k]
        print(f"Δ {k:55s} {r['mean']:+.4f} [{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}]", flush=True)
    for k, v in res["AP"].items():
        print(f"  AP {k:32s} {v:.4f}", flush=True)
    (OUT / "heads_compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
