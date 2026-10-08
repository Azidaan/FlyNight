"""Более сильная голова отдельной жизни — градиентный бустинг, парно без мухи и с мухой (карточка life_filter_boost_v1).

life_filter_head_v2 (исправленные жизни): линейная голова — AP на dev 0,535 без мухи / 0,539 с мухой, добавка мухи не
доказана. Гипотеза (выбор автора 09.10, вариант 2): линейная голова плохо использует сочетания признаков мухи, уверенности
детектора и длительности. Здесь — те же сохранённые признаки и метки (runs/life_filter_head_v2/lives.pkl), голова —
HistGradientBoostingClassifier (scikit-learn). Обе головы — одинаковая процедура: настройки из небольшой сетки (4) выбираются
по средней AP на train при GroupKFold(3) по видео; итоговая модель — на всём train; dev — заранее записанный критерий.
Критерий (до прогона): AP с мухой − без мухи на dev > 0, ДИ по видео > 0 (бутстрэп 2000). Также — качество удаления при
бюджете потерь 1 % (порог на dev): ложные/верные рамки, потерянные объекты; справочно — AP линейных голов v2.
Дополнительно (диагностика для решения об истории мухи, не критерий): у полностью ложных жизней dev — доля с наблюдениями
вне окна сигнатуры (первые 5 и последние 5 наблюдений) и доля их учитываемых ложных рамок вне этого окна.

Запуск: python scripts/life_filter_boost.py
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.life_filter_head import GRID, SRC, losses  # noqa: E402

LIVES = PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl"
OUT = PROJECT_ROOT / "runs/life_filter_boost"
CONFIGS = [{"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 200}, {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 200},
           {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 400}, {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 400}]
K = 5   # окно сигнатуры мухи: первые и последние K наблюдений (scripts/build_fly_signatures.py)


def model(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(l2_regularization=1.0, early_stopping=False, random_state=0, **cfg)


def interior_task(c):
    """Для полностью ложных жизней видео: число наблюдений, учитываемые ложные рамки внутри и вне окна сигнатуры."""
    from scripts.life_segments import box_segments
    from scripts.output_filter_oracle import classify
    vid = c["video_id"]
    res = json.loads((SRC["dev"] / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / c["split"] / f"{vid}.json").read_text(encoding="utf-8"))
    matches = json.loads((Path(c["root"]) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    seg, _ = box_segments(vid, c["sig_dir"], c["root"], res); tags = classify(res, gt)
    z = np.load(Path(c["sig_dir"]) / vid / "signatures.npz"); o = z["outputs"]
    false = set(np.nonzero(c["label"] == 1)[0].tolist())
    obs = {}
    for i in false:
        f = o[(o[:, 5] == z["track_id"][i]) & (o[:, 0] >= z["start"][i]) & (o[:, 0] <= z["end"][i]), 0].astype(int)
        f = np.sort(f); obs[i] = (len(f), set(f[:K].tolist()) | set(f[-K:].tolist()))
    n_fp_in = n_fp_out = 0
    for k, (tg, sv) in enumerate(zip(tags, seg)):
        frame = int(matches[k]["video_frame_index"])
        for (t, _), i in zip(tg, sv):
            if i in false and t == "fp":
                if frame in obs[i][1]:
                    n_fp_in += 1
                else:
                    n_fp_out += 1
    return {"lives": len(false), "lives_with_interior": int(sum(1 for i in false if obs[i][0] > 2 * K)), "fp_in_window": n_fp_in, "fp_outside_window": n_fp_out,
            "obs": [obs[i][0] for i in false]}


def main() -> int:
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(LIVES, "rb"))
    tr = L["train"]; ytr = np.concatenate([c["label"] for c in tr]); gtr = np.concatenate([np.full(len(c["label"]), i) for i, c in enumerate(tr)]); ok = ytr >= 0
    rep = {"configs": CONFIGS, "cv": {}, "chosen": {}, "AP": {}, "AP_diff": None, "budget": {}, "gate": None}
    probs = {}
    for var in ("nofly", "fly"):
        X = np.concatenate([c[var] for c in tr])[ok]; y = ytr[ok]; g = gtr[ok]
        cv = []
        for ci, cfg in enumerate(CONFIGS):
            aps = []
            for a_, b_ in GroupKFold(3).split(X, y, g):
                aps.append(average_precision_score(y[b_], model(cfg).fit(X[a_], y[a_]).predict_proba(X[b_])[:, 1]))
            cv.append(float(np.mean(aps)))
            print(f"[CV] {var} конфигурация {ci} {cfg}: AP train (3 части по видео) {cv[-1]:.4f}", flush=True)
        best = int(np.argmax(cv)); rep["cv"][var] = cv; rep["chosen"][var] = best
        m = model(CONFIGS[best]).fit(X, y)
        probs[var] = {c["video_id"]: m.predict_proba(c[var])[:, 1] for c in L["dev"]}
        print(f"[FIT] {var}: выбрана конфигурация {best}", flush=True)
    np.savez_compressed(OUT / "probs_dev.npz", **{f"{var}|dev|{v}": p for var, d in probs.items() for v, p in d.items()})
    lab = {c["video_id"]: c["label"] for c in L["dev"]}; vids = list(lab)

    def ap(var, vs):
        y = np.concatenate([lab[v] for v in vs]); p = np.concatenate([probs[var][v] for v in vs]); k = y >= 0
        return float(average_precision_score(y[k], p[k]))
    for var in ("nofly", "fly"):
        rep["AP"][var] = ap(var, vids)
    rng = np.random.default_rng(1); bs = []
    for _ in range(2000):
        vs = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
        try:
            bs.append(ap("fly", vs) - ap("nofly", vs))
        except ValueError:
            pass
    rep["AP_diff"] = {"mean": rep["AP"]["fly"] - rep["AP"]["nofly"], "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    yall = np.concatenate([lab[v] for v in vids]); rep["random_AP"] = float((yall == 1).sum() / (yall >= 0).sum())
    lin = json.loads((PROJECT_ROOT / "runs/life_filter_head_v2/report_dev.json").read_text(encoding="utf-8"))["AP"]
    rep["linear_AP"] = {"nofly": lin["nofly|dev"], "fly": lin["fly|dev"]}
    d = rep["AP_diff"]
    print(f"[AP] dev: бустинг без мухи {rep['AP']['nofly']:.4f}, с мухой {rep['AP']['fly']:.4f}, разность {d['mean']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}]; "
          f"линейные v2 {lin['nofly|dev']:.4f} / {lin['fly|dev']:.4f}; случайное ранжирование {rep['random_AP']:.4f}", flush=True)
    for var in ("nofly", "fly"):
        grid = {t: losses(L["dev"], probs[var], t) for t in GRID}
        okt = [t for t in GRID if grid[t]["lost_share"] is not None and grid[t]["lost_share"] <= 0.01]
        thr = max(okt, key=lambda t: (grid[t]["fp_removed"], t)) if okt else None
        rep["budget"][var] = {"threshold": thr, "dev": grid[thr] if thr is not None else None}
        if thr is not None:
            r = grid[thr]
            print(f"[BUDGET 1 %] {var}: порог {thr}; dev — ложных рамок убрано {r['fp_removed']}, верных {r['tp_removed']}, потеряно {r['lost']}/{r['hit']}, точность удаления {r['precision']:.3f}", flush=True)
    rep["gate"] = bool(d["mean"] > 0 and d["ci95"][0] > 0)
    print(f"[GATE] AP с мухой − без мухи на dev {d['mean']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}] → критерий {'выполнен' if rep['gate'] else 'НЕ выполнен'}", flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        it = list(pool.map(interior_task, L["dev"]))
    obs = np.concatenate([np.asarray(x["obs"]) for x in it]) if it else np.zeros(0)
    rep["interior"] = {"false_lives": int(sum(x["lives"] for x in it)), "with_interior": int(sum(x["lives_with_interior"] for x in it)),
                       "fp_in_window": int(sum(x["fp_in_window"] for x in it)), "fp_outside_window": int(sum(x["fp_outside_window"] for x in it)),
                       "obs_median": float(np.median(obs)) if len(obs) else None, "obs_q90": float(np.quantile(obs, .9)) if len(obs) else None}
    ii = rep["interior"]
    print(f"[INTERIOR] полностью ложных жизней dev {ii['false_lives']}: с наблюдениями вне окна сигнатуры (> {2 * K}) {ii['with_interior']} ({ii['with_interior'] / max(ii['false_lives'], 1):.1%}); "
          f"учитываемых ложных рамок в окне {ii['fp_in_window']}, вне окна {ii['fp_outside_window']}; наблюдений медиана {ii['obs_median']}, 90-й процентиль {ii['obs_q90']}", flush=True)
    (OUT / "report_dev.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
