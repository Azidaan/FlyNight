"""P5 и P10 блока «Проблемы мухи» на событиях потери трека (карточки fly_p5_disputes_v1, fly_p10_data_regime_v1).

Общие данные — события scripts/fly_lifecycle_alive.py (потеря трека стендом, метка «жив / ушёл» по разметке,
признаки на момент потери: геометрия, муха last_fly, вход глаза last_hexin, внешность).

P5 (--p5): доля спорных решений. Головы geometry и geometry_fly (вероятности и пороги из runs/fly_lifecycle/m000).
Действия: «продолжить» (p ≥ порога F1 на dev; только события, где в кадрах end … end + 2 есть слабый кандидат
детектора conf < 0,25 того же класса с IoU ≥ 0,3 к последней рамке) и «удалить» (p < порога удаления).
По каждому действию — доля событий, где головы решают по-разному, и доля верных решений мухи среди расхождений;
разрезы: ночь / день, мелкие / крупные (площадь < 32² px), короткие / длинные жизни (< / ≥ 30 кадров).
Правило: действие стоит пробовать с мухой, если расхождений ≥ 10 % и муха права в ≥ 60 % из них.

P10 (--p10): режим данных. Головы geometry_fly, geometry_hexin, geometry_appearance, geometry обучаются
(a) на 5, 10, 20, 40, 70 случайных видео train (3 выборки на размер), оценка — test;
(b) только на дневных видео train → оценка на ночных test (и ночь → день).
Показатель — AP «жив ли»; разности муха − вход, муха − внешность; бутстреп по видео.

Запуск: python scripts/fly_p5_p10.py --p5 --p10
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fly_lifecycle_alive import HEADS, ap, build_X, sets, video_events  # noqa: E402
from scripts.oracle_detection_ladder import iou_mat  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_p5_p10"


def conditions(vids, split):
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    cond = {}
    for c in ("normal", "night"):
        for v in filter_videos_by_subset(attrs, vids, c):
            cond.setdefault(v, c)
    return cond


def weak_flags(task):
    """Для событий видео: есть ли слабый кандидат рядом с последней рамкой в кадрах end … end + 2."""
    vid, root, rows = task
    from scripts.run_bdd_video_rate_experiment import load_detection_cache
    d = load_detection_cache(Path(root) / vid / "detections" / "raw_video.npz")
    out = {}
    for r in rows:
        box = r["geo"][:4] * np.array([1280, 720, 1280, 720], np.float32)
        cls = int(np.argmax(r["geo"][12:20]))
        sel = np.flatnonzero((d["frame_idx"] >= r["end"]) & (d["frame_idx"] <= r["end"] + 2) & (d["conf"] < 0.25) & (d["cls"].astype(int) == cls))
        out[(vid, r["tid"], r["end"])] = bool(len(sel) and (iou_mat(box[None], d["xyxy"][sel].astype(np.float32))[0] >= 0.3).any())
    return out


def load_events(workers, member="000"):
    sfx = "" if member == "000" else f"_m{member}"
    S = sets(PROJECT_ROOT / f"runs/fly_signatures{sfx}", PROJECT_ROOT / f"runs/fly_signatures{sfx}_val")
    ev = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for s, tasks in S.items():
            ev[s] = dict(pool.map(video_events, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in tasks]))
    return S, ev


def p5(S, ev, workers):
    P = pickle.load(open(PROJECT_ROOT / "runs/fly_lifecycle/m000/probs.pkl", "rb"))
    res = {}
    for s in ("dev", "test"):
        rows_by_v = ev[s]
        root = S[s][0][2]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            weak = {}
            for part in pool.map(weak_flags, [(v, str(root), rows_by_v[v]) for v in rows_by_v if rows_by_v[v]]):
                weak.update(part)
        cond = conditions(list(rows_by_v), "train")
        recs = []
        for v, rows in rows_by_v.items():
            for r in rows:
                k = (v, r["tid"], r["end"])
                if k not in P["probs"]["geometry"]:
                    continue
                area = (r["geo"][2] - r["geo"][0]) * 1280 * (r["geo"][3] - r["geo"][1]) * 720
                recs.append(dict(v=v, y=r["alive"], pg=P["probs"]["geometry"][k], pf=P["probs"]["geometry_fly"][k], weak=weak.get(k, False),
                                 night=cond.get(v) == "night", small=area < 32 ** 2, short=r["geo"][7] < 1.0))  # geo[7] = длина жизни / 30
        res[s] = {}
        for action in ("continue", "delete"):
            for subset, flt in (("all", lambda x: True), ("night", lambda x: x["night"]), ("day", lambda x: not x["night"]),
                                ("small", lambda x: x["small"]), ("large", lambda x: not x["small"]), ("short_life", lambda x: x["short"]), ("long_life", lambda x: not x["short"])):
                rr = [x for x in recs if flt(x) and (x["weak"] if action == "continue" else True)]
                if not rr:
                    continue
                if action == "continue":
                    dg = np.array([x["pg"] >= P["thr"]["geometry"] for x in rr]); df = np.array([x["pf"] >= P["thr"]["geometry_fly"] for x in rr])
                    correct_f = lambda x, d: d == bool(x["y"])
                else:
                    dg = np.array([x["pg"] < P["thr_del"]["geometry"] for x in rr]); df = np.array([x["pf"] < P["thr_del"]["geometry_fly"] for x in rr])
                    correct_f = lambda x, d: d == (not bool(x["y"]))
                dis = np.flatnonzero(dg != df)
                fly_right = float(np.mean([correct_f(rr[i], df[i]) for i in dis])) if len(dis) else float("nan")
                res[s][f"{action}|{subset}"] = {"events": len(rr), "disagree_share": float(len(dis) / len(rr)), "disagree_n": int(len(dis)),
                                                "fly_right_share": fly_right, "geometry_acts_share": float(dg.mean()), "fly_acts_share": float(df.mean())}
                x = res[s][f"{action}|{subset}"]
                print(f"P5 {s:4s} {action:8s} {subset:10s} событий {x['events']:5d}  расхождений {x['disagree_share']:6.1%} ({x['disagree_n']})  муха права {x['fly_right_share']:.1%}  действует: геом {x['geometry_acts_share']:.1%} / муха {x['fly_acts_share']:.1%}", flush=True)
    return res


def fit(rows, parts):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.array([r["alive"] for r in rows])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    return clf.fit(build_X(rows, parts), y)


def eval_heads(models, ev_s, vids, rng, nboot=1000):
    per = {}
    for name, m in models.items():
        per[name] = {v: (np.array([r["alive"] for r in ev_s[v]]), m.predict_proba(build_X(ev_s[v], HEADS[name]))[:, 1]) for v in vids if ev_s[v]}
    vv = [v for v in vids if ev_s[v]]
    aps = {n: ap(np.concatenate([per[n][v][0] for v in vv]), np.concatenate([per[n][v][1] for v in vv])) for n in models}
    diffs = {}
    for a, b in (("geometry_fly", "geometry_hexin"), ("geometry_fly", "geometry_appearance"), ("geometry_fly", "geometry")):
        bs = []
        for _ in range(nboot):
            pick = [vv[i] for i in rng.integers(0, len(vv), len(vv))]
            yy = np.concatenate([per[a][v][0] for v in pick])
            bs.append(ap(yy, np.concatenate([per[a][v][1] for v in pick])) - ap(yy, np.concatenate([per[b][v][1] for v in pick])))
        bs = np.array([x for x in bs if np.isfinite(x)])
        diffs[f"{a}-{b}"] = {"mean": float(aps[a] - aps[b]), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    return aps, diffs


def p10(S, ev):
    rng = np.random.default_rng(0)
    names = ["geometry", "geometry_fly", "geometry_hexin", "geometry_appearance"]
    train_v = [v for v in ev["train"] if ev["train"][v]]; test_v = [v for v in ev["test"] if ev["test"][v]]
    res = {"curve": {}, "shift": {}}
    for n in (5, 10, 20, 40, len(train_v)):
        draws = []
        for d in range(3 if n < len(train_v) else 1):
            pick = list(rng.choice(train_v, n, replace=False))
            rows = [r for v in pick for r in ev["train"][v]]
            if len({r["alive"] for r in rows}) < 2:
                continue
            models = {nm: fit(rows, HEADS[nm]) for nm in names}
            aps, diffs = eval_heads(models, ev["test"], test_v, rng, 500)
            draws.append({"videos": pick, "events": len(rows), "AP": aps, "diff": diffs})
        res["curve"][str(n)] = draws
        m = lambda k: np.mean([dr["diff"][k]["mean"] for dr in draws])
        print(f"P10 обучение на {n:3d} видео ({len(draws)} выб.): муха − вход {m('geometry_fly-geometry_hexin'):+.4f}, муха − внешность {m('geometry_fly-geometry_appearance'):+.4f}, муха − геометрия {m('geometry_fly-geometry'):+.4f}", flush=True)
    cond_tr = conditions(train_v, "train"); cond_te = conditions(test_v, "train")
    for src, dst in (("normal", "night"), ("night", "normal")):
        rows = [r for v in train_v if cond_tr.get(v) == src for r in ev["train"][v]]
        tv = [v for v in test_v if cond_te.get(v) == dst]
        models = {nm: fit(rows, HEADS[nm]) for nm in names}
        aps, diffs = eval_heads(models, ev["test"], tv, rng, 2000)
        res["shift"][f"{src}->{dst}"] = {"train_videos": sum(1 for v in train_v if cond_tr.get(v) == src), "test_videos": len(tv), "AP": aps, "diff": diffs}
        dd = diffs["geometry_fly-geometry_hexin"]
        print(f"P10 сдвиг {src}→{dst}: обучение {res['shift'][f'{src}->{dst}']['train_videos']} видео, проверка {len(tv)}; муха − вход {dd['mean']:+.4f} [{dd['ci95'][0]:+.4f}; {dd['ci95'][1]:+.4f}], муха − внешность {diffs['geometry_fly-geometry_appearance']['mean']:+.4f}", flush=True)
    return res


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--p5", action="store_true")
    ap_.add_argument("--p10", action="store_true")
    ap_.add_argument("--workers", type=int, default=6)
    ap_.add_argument("--member", default="000", help="модель ансамбля FlyVis (сигнатуры runs/fly_signatures_m<id>)")
    ap_.add_argument("--tag", default="", help="приписка к имени выходного файла (например, _train130)")
    args = ap_.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S, ev = load_events(args.workers, args.member)
    if args.p5:
        (OUT / f"p5{args.tag}.json").write_text(json.dumps(p5(S, ev, args.workers), ensure_ascii=False, indent=1), encoding="utf-8")
    if args.p10:
        (OUT / (("p10" if args.member == "000" else f"p10_m{args.member}") + args.tag + ".json")).write_text(json.dumps(p10(S, ev), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
