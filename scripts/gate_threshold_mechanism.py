"""Г0а: объясняет ли неточная рамка занижение уверенности (карточка threshold_mechanism_iou_v1).

Зачем. Цель уверенности при обучении Ultralytics умножается на IoU лучшей рамки
(`ultralytics/utils/tal.py:140-142`): объект с неточной рамкой учат оценивать низко.
Если ночной (и «новорождённый») дефицит на пороге — от худшей локализации, то при
равном IoU внутри класса и размера разрыв должен сократиться.

Объекты: эталонные рамки, у которых есть кандидат того же класса с IoU ≥ 0,5
(ступени НАЙДЕН и ПОРОГ лестницы `miss_ladder.py`). Потеря на пороге — уверенность
лучшего такого кандидата < 0,25. IoU объекта — IoU этого самого кандидата
(именно он решает, найден объект или нет); для проверки — также максимальный IoU.

Прямая стандартизация: разрыв = Σ_s w_A(s)·(L_A(s) − L_B(s)) по общим стратам s,
веса — доли группы A (ночь; возраст 0). Контроль — страты класс × размер; проверка —
класс × размер × полоса IoU. Сокращение = 1 − разрыв_IoU / разрыв_контроль.
ДИ — бутстреп по видео. Разметка используется только для оценки.

Запуск: python scripts/gate_threshold_mechanism.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.birth_cost import iou_mat, load_gt_tracks, size_class  # noqa: E402

CACHE = PROJECT_ROOT / "runs" / "det_cache_ctl_960" / "val"
OUT = PROJECT_ROOT / "runs" / "gate_g0a"
IOU_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def collect(conf_thr: float) -> pd.DataFrame:
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes

    files = sorted(CACHE.glob("*.npz"))
    attrs = load_video_attributes(str(PROJECT_ROOT / "data/bdd100k/labels/bdd100k_labels_images_val.json"))
    cond_of: dict[str, str] = {}
    for cond in ("normal", "night"):
        for v in filter_videos_by_subset(attrs, sorted(f.stem for f in files), cond):
            cond_of.setdefault(v, cond)
    occl = {}
    rows = []
    for f in files:
        cond = cond_of.get(f.stem)
        if cond is None:
            continue
        data = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20/val" / f"{f.stem}.json").read_text(encoding="utf-8"))
        occl = {(fr["frameIndex"], l["id"]): bool((l.get("attributes") or {}).get("occluded", False)) for fr in data for l in fr.get("labels") or []}
        per_frame, _, age_of = load_gt_tracks(f.stem)
        d = np.load(f)
        names, fidx = d["frame_names"], d["frame_idx"]
        xyxy, conf, cls = d["xyxy"], d["conf"], d["cls"].astype(np.int32)
        for fi in range(len(names)):
            gt = per_frame.get(fi)
            if gt is None or len(gt[0]) == 0:
                continue
            gb, gc, tids = gt
            sel = fidx == fi
            db, dc, dconf = xyxy[sel], cls[sel], conf[sel]
            if len(db) == 0:
                continue
            m = np.where(gc[:, None] == dc[None, :], iou_mat(gb, db), -1.0)
            ok = m >= 0.5
            has = ok.any(1)
            cg = np.where(ok, dconf[None, :], -1.0)
            jb = cg.argmax(1)
            sz = size_class(gb)
            for k in np.flatnonzero(has):
                rows.append((f.stem, cond, int(gc[k]), int(sz[k]), age_of[(tids[k], fi)], occl.get((fi, tids[k]), False),
                             float(cg[k, jb[k]]), float(m[k, jb[k]]), float(m[k].max())))
    df = pd.DataFrame(rows, columns=["video", "cond", "cls", "size", "age", "occluded", "conf", "iou_best", "iou_max"])
    df["loss"] = (df["conf"] < conf_thr).astype(float)
    df["iou_bin"] = np.digitize(df["iou_best"], IOU_BINS) - 1
    df["iou_max_bin"] = np.digitize(df["iou_max"], IOU_BINS) - 1
    return df


def std_gap(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> float:
    """Σ_s w_A(s)(L_A(s) − L_B(s)) по общим стратам."""
    ga = a.groupby(keys)["loss"].agg(["mean", "size"])
    gb = b.groupby(keys)["loss"].mean()
    common = ga.index.intersection(gb.index)
    ga = ga.loc[common]
    w = ga["size"] / ga["size"].sum()
    return float((w * (ga["mean"] - gb.loc[common])).sum())


def gaps(df: pd.DataFrame, mask_a, mask_b, base: list[str], iou_key: str) -> dict:
    a, b = df[mask_a], df[mask_b]
    raw = float(a["loss"].mean() - b["loss"].mean())
    ctl = std_gap(a, b, base)
    iou = std_gap(a, b, base + [iou_key])
    return {"raw": raw, "ctl": ctl, "iou": iou, "reduction": 1 - iou / ctl if ctl else float("nan")}


def comparisons(df: pd.DataFrame, iou_key: str) -> dict:
    out = {"night_vs_day": gaps(df, df.cond == "night", df.cond == "normal", ["cls", "size"], iou_key)}
    out["age0_vs_10plus"] = gaps(df, df.age == 0, df.age >= 10, ["cls", "size", "cond"], iou_key)
    n = df[df.cond == "night"]
    out["night_age0_vs_10plus"] = gaps(n, n.age == 0, n.age >= 10, ["cls", "size"], iou_key)
    out["occluded_vs_not_night"] = gaps(n, n.occluded, ~n.occluded, ["cls", "size"], iou_key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect(args.conf)
    df.to_csv(OUT / "objects.csv.gz", index=False)
    print(f"объектов с кандидатом IoU>=0.5: {len(df)}; видео {df.video.nunique()} (ночь {df[df.cond=='night'].video.nunique()})", flush=True)

    res = {"n_objects": int(len(df)), "conf_thr": args.conf, "point": {}, "ci95": {}}
    for key in ("iou_bin", "iou_max_bin"):
        res["point"][key] = comparisons(df, key)
    # conf и доля потерь по полосам IoU — для описания механизма
    tab = df.groupby(["cond", "iou_bin"]).agg(n=("loss", "size"), loss=("loss", "mean"), conf_median=("conf", "median")).reset_index()
    tab.to_csv(OUT / "loss_by_iou.csv", index=False)
    res["spearman_conf_iou"] = {c: float(df[df.cond == c][["conf", "iou_best"]].corr("spearman").iloc[0, 1]) for c in ("normal", "night")}

    rng = np.random.default_rng(0)
    vids = {c: df[df.cond == c].video.unique() for c in ("normal", "night")}
    groups = {v: g for v, g in df.groupby("video")}
    boots: dict[str, list] = {}
    for b in range(args.boot):
        pick = np.concatenate([rng.choice(vids[c], len(vids[c])) for c in vids])
        bd = pd.concat([groups[v] for v in pick], ignore_index=True)
        for name, r in comparisons(bd, "iou_bin").items():
            for k, v in r.items():
                boots.setdefault(f"{name}.{k}", []).append(v)
        if (b + 1) % 200 == 0:
            print(f"[boot] {b + 1}/{args.boot}", flush=True)
    res["ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boots.items()}
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, r in res["point"]["iou_bin"].items():
        ci = res["ci95"][f"{name}.reduction"]
        print(f"{name}: raw {r['raw']:+.4f} ctl {r['ctl']:+.4f} iou {r['iou']:+.4f} сокращение {r['reduction']:.1%} [{ci[0]:.1%}; {ci[1]:.1%}]")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
