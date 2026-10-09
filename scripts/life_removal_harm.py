"""Разбор удалённых жизней фильтра (бустинг) на сохранённых данных — где сосредоточен вред (карточка life_removal_harm_diag_v1).

План автора 09.10, шаг 4.1. Без обучения и без HOTA: прогнозы — runs/life_filter_boost_hota/probs.npz, метки и признаки —
runs/life_filter_head_v2/lives.pkl, пороги — из dev (без мухи 0,6, с мухой 0,55). Удалённые жизни делятся на полностью
ложные (метка 1), смешанные (есть верные и ложные рамки), настоящие (только верные) и без учитываемых рамок.
Группы: длина жизни (кадров), средняя уверенность детектора, класс, освещение (datasets.subset_filter: night / иначе).
Для каждой группы — число удалённых жизней по типам, удалённые ложные и верные рамки.
Правило «очаг вреда» (записано до расчёта): в группе отношение удалённых верных рамок к удалённым ложным больше чем вдвое
выше общего И группа содержит ≥ 30 % всех удалённых верных рамок. dev — основной набор (порог выбран на нём);
test, val7 — диагностика.

Запуск: python scripts/life_removal_harm.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LIVES = PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl"
PROBS = PROJECT_ROOT / "runs/life_filter_boost_hota/probs.npz"
OUT = PROJECT_ROOT / "runs/life_removal_harm"
CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def bins_of(c, night):
    com = c["nofly"][:, :8]; cls = c["nofly"][:, 8:16]
    length = np.expm1(com[:, 0]) + 1; conf = com[:, 2]
    lb = np.digitize(length, [2, 5, 10, 30]); cb = np.digitize(conf, [0.3, 0.4, 0.5, 0.7])
    return {"length": [["1", "2–4", "5–9", "10–29", "30+"][i] for i in lb], "conf": [["<0,3", "0,3–0,4", "0,4–0,5", "0,5–0,7", "≥0,7"][i] for i in cb],
            "class": [str(int(np.argmax(r))) for r in cls], "light": ["ночь" if night else "не ночь"] * len(length)}


def main() -> int:
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(LIVES, "rb")); z = np.load(PROBS)
    thr = json.loads((PROJECT_ROOT / "runs/life_filter_boost_hota/report.json").read_text(encoding="utf-8"))["thresholds"]
    rep = {"thresholds": thr, "rule": "отношение верных к ложным удалённым рамкам > 2× общего и ≥ 30 % удалённых верных рамок", "sets": {}}
    for s in ("dev", "test", "val7"):
        split = L[s][0]["split"]; vids = [c["video_id"] for c in L[s]]
        attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
        night = set(filter_videos_by_subset(attrs, vids, "night"))
        rep["sets"][s] = {}
        for var in ("nofly", "fly"):
            tot = {"false": 0, "mixed": 0, "true": 0, "uncounted": 0, "fp_removed": 0, "tp_removed": 0, "tp_removed_in_false": 0}
            groups = {k: {} for k in ("length", "conf", "class", "light")}
            share_tp = []
            for c in L[s]:
                p = z[f"{var}|{s}|{c['video_id']}"]; rm = p > thr[var]
                lab = c["label"]; tp = c["n_tp"]; fp = c["n_fp"]
                kind = np.where(lab == 1, "false", np.where(lab < 0, "uncounted", np.where(fp > 0, "mixed", "true")))
                b = bins_of(c, c["video_id"] in night)
                for i in np.nonzero(rm)[0]:
                    tot[kind[i]] += 1; tot["fp_removed"] += int(fp[i]); tot["tp_removed"] += int(tp[i])
                    if kind[i] in ("mixed", "true"):
                        share_tp.append(tp[i] / (tp[i] + fp[i]))
                    for g, vals in b.items():
                        x = groups[g].setdefault(vals[i], {"false": 0, "mixed": 0, "true": 0, "uncounted": 0, "fp": 0, "tp": 0})
                        x[kind[i]] += 1; x["fp"] += int(fp[i]); x["tp"] += int(tp[i])
            ratio_all = tot["tp_removed"] / max(tot["fp_removed"], 1)
            hot = []
            for g, d in groups.items():
                for k, x in d.items():
                    r_ = x["tp"] / max(x["fp"], 1)
                    if r_ > 2 * ratio_all and x["tp"] >= 0.3 * tot["tp_removed"] and tot["tp_removed"] > 0:
                        hot.append(f"{g}={k}")
            rep["sets"][s][var] = {"total": tot, "tp_per_fp": ratio_all, "groups": groups, "hotspots": hot,
                                   "tp_share_in_removed_nonfalse": {"median": float(np.median(share_tp)) if share_tp else None, "mean": float(np.mean(share_tp)) if share_tp else None}}
            print(f"[{s} {var}] удалено жизней: ложных {tot['false']}, смешанных {tot['mixed']}, настоящих {tot['true']}, без учитываемых {tot['uncounted']}; рамок ложных {tot['fp_removed']}, верных {tot['tp_removed']} (верных на ложную {ratio_all:.2f}); очаги: {hot or 'нет'}", flush=True)
            for g in ("length", "conf", "light"):
                print(f"    {g}: " + "; ".join(f"{k}: ложн {x['false']} / смеш {x['mixed']} / наст {x['true']}, рамок F {x['fp']} T {x['tp']}" for k, x in sorted(groups[g].items())), flush=True)
            print(f"    class: " + "; ".join(f"{CLASSES[int(k)] if int(k) < 8 else k}: ложн {x['false']} / смеш {x['mixed']} / наст {x['true']}, F {x['fp']} T {x['tp']}" for k, x in sorted(groups['class'].items())), flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
