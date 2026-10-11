"""Один ограниченный разбор после итога life_fly_history_v1 (решение автора 11.10): муха выбирает не те жизни или неверно
оценивает цену удаления? Только объяснение — без нового порога, без новых моделей, без dev.

Прогнозы вне обучения на train 130 (runs/life_fly_history/probs.npz, oof_*), порог 0,7 (допустим для обеих голов).
Удаления делятся на «обе головы», «только с мухой», «только без мухи». Для каждой группы: жизней, по метке (полностью ложная /
настоящая или смешанная / без метки), ложных и верных рамок (n_fp, n_tp), объектов разметки, затронутых удалением верных
рамок; по длине жизни (≤ 5 / > 5 измерений). Полный ответ оценщика при 0,7 (runs/life_fly_history/full_metrics.json:
train_fly_0.7, train_nofly_0.7, train_none) — разности HOTA, DetA, AssA, CLR_FP, CLR_FN по видео (бутстрэп 5000).
Выход — runs/life_fly_history/disputes.json.

Запуск: python scripts/life_fly_history_disputes.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "runs/life_fly_history"
THR = 0.7


def main() -> int:
    L = pickle.load(open(PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl", "rb"))["train"]
    P = np.load(OUT / "probs.npz")
    groups = {g: {"lives": 0, "false": 0, "true_or_mixed": 0, "unlabeled": 0, "fp": 0, "tp": 0, "objects_touched": 0, "lives_gt5": 0, "fp_gt5": 0, "tp_gt5": 0} for g in ("both", "fly_only", "nofly_only")}
    for c in L:
        v = c["video_id"]; rn = P[f"oof_nofly|{v}"] > THR; rf = P[f"oof_fly|{v}"] > THR
        z = np.load(PROJECT_ROOT / "runs/fly_signatures_hist" / v / "signatures.npz"); n = z["end"] - z["start"]
        objs = {}
        for i, g, k in c["tp_map"]:
            objs.setdefault(int(i), set()).add(g)
        for name, m in (("both", rn & rf), ("fly_only", rf & ~rn), ("nofly_only", rn & ~rf)):
            G = groups[name]; lab = c["label"][m]
            G["lives"] += int(m.sum()); G["false"] += int((lab == 1).sum()); G["true_or_mixed"] += int((lab == 0).sum()); G["unlabeled"] += int((lab < 0).sum())
            G["fp"] += int(c["n_fp"][m].sum()); G["tp"] += int(c["n_tp"][m].sum())
            G["objects_touched"] += len(set().union(*[objs.get(int(i), set()) for i in np.nonzero(m)[0]])) if m.any() else 0
            g5 = m & (n > 5); G["lives_gt5"] += int(g5.sum()); G["fp_gt5"] += int(c["n_fp"][g5].sum()); G["tp_gt5"] += int(c["n_tp"][g5].sum())
    F = json.loads((OUT / "full_metrics.json").read_text(encoding="utf-8"))
    fly, nof, none = F[f"train_fly_{THR}"], F[f"train_nofly_{THR}"], F["train_none"]
    vids = [c["video_id"] for c in L]
    rng = np.random.default_rng(1); diff = {}
    for a_name, a, b_name, b in (("fly", fly, "nofly", nof), ("fly", fly, "none", none), ("nofly", nof, "none", none)):
        for k in ("HOTA", "DetA", "AssA", "CLR_FP", "CLR_FN", "CLR_TP"):
            d = np.array([a[v]["metrics"][k] - b[v]["metrics"][k] for v in vids], float); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            diff[f"{a_name}-{b_name}|{k}"] = {"mean_per_video": float(d.mean()), "sum": float(d.sum()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        diff[f"{a_name}-{b_name}|lost_objects"] = {"sum": int(sum(a[v]["lost"] - b[v]["lost"] for v in vids))}
    rep = {"threshold": THR, "videos": len(vids), "groups": groups, "diff": diff}
    (OUT / "disputes.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, G in groups.items():
        print(f"[{g}] жизней {G['lives']} (ложных {G['false']}, настоящих/смешанных {G['true_or_mixed']}, без метки {G['unlabeled']}); рамок ложных {G['fp']}, верных {G['tp']}; объектов затронуто {G['objects_touched']}; "
              f"из них > 5 измерений: жизней {G['lives_gt5']}, ложных рамок {G['fp_gt5']}, верных {G['tp_gt5']}", flush=True)
    for k, x in diff.items():
        if "ci95" in x:
            print(f"   {k}: {x['mean_per_video']:+.4f} [{x['ci95'][0]:+.4f}; {x['ci95'][1]:+.4f}] (сумма {x['sum']:+.1f})", flush=True)
        else:
            print(f"   {k}: {x['sum']:+d}", flush=True)
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
