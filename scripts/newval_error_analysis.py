"""Разбор ошибок фильтра жизней на 97 новых val-видео при зафиксированном пороге 0,7 (карточка newval_error_analysis_v1).

Без обучения, без GPU, порог и модели не меняются (только объяснение ошибки). Данные — runs/newval_eval/lives.pkl (метки,
признаки по интервалу записи), runs/newval_eval/probs.npz (оценки голов без мухи и с мухой). Группы при пороге 0,7 для
каждой головы:
  ошибочно удалённые настоящие (метка 0 — есть верная рамка; «настоящие» без ложных и «смешанные» отдельно),
  правильно удалённые ложные (метка 1), пропущенные ложные (метка 1, оценка ≤ 0,7), оставленные настоящие (справочно).
Для групп: число, длительность (кадров), число наблюдений, средняя уверенность детектора, класс, ночь/день, оценки обеих
голов, учитываемые рамки. Качество верхушки ранжирования: точность среди k жизней с наибольшей оценкой (k = 50, 100,
200, 400) и по корзинам оценки. Решения, которые меняет муха: удаляет с мухой и не удаляет без (и наоборот) — по меткам.
Правило (записано до расчёта): история мухи внутри жизни оправдана как следующий шаг, только если среди ошибочно удалённых
настоящих жизней (голова с мухой, 0,7) доля жизней с > 10 наблюдений (вне окна сигнатуры — первые 5 и последние 5) ≥ 30 %.

Запуск: python scripts/newval_error_analysis.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EV = PROJECT_ROOT / "runs/newval_eval"
OUT = PROJECT_ROOT / "runs/newval_error_analysis"
THR = 0.7
CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(EV / "lives.pkl", "rb")); zz = np.load(EV / "probs.npz")
    groups = json.loads((EV / "report.json").read_text(encoding="utf-8"))["groups"]
    night = set(groups["night"])
    rows = []
    for c in L:
        com = c["nofly"][:, :8]; cls = c["nofly"][:, 8:16]
        for i in range(len(c["label"])):
            if c["label"][i] < 0:
                continue
            rows.append({"vid": c["video_id"], "label": int(c["label"][i]), "kind": "false" if c["label"][i] == 1 else ("mixed" if c["n_fp"][i] > 0 else "true"),
                         "length": float(np.expm1(com[i, 0]) + 1), "obs": float(np.expm1(com[i, 1])), "conf": float(com[i, 2]), "cls": CLASSES[int(np.argmax(cls[i]))],
                         "night": c["video_id"] in night, "p_nofly": float(zz[f"nofly|{c['video_id']}"][i]), "p_fly": float(zz[f"fly|{c['video_id']}"][i]),
                         "boxes": int(c["n_tp"][i] + c["n_fp"][i]), "tp": int(c["n_tp"][i]), "fp": int(c["n_fp"][i])})
    rep = {"threshold": THR, "n_labeled_lives": len(rows), "groups": {}, "topk": {}, "bins": {}, "flips": {}, "rule": {}}

    def describe(sel):
        if not sel:
            return {"n": 0}
        a = lambda k: np.array([r[k] for r in sel], float)
        cl = {}
        for r in sel:
            cl[r["cls"]] = cl.get(r["cls"], 0) + 1
        return {"n": len(sel), "length_median": float(np.median(a("length"))), "obs_median": float(np.median(a("obs"))), "obs_gt10_share": float((a("obs") > 10).mean()),
                "conf_mean": float(a("conf").mean()), "night_share": float(a("night").mean()), "p_nofly_mean": float(a("p_nofly").mean()), "p_fly_mean": float(a("p_fly").mean()),
                "boxes": int(a("boxes").sum()), "tp_boxes": int(a("tp").sum()), "fp_boxes": int(a("fp").sum()), "classes": dict(sorted(cl.items(), key=lambda x: -x[1])[:5])}
    for head in ("nofly", "fly"):
        rm = lambda r: r[f"p_{head}"] > THR
        g = {"wrong_removed_true": [r for r in rows if rm(r) and r["kind"] == "true"], "wrong_removed_mixed": [r for r in rows if rm(r) and r["kind"] == "mixed"],
             "right_removed_false": [r for r in rows if rm(r) and r["label"] == 1], "missed_false": [r for r in rows if not rm(r) and r["label"] == 1],
             "kept_true": [r for r in rows if not rm(r) and r["label"] == 0]}
        rep["groups"][head] = {k: describe(v) for k, v in g.items()}
        order = sorted(rows, key=lambda r: -r[f"p_{head}"])
        rep["topk"][head] = {str(k): float(np.mean([r["label"] == 1 for r in order[:k]])) for k in (50, 100, 200, 400)}
        bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        rep["bins"][head] = {f"{lo}-{hi}": {"n": int(sum(lo <= r[f"p_{head}"] < hi for r in rows)), "share_false": float(np.mean([r["label"] == 1 for r in rows if lo <= r[f"p_{head}"] < hi])) if any(lo <= r[f"p_{head}"] < hi for r in rows) else None} for lo, hi in bins}
    only_fly = [r for r in rows if r["p_fly"] > THR and r["p_nofly"] <= THR]; only_nofly = [r for r in rows if r["p_nofly"] > THR and r["p_fly"] <= THR]
    rep["flips"] = {"fly_removes_only": {"false": sum(r["label"] == 1 for r in only_fly), "true_or_mixed": sum(r["label"] == 0 for r in only_fly),
                                         "fp_boxes": sum(r["fp"] for r in only_fly), "tp_boxes": sum(r["tp"] for r in only_fly)},
                    "nofly_removes_only": {"false": sum(r["label"] == 1 for r in only_nofly), "true_or_mixed": sum(r["label"] == 0 for r in only_nofly),
                                           "fp_boxes": sum(r["fp"] for r in only_nofly), "tp_boxes": sum(r["tp"] for r in only_nofly)}}
    wr = rep["groups"]["fly"]["wrong_removed_true"]
    rep["rule"] = {"obs_gt10_share_in_wrong_removed_true_fly": wr.get("obs_gt10_share"), "history_justified": bool((wr.get("obs_gt10_share") or 0) >= 0.30)}
    for head in ("nofly", "fly"):
        print(f"=== голова {'без мухи' if head == 'nofly' else 'с мухой'}, порог {THR}", flush=True)
        for k, d in rep["groups"][head].items():
            if d["n"]:
                print(f"  {k:20s} n={d['n']:5d}  длит.мед {d['length_median']:5.0f}  набл.мед {d['obs_median']:5.0f}  >10 набл {d['obs_gt10_share']:.0%}  увер {d['conf_mean']:.2f}  ночь {d['night_share']:.0%}  "
                      f"p без мухи {d['p_nofly_mean']:.2f} / с мухой {d['p_fly_mean']:.2f}  рамок {d['boxes']} (верных {d['tp_boxes']}, ложных {d['fp_boxes']})  классы {d['classes']}", flush=True)
        print(f"  точность верхушки (доля ложных среди k лучших): {rep['topk'][head]}", flush=True)
        print(f"  по корзинам оценки: {rep['bins'][head]}", flush=True)
    print(f"[FLIPS] {rep['flips']}", flush=True)
    print(f"[RULE] {rep['rule']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
