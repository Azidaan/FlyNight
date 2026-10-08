"""Оракул удаления отдельных жизней внутри готовых цепочек после MASA, только dev (карточка life_filter_oracle_v1).

Разбор автора 09.10: проверить, есть ли запас у действия тоньше целой цепочки. Готовый выход сшивки «база + MASA»
(runs/chain_filter_rank/repro/dev — пересборка, совпадающая с исходным) не пересчитывается: итоговые номера и связи
сохраняются, удаляются только рамки. Ступени:
  1 — исходный выход;
  2 — удалены полностью ложные цепочки (все учитываемые рамки ложные);
  3 — ступень 2 + удалены полностью ложные жизни внутри оставшихся цепочек (жизнь — непрерывный фрагмент трекера до
      сшивки; все её учитываемые рамки ложные); жизни без учитываемых рамок не трогаются;
  4 — удалены все ложные рамки (справочный потолок).
Правило «ложная» — как в output_filter_oracle (IoU ≥ 0,5, венгерское по классу, crowd — игнор). Принадлежность рамки
выхода к жизни — по совпадению координат с строкой выхода трекера того же кадра (runs/fly_signatures/<id>/outputs);
сверка — все рамки сопоставлены однозначно, ступень 1 = исходный HOTA.
Отчёт: HOTA, DetA, AssA, CLR_TP/FP/FN общие и по видео; потери ранее найденных объектов — объекты разметки,
сопоставленные на ступени 1 и не сопоставленные после повторного сопоставления оставшихся рамок; удалённые на ступени 3
рамки с IoU ≥ 0,3 / ≥ 0,1. Парные разности по видео (бутстрэп 5000).
Критерий перехода (до прогона): ступень 3 − ступень 2 по HOTA — средняя по видео > 0 с ДИ > 0 и общий прирост ≥ +0,3.

Запуск: python scripts/life_filter_oracle.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.output_filter_oracle import classify  # noqa: E402

SRC = PROJECT_ROOT / "runs/chain_filter_rank/repro/dev"
OUT = PROJECT_ROOT / "runs/life_filter_oracle"
KEYS = ("HOTA", "DetA", "AssA", "CLR_TP", "CLR_FP", "CLR_FN")
MIN_GAIN = 0.3


def matched_gt(res, gt):
    """Множество id объектов разметки, сопоставленных хотя бы с одной рамкой выхода (правило оракула)."""
    from scripts.chain_filter_head import classify_ids
    out = set()
    for i, tg in enumerate(classify_ids(res, gt)):
        for t, g in tg:
            if t == "tp":
                out.add(g)
    return out


def video(task):
    from evaluation.hota_eval import evaluate_hota
    from scripts.run_bdd_video_rate_experiment import save_json
    vid, sig_dir, root, split = task
    res = json.loads((SRC / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    o = np.load(Path(sig_dir) / vid / "signatures.npz")["outputs"]
    by_frame = {}
    for r in o:
        by_frame.setdefault(int(r[0]), []).append(r)
    # принадлежность рамок выхода к жизням (tid трекера)
    life, bad = [], 0
    for i, fr in enumerate(res):
        rows = by_frame.get(int(matches[i]["video_frame_index"]), [])
        cur = []
        for l in fr.get("labels") or []:
            b = l["box2d"]; key = np.array([b["x1"], b["y1"], b["x2"], b["y2"]])
            hit = [int(r[5]) for r in rows if np.allclose(r[1:5], key, atol=1e-2)]
            if len(hit) != 1:
                bad += 1; cur.append(None)
            else:
                cur.append(hit[0])
        life.append(cur)
    tags = classify(res, gt)
    ch, lf = {}, {}
    for fr, tg, lv in zip(res, tags, life):
        for l, (t, mx), tid in zip(fr.get("labels") or [], tg, lv):
            for d, k in ((ch, l["id"]), (lf, tid)):
                x = d.setdefault(k, {"tp": 0, "fp": 0}); x[t] = x.get(t, 0) + 1
    false_chain = {k for k, x in ch.items() if x["tp"] == 0 and x["fp"] > 0}
    false_life = {k for k, x in lf.items() if k is not None and x["tp"] == 0 and x["fp"] > 0}
    found1 = matched_gt(res, gt)
    out = {"video_id": vid, "unmatched_boxes": bad, "metrics": {}, "lost": {}, "removed": {}}
    for step in (1, 2, 3, 4):
        rows_, rm = [], {"n": 0, "lives": set(), "iou_ge_0.3": 0, "iou_ge_0.1": 0}
        for fr, tg, lv in zip(res, tags, life):
            keep = []
            for l, (t, mx), tid in zip(fr.get("labels") or [], tg, lv):
                drop = (step in (2, 3) and l["id"] in false_chain) or (step == 3 and tid in false_life and l["id"] not in false_chain) or (step == 4 and t == "fp")
                if step == 3 and tid in false_life and l["id"] not in false_chain:
                    rm["n"] += 1; rm["lives"].add(tid); rm["iou_ge_0.3"] += int(mx >= 0.3); rm["iou_ge_0.1"] += int(mx >= 0.1)
                if not drop:
                    keep.append(l)
            rows_.append({**fr, "labels": keep})
        d = OUT / f"step{step}"; d.mkdir(parents=True, exist_ok=True); save_json(d / f"{vid}.json", rows_)
        m = evaluate_hota(d, [vid], split=split)
        out["metrics"][step] = {k: m.get(k) for k in KEYS}
        out["lost"][step] = len(found1 - matched_gt(rows_, gt))
        if step == 3:
            out["removed"] = {"boxes": rm["n"], "lives": len(rm["lives"]), "iou_ge_0.3": rm["iou_ge_0.3"], "iou_ge_0.1": rm["iou_ge_0.1"]}
    out["found1"] = len(found1)
    in_kept = {tid for fr, lv in zip(res, life) for l, tid in zip(fr.get("labels") or [], lv) if l["id"] not in false_chain}
    out["counts"] = {"chains": len(ch), "false_chains": len(false_chain), "lives": sum(1 for k in lf if k is not None), "false_lives": len(false_life),
                     "false_lives_in_kept_chains": len(false_life & in_kept)}
    return out


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    cond = json.loads((PROJECT_ROOT / "runs/oracle_ladder/report.json").read_text(encoding="utf-8"))["cond"]
    tasks = [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["dev"]]
    with ProcessPoolExecutor(max_workers=6) as pool:
        per = {r["video_id"]: r for r in pool.map(video, tasks)}
    vids = list(per); split = S["dev"][0][3]
    rep = {"rule": "IoU 0,5, венгерское по классу, crowd — игнор", "min_gain": MIN_GAIN, "unmatched_boxes": int(sum(r["unmatched_boxes"] for r in per.values())),
           "source_HOTA": evaluate_hota(SRC, vids, split=split)["HOTA"], "pooled": {}, "diff": {}, "lost": {}, "removed": {}, "counts": {}}
    print(f"[CHECK] рамок без однозначной жизни: {rep['unmatched_boxes']}; исходный HOTA dev {rep['source_HOTA']:.3f}", flush=True)
    groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
    for g, gv in groups.items():
        if not gv:
            continue
        for step in (1, 2, 3, 4):
            m = evaluate_hota(OUT / f"step{step}", gv, split=split)
            rep["pooled"][f"{g}|step{step}"] = {k: m.get(k) for k in KEYS}
            rep["lost"][f"{g}|step{step}"] = {"lost": int(sum(per[v]["lost"][step] for v in gv)), "found_step1": int(sum(per[v]["found1"] for v in gv))}
        rng = np.random.default_rng(1)
        for a_, b_ in ((3, 2), (2, 1), (3, 1), (4, 2), (4, 1)):
            for met in ("HOTA", "DetA", "AssA"):
                d = np.array([per[v]["metrics"][a_][met] - per[v]["metrics"][b_][met] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                rep["diff"][f"{g}|step{a_}-step{b_}|{met}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
    rep["removed"] = {k: int(sum(per[v]["removed"][k] for v in vids)) for k in ("boxes", "lives", "iou_ge_0.3", "iou_ge_0.1")}
    rep["counts"] = {k: int(sum(per[v]["counts"][k] for v in vids)) for k in next(iter(per.values()))["counts"]}
    P = lambda st, g="all": rep["pooled"][f"{g}|step{st}"]
    d32 = rep["diff"]["all|step3-step2|HOTA"]; gain = P(3)["HOTA"] - P(2)["HOTA"]
    rep["gate"] = bool(d32["mean_per_video"] > 0 and d32["ci95"][0] > 0 and gain >= MIN_GAIN)
    den = P(4)["HOTA"] - P(2)["HOTA"]; rep["share_of_rest"] = gain / den if den > 0 else None
    for st in (1, 2, 3, 4):
        print(f"[STEP {st}] HOTA {P(st)['HOTA']:.3f} DetA {P(st)['DetA']:.2f} AssA {P(st)['AssA']:.2f} TP {P(st)['CLR_TP']} FP {P(st)['CLR_FP']} FN {P(st)['CLR_FN']}; потеряно найденных объектов {rep['lost'][f'all|step{st}']['lost']}/{rep['lost'][f'all|step{st}']['found_step1']}", flush=True)
    for k in ("all|step3-step2|HOTA", "all|step3-step2|DetA", "all|step3-step2|AssA", "all|step2-step1|HOTA", "all|step4-step2|HOTA", "night|step3-step2|HOTA"):
        if k in rep["diff"]:
            r = rep["diff"][k]; print(f"[DIFF] {k}: по видео {r['mean_per_video']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']}", flush=True)
    print(f"[STEP 3] удалено жизней {rep['removed']['lives']}, рамок {rep['removed']['boxes']} (IoU ≥ 0,3 — {rep['removed']['iou_ge_0.3']}, ≥ 0,1 — {rep['removed']['iou_ge_0.1']}); счётчики {rep['counts']}", flush=True)
    print(f"[GATE] ступень 3 − 2: общий {gain:+.3f}, по видео {d32['mean_per_video']:+.3f} [{d32['ci95'][0]:+.3f}; {d32['ci95'][1]:+.3f}]; доля остатка до ступени 4 {rep['share_of_rest']} → критерий {'выполнен' if rep['gate'] else 'НЕ выполнен'}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
