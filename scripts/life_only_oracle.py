"""Оракул ровно того действия, что выполняет обучаемая голова жизни (карточка life_only_oracle_v1), только dev.

Поправка автора 09.10: запас life_filter_oracle_v2 (+0,956) получен после идеального удаления ложных цепочек, а голова жизни
применяется к исходному выходу после MASA и удаляет жизни в любых цепочках. Для доли потолка обученной головы нужен оракул
того же действия на том же исходном выходе: из готового выхода (runs/chain_filter_rank/repro/dev) удаляются рамки всех
полностью ложных жизней (запись = номер трекера + интервал, scripts/life_segments.py; правило «ложная» — IoU 0,5,
венгерское по классу, crowd — игнор); жизни без учитываемых рамок не трогаются; номера и связи не меняются.
Отчёт: HOTA, DetA, AssA, CLR_TP/FP/FN общие; потери найденных объектов; разность с исходным выходом по видео (бутстрэп 5000),
ночь/день (datasets.subset_filter); для сравнения — ступени оракула v2 (цепочки; цепочки + жизни).

Запуск: python scripts/life_only_oracle.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SRC = PROJECT_ROOT / "runs/chain_filter_rank/repro/dev"
OUT = PROJECT_ROOT / "runs/life_only_oracle"
KEYS = ("HOTA", "DetA", "AssA", "CLR_TP", "CLR_FP", "CLR_FN")


def video(task):
    from evaluation.hota_eval import evaluate_hota
    from scripts.chain_filter_head import classify_ids
    from scripts.life_segments import box_segments
    from scripts.run_bdd_video_rate_experiment import save_json
    vid, sig_dir, root, split = task
    res = json.loads((SRC / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    seg, st = box_segments(vid, sig_dir, root, res); tags = classify_ids(res, gt)
    cnt = {}
    for tg, sv in zip(tags, seg):
        for (t, _), k in zip(tg, sv):
            if k is not None:
                d = cnt.setdefault(k, {"tp": 0, "fp": 0}); d[t] = d.get(t, 0) + 1
    false = {k for k, d in cnt.items() if d.get("tp", 0) == 0 and d.get("fp", 0) > 0}
    before = {g for tg in tags for t, g in tg if t == "tp"}
    out = {"video_id": vid, "segstats": st, "false_lives": len(false), "metrics": {}}
    for step in ("orig", "lives"):
        rows = [{**fr, "labels": [l for l, k in zip(fr.get("labels") or [], sv) if step == "orig" or k not in false]} for fr, sv in zip(res, seg)]
        d = OUT / step; d.mkdir(parents=True, exist_ok=True); save_json(d / f"{vid}.json", rows)
        m = evaluate_hota(d, [vid], split=split)
        after = {g for tg in classify_ids(rows, gt) for t, g in tg if t == "tp"}
        out["metrics"][step] = {**{k: m.get(k) for k in KEYS}, "lost": len(before - after), "hit": len(before)}
    out["removed_boxes"] = int(sum(1 for sv in seg for k in sv if k in false))
    return out


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    with ProcessPoolExecutor(max_workers=4) as pool:
        per = {r["video_id"]: r for r in pool.map(video, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["dev"]])}
    vids = list(per); split = S["dev"][0][3]
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    night = list(filter_videos_by_subset(attrs, vids, "night"))
    rep = {"pooled": {}, "diff": {}, "lost": {}, "segstats": {k: int(sum(r["segstats"][k] for r in per.values())) for k in next(iter(per.values()))["segstats"]},
           "false_lives": int(sum(r["false_lives"] for r in per.values())), "removed_boxes": int(sum(r["removed_boxes"] for r in per.values()))}
    rng = np.random.default_rng(1)
    for g, gv in (("all", vids), ("night", night)):
        for step in ("orig", "lives"):
            m = evaluate_hota(OUT / step, gv, split=split)
            rep["pooled"][f"{g}|{step}"] = {k: m.get(k) for k in KEYS}
            rep["lost"][f"{g}|{step}"] = {"lost": int(sum(per[v]["metrics"][step]["lost"] for v in gv)), "hit": int(sum(per[v]["metrics"][step]["hit"] for v in gv))}
        for met in ("HOTA", "DetA", "AssA"):
            d = np.array([per[v]["metrics"]["lives"][met] - per[v]["metrics"]["orig"][met] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rep["diff"][f"{g}|lives-orig|{met}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
    v2 = json.loads((PROJECT_ROOT / "runs/life_filter_oracle_v2/report.json").read_text(encoding="utf-8"))["pooled"]
    rep["oracle_v2_ref"] = {f"step{s}": v2[f"all|step{s}"]["HOTA"] for s in (1, 2, 3, 4)}
    P = rep["pooled"]
    print(f"[CHECK] сопоставление {rep['segstats']}; исходный выход {P['all|orig']['HOTA']:.3f} (оракул v2 ступень 1 {rep['oracle_v2_ref']['step1']:.3f})", flush=True)
    for step in ("orig", "lives"):
        x = P[f"all|{step}"]; l_ = rep["lost"][f"all|{step}"]
        print(f"[{step}] HOTA {x['HOTA']:.3f} DetA {x['DetA']:.2f} AssA {x['AssA']:.2f} TP {x['CLR_TP']} FP {x['CLR_FP']} FN {x['CLR_FN']}; потеряно {l_['lost']}/{l_['hit']}", flush=True)
    for k in ("all|lives-orig|HOTA", "all|lives-orig|DetA", "all|lives-orig|AssA", "night|lives-orig|HOTA"):
        r = rep["diff"][k]; print(f"[DIFF] {k}: по видео {r['mean_per_video']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']}", flush=True)
    print(f"[REF] оракул v2: ступени 1–4 {rep['oracle_v2_ref']}; удалено жизней {rep['false_lives']}, рамок {rep['removed_boxes']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
