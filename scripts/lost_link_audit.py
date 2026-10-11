"""Аудит А1: неисправленные разрывы свыше 30 кадров на готовом выходе после MASA и запас связей (карточка lost_link_audit_v1).

Решение автора 11.10 (раздел 7.10 плана). Только диагностика на dev 35, без прогона трекера и без мухи.
Выход — жадная сшивка «база + MASA», порог 0,6 (runs/chain_filter_rank/repro/dev; номера — итоговые цепочки).
Сопоставление рамки с эталоном — правило оценщика (scripts/chain_filter_head.classify_ids: венгерское по классу, IoU ≥ 0,5,
crowd — игнор). Для каждого объекта разметки — последовательность его попаданий (опорный кадр, кадр видео, номер выхода).
  разрыв — смена номера между соседними попаданиями объекта; пропуск — кадр первого попадания нового номера минус кадр
    последнего попадания старого (кадры видео);
  длинный — пропуск > 30 кадров (буфер стенда 30);
  допустимый — старый и новый номер ни разу не встречаются в одном опорном кадре (слияние не даёт двух рамок с одним номером);
  в окне повторного сопоставления — допустимый, пропуск ≤ 150 кадров и расстояние центров ≤ 5 размеров последней рамки
    старого номера (окно сшивки stitch_eval_v2).
Запас связей — HOTA (TrackEval) после слияния номеров по разметке (объединение номеров по выбранным разрывам; если после
слияния в кадре два одинаковых номера — вторая рамка сохраняет свой номер, конфликт считается):
  links_long — все допустимые длинные разрывы; links_long_window — только длинные в окне; links_all — все допустимые (справочно).
Критерий остановки (до расчёта): если links_long даёт на dev общий прирост HOTA < +0,3 или ДИ средней по видео разности
содержит 0 — А1 остановить.

Запуск: python scripts/lost_link_audit.py
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
OUT = PROJECT_ROOT / "runs/lost_link_audit"
LONG, WIN_GAP, WIN_DIST = 30, 150, 5.0
KEYS = ("HOTA", "DetA", "AssA", "IDF1", "IDSw")


def merge_ids(res, sel):
    """Слияние номеров выхода по выбранным разрывам (union-find). Возвращает (кадры с новыми номерами, число конфликтов)."""
    parent = {}
    max_id = max((int(l["id"]) for fr in res for l in (fr.get("labels") or [])), default=0)

    def root_of(x):
        while x in parent:
            x = parent[x]
        return x
    for b in sel:
        ro, rn = root_of(b["old"]), root_of(b["new"])
        if ro != rn:
            parent[rn] = ro
    rows, conf = [], 0
    for fr in res:
        seen, labs = set(), []
        for l in fr.get("labels") or []:
            nid = root_of(str(l["id"]))
            if nid in seen:   # конфликт после слияния цепочкой связей — рамке новый уникальный номер (свой может быть уже занят)
                # целые номера — требование TrackEval; сразу за наибольшим номером видео (огромный номер заставляет TrackEval
                # строить таблицу длиной в номер — 6,7 ГБ памяти на процесс, сбой 11.10)
                nid = str(l["id"]) if str(l["id"]) not in seen else str(max_id + 1 + conf); conf += 1
            seen.add(nid); labs.append({**l, "id": nid})
        rows.append({**fr, "labels": labs})
    return rows, conf


def video(task):
    from evaluation.hota_eval import evaluate_hota
    from scripts.chain_filter_head import classify_ids
    from scripts.run_bdd_video_rate_experiment import save_json
    vid, root, split = task
    res = json.loads((SRC / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    vf = [int(m["video_frame_index"]) for m in json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]]
    tags = classify_ids(res, gt)
    frames_of = {}
    seq = {}
    for k, (fr, tg) in enumerate(zip(res, tags)):
        for l, (t, g) in zip(fr.get("labels") or [], tg):
            frames_of.setdefault(str(l["id"]), set()).add(k)
            if t == "tp":
                b = l["box2d"]; seq.setdefault(g, []).append((k, vf[k], str(l["id"]), (b["x1"], b["y1"], b["x2"], b["y2"])))
    breaks = []
    for g, s in seq.items():
        s.sort()
        for a, b in zip(s, s[1:]):
            if a[2] == b[2]:
                continue
            gap = b[1] - a[1]
            legal = not (frames_of[a[2]] & frames_of[b[2]])
            ba, bb = np.array(a[3]), np.array(b[3])
            size = np.sqrt(max((ba[2] - ba[0]) * (ba[3] - ba[1]), 1.0))
            dist = float(np.linalg.norm((ba[:2] + ba[2:]) / 2 - (bb[:2] + bb[2:]) / 2) / size)
            breaks.append({"gt": g, "old": a[2], "new": b[2], "gap": int(gap), "legal": bool(legal), "dist": dist,
                           "long": gap > LONG, "window": bool(legal and gap > LONG and gap <= WIN_GAP and dist <= WIN_DIST)})
    out = {"video_id": vid, "breaks": breaks, "objects": len(seq), "metrics": {}, "conflicts": {}}

    def relabel(sel, tag):
        rows, conf = merge_ids(res, sel)
        d = OUT / tag; d.mkdir(parents=True, exist_ok=True); save_json(d / f"{vid}.json", rows)
        m = evaluate_hota(d, [vid], split=split)
        out["metrics"][tag] = {k: m.get(k) for k in KEYS}; out["conflicts"][tag] = conf
    relabel([], "asis")
    relabel([b for b in breaks if b["legal"] and b["long"]], "links_long")
    relabel([b for b in breaks if b["window"]], "links_long_window")
    relabel([b for b in breaks if b["legal"]], "links_all")
    return out


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    with ProcessPoolExecutor(max_workers=4) as pool:
        per = {r["video_id"]: r for r in pool.map(video, [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]])}
    vids = list(per); split = S["dev"][0][3]
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    night = list(filter_videos_by_subset(attrs, vids, "night"))
    B = [b for r in per.values() for b in r["breaks"]]
    cnt = lambda f: int(sum(1 for b in B if f(b)))
    rep = {"counts": {"objects": int(sum(r["objects"] for r in per.values())), "breaks": len(B), "gap_le_30": cnt(lambda b: not b["long"]),
                      "gap_30_150": cnt(lambda b: b["long"] and b["gap"] <= WIN_GAP), "gap_gt_150": cnt(lambda b: b["gap"] > WIN_GAP),
                      "long_legal": cnt(lambda b: b["long"] and b["legal"]), "long_window": cnt(lambda b: b["window"]), "long_illegal": cnt(lambda b: b["long"] and not b["legal"]),
                      "objects_with_long_break": len({(v, b["gt"]) for v, r in per.items() for b in r["breaks"] if b["long"]})},
           "pooled": {}, "diff": {}, "conflicts": {t: int(sum(r["conflicts"][t] for r in per.values())) for t in ("links_long", "links_long_window", "links_all")}}
    rng = np.random.default_rng(1)
    for g, gv in (("all", vids), ("night", night)):
        for tag in ("asis", "links_long", "links_long_window", "links_all"):
            m = evaluate_hota(OUT / tag, gv, split=split)
            rep["pooled"][f"{tag}|{g}"] = {k: m.get(k) for k in KEYS}
        for tag in ("links_long", "links_long_window", "links_all"):
            d = np.array([per[v]["metrics"][tag]["HOTA"] - per[v]["metrics"]["asis"]["HOTA"] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rep["diff"][f"{tag}-asis|{g}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
    P = rep["pooled"]; gain = P["links_long|all"]["HOTA"] - P["asis|all"]["HOTA"]; d = rep["diff"]["links_long-asis|all"]
    rep["stop"] = {"pooled_gain_links_long": gain, "criterion_met": bool(gain >= 0.3 and d["ci95"][0] > 0)}
    print(f"[COUNTS] {rep['counts']}", flush=True)
    for g in ("all", "night"):
        print(f"[HOTA] {g}: как есть {P[f'asis|{g}']['HOTA']:.3f} (AssA {P[f'asis|{g}']['AssA']:.2f}, IDSw {P[f'asis|{g}']['IDSw']}); " +
              "; ".join(f"{t} {P[f'{t}|{g}']['HOTA']:.3f} (AssA {P[f'{t}|{g}']['AssA']:.2f}, IDSw {P[f'{t}|{g}']['IDSw']}), по видео {rep['diff'][f'{t}-asis|{g}']['mean_per_video']:+.3f} [{rep['diff'][f'{t}-asis|{g}']['ci95'][0]:+.3f}; {rep['diff'][f'{t}-asis|{g}']['ci95'][1]:+.3f}]" for t in ("links_long", "links_long_window", "links_all")), flush=True)
    print(f"[CONFLICTS] {rep['conflicts']}", flush=True)
    print(f"[STOP] {rep['stop']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
