"""Запас фильтрации готового выхода по меткам при IoU = 0,5 (карточка output_filter_oracle_v1, вариант А).

Готовый выход трекера (scalabel по опорным кадрам) не пересчитывается: из него удаляются рамки по разметке (оракул).
Два выхода: стенд без сшивки (runs/stitch_v2/gap150_dist5/stitch/geometry_2) и сшивка «база + MASA» с жадным
сшивателем (…/stitch/base_masa_0.6). Ступени:
  1 — без фильтра (воспроизведение исходного HOTA);
  2 — удалить целые цепочки (итоговый id), у которых все учитываемые рамки ложные;
  3 — удалить все ложные рамки (ступень 2 + ложные рамки смешанных цепочек).
Диагностическое правило «ложная рамка» (явно заданное, не полное правило HOTA — метрика усредняет пороги 0,05–0,95 и
учитывает согласованность номеров): в кадре и классе — венгерское сопоставление с эталоном при IoU ≥ 0,5 (как
предобработка TrackEval BDD100K); несопоставленная рамка, лежащая > 50 % (IoA) в области crowd, — игнорируется (как в
TrackEval), остальные несопоставленные — ложные. Классы вне восьми оцениваемых TrackEval отбрасывает — здесь тоже.
Для удалённых рамок — сколько из них имели IoU ≥ 0,3 / ≥ 0,1 с эталоном того же класса (верные на менее строгих порогах).
Метрики — evaluation/hota_eval.evaluate_hota: общая по набору (pooled) и по видео; парные разности ступеней — по видео,
бутстрэп 5000. Доля запаса (HOTA₂ − HOTA₁)/(HOTA₃ − HOTA₁) — по общей метрике, отдельно для каждого выхода, только при
положительном знаменателе. Ночь/день — разметка условий из runs/oracle_ladder/report.json (cond).

Запуск: python scripts/output_filter_oracle.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SRC = PROJECT_ROOT / "runs/stitch_v2/gap150_dist5/stitch"
OUTPUTS = {"stand": "geometry_2", "masa_stitch": "base_masa_0.6"}
OUT = PROJECT_ROOT / "runs/output_filter_oracle"
VALID = ("pedestrian", "rider", "car", "bus", "truck", "train", "motorcycle", "bicycle")
KEYS = ("HOTA", "DetA", "AssA", "CLR_TP", "CLR_FP", "CLR_FN", "IDSw")


def _box(l):
    b = l["box2d"]; return [b["x1"], b["y1"], b["x2"], b["y2"]]


def _iou(A, B, ioa=False):
    A = np.asarray(A, np.float64).reshape(-1, 4); B = np.asarray(B, np.float64).reshape(-1, 4)
    if not len(A) or not len(B):
        return np.zeros((len(A), len(B)))
    x1 = np.maximum(A[:, None, 0], B[None, :, 0]); y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2]); y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(A[:, 2] - A[:, 0], 0, None) * np.clip(A[:, 3] - A[:, 1], 0, None)
    bb = np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None)
    den = aa[:, None] if ioa else aa[:, None] + bb[None, :] - inter
    return inter / np.maximum(den, 1e-12)


def classify(res: list, gt: list) -> list:
    """Для каждой рамки выхода: 'tp' / 'fp' / 'ign' / 'skip' (класс не оценивается) и max IoU с эталоном своего класса."""
    from scipy.optimize import linear_sum_assignment
    out = []
    for fr_r, fr_g in zip(res, gt):
        R = fr_r.get("labels") or []; G = fr_g.get("labels") or []
        tags = [["skip", 0.0] for _ in R]
        crowd = [_box(g) for g in G if (g.get("attributes") or {}).get("crowd", False)]
        for c in VALID:
            ri = [i for i, l in enumerate(R) if l.get("category") == c]
            gi = [j for j, g in enumerate(G) if g.get("category") == c and not (g.get("attributes") or {}).get("crowd", False)]
            if not ri:
                continue
            S = _iou([_box(R[i]) for i in ri], [_box(G[j]) for j in gi])
            matched = set()
            if S.size:
                M = S.copy(); M[M < 0.5 - np.finfo(float).eps] = 0
                rr, cc = linear_sum_assignment(-M.T)       # строки — эталон, как в TrackEval
                for g_, r_ in zip(rr, cc):
                    if M.T[g_, r_] > np.finfo(float).eps:
                        matched.add(r_)
            ioa = _iou([_box(R[i]) for i in ri], crowd, ioa=True)
            for k, i in enumerate(ri):
                mx = float(S[k].max()) if S.size else 0.0
                if k in matched:
                    tags[i] = ["tp", mx]
                elif ioa.size and (ioa[k] > 0.5 + np.finfo(float).eps).any():
                    tags[i] = ["ign", mx]
                else:
                    tags[i] = ["fp", mx]
        out.append(tags)
    return out


def filter_video(task):
    """Ступени 2 и 3 для одного видео; записывает отфильтрованные выходы и считает HOTA по видео."""
    from evaluation.hota_eval import evaluate_hota
    from scripts.run_bdd_video_rate_experiment import save_json
    out_name, vid, split_dir, label_split, out_root = task
    res = json.loads((SRC / OUTPUTS[out_name] / split_dir / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / label_split / f"{vid}.json").read_text(encoding="utf-8"))
    tags = classify(res, gt)
    per_id = {}
    for fr, tg in zip(res, tags):
        for l, (t, _) in zip(fr.get("labels") or [], tg):
            d = per_id.setdefault(l["id"], {"tp": 0, "fp": 0, "ign": 0, "skip": 0}); d[t] += 1
    fully_false = {i for i, d in per_id.items() if d["tp"] == 0 and d["fp"] > 0}
    mixed = {i for i, d in per_id.items() if d["tp"] > 0 and d["fp"] > 0}
    stats = {"ids": len(per_id), "fully_false_ids": len(fully_false), "mixed_ids": len(mixed),
             "true_ids": sum(1 for d in per_id.values() if d["tp"] > 0 and d["fp"] == 0),
             "boxes_tp": sum(d["tp"] for d in per_id.values()), "boxes_fp": sum(d["fp"] for d in per_id.values()),
             "boxes_ign": sum(d["ign"] for d in per_id.values()), "boxes_fp_in_mixed": sum(per_id[i]["fp"] for i in mixed)}
    result = {"video_id": vid, "stats": stats, "metrics": {}, "removed": {}}
    for step in (1, 2, 3):
        rows, rm = [], {"n": 0, "iou_ge_0.3": 0, "iou_ge_0.1": 0}
        for fr, tg in zip(res, tags):
            keep = []
            for l, (t, mx) in zip(fr.get("labels") or [], tg):
                drop = (step >= 2 and l["id"] in fully_false and t == "fp") or (step == 3 and t == "fp")
                if drop:
                    rm["n"] += 1; rm["iou_ge_0.3"] += int(mx >= 0.3); rm["iou_ge_0.1"] += int(mx >= 0.1)
                else:
                    keep.append(l)
            rows.append({**fr, "labels": keep})
        d = Path(out_root) / out_name / f"step{step}" / split_dir
        d.mkdir(parents=True, exist_ok=True); save_json(d / f"{vid}.json", rows)
        m = evaluate_hota(d, [vid], split=label_split)
        result["metrics"][step] = {k: m.get(k) for k in KEYS}; result["removed"][step] = rm
    return result


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    from scripts.build_dino_embeddings import VAL7
    cond = json.loads((PROJECT_ROOT / "runs/oracle_ladder/report.json").read_text(encoding="utf-8"))["cond"]
    sets = {"test": (sel["test"], "train"), "val7": (VAL7, "val")}
    sets = {s: ([v for v in vids if (SRC / "geometry_2" / s / f"{v}.json").exists()], sp) for s, (vids, sp) in sets.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"rule": "IoU 0,5, венгерское по классу, crowd IoA > 0,5 — игнор (TrackEval BDD100K)", "outputs": OUTPUTS, "pooled": {}, "per_video": {}, "diff": {}, "share": {}, "stats": {}, "removed": {}}
    rng = np.random.default_rng(1)
    for out_name in OUTPUTS:
        for s, (vids, sp) in sets.items():
            with ProcessPoolExecutor(max_workers=6) as pool:
                per = {r["video_id"]: r for r in pool.map(filter_video, [(out_name, v, s, sp, str(OUT)) for v in vids])}
            key = f"{out_name}|{s}"
            rep["per_video"][key] = {v: {str(k): x for k, x in r["metrics"].items()} for v, r in per.items()}
            rep["stats"][key] = {k: int(sum(r["stats"][k] for r in per.values())) for k in next(iter(per.values()))["stats"]}
            rep["removed"][key] = {str(st): {k: int(sum(r["removed"][st][k] for r in per.values())) for k in ("n", "iou_ge_0.3", "iou_ge_0.1")} for st in (2, 3)}
            groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
            for g, gv in groups.items():
                if not gv:
                    continue
                for step in (1, 2, 3):
                    m = evaluate_hota(OUT / out_name / f"step{step}" / s, gv, split=sp)
                    rep["pooled"][f"{key}|{g}|step{step}"] = {k: m.get(k) for k in KEYS}
                    if step == 1 and g == "all":
                        src = evaluate_hota(SRC / OUTPUTS[out_name] / s, gv, split=sp)
                        rep["pooled"][f"{key}|all|source"] = {k: src.get(k) for k in KEYS}
                for a_, b_ in ((2, 1), (3, 1), (3, 2)):
                    for met in ("HOTA", "DetA", "AssA"):
                        d = np.array([per[v]["metrics"][a_][met] - per[v]["metrics"][b_][met] for v in gv])
                        bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                        rep["diff"][f"{key}|{g}|step{a_}-step{b_}|{met}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
                P = lambda st: rep["pooled"][f"{key}|{g}|step{st}"]["HOTA"]
                den = P(3) - P(1)
                rep["share"][f"{key}|{g}"] = (P(2) - P(1)) / den if den > 0 else None
            pl = rep["pooled"]; df = rep["diff"]
            print(f"[{key}] источник HOTA {pl[key + '|all|source']['HOTA']:.3f}; ступени (общая): " + " / ".join(f"{pl[f'{key}|all|step{st}']['HOTA']:.3f}" for st in (1, 2, 3))
                  + f"; DetA " + " / ".join(f"{pl[f'{key}|all|step{st}']['DetA']:.2f}" for st in (1, 2, 3))
                  + f"; AssA " + " / ".join(f"{pl[f'{key}|all|step{st}']['AssA']:.2f}" for st in (1, 2, 3)), flush=True)
            for a_, b_ in ((2, 1), (3, 1)):
                r = df[f"{key}|all|step{a_}-step{b_}|HOTA"]
                print(f"[{key}] ступень {a_} − {b_}: HOTA по видео {r['mean_per_video']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']}", flush=True)
            tp1 = rep["pooled"][f"{key}|all|step1"]["CLR_TP"]; fp1 = rep["pooled"][f"{key}|all|step1"]["CLR_FP"]
            rep["rule_check"] = rep.get("rule_check", {}); rep["rule_check"][key] = {"tp_rule": rep["stats"][key]["boxes_tp"], "CLR_TP": tp1, "fp_rule": rep["stats"][key]["boxes_fp"], "CLR_FP": fp1}
            print(f"[{key}] сверка правила с TrackEval: верных {rep['stats'][key]['boxes_tp']} vs CLR_TP {tp1}, ложных {rep['stats'][key]['boxes_fp']} vs CLR_FP {fp1}", flush=True)
            print(f"[{key}] доля запаса (общая) {rep['share'][key + '|all']}; цепочек {rep['stats'][key]}; удалено {rep['removed'][key]}", flush=True)
            (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
