"""К16, ступень 1, подготовка: связи FlyWire (FAFB 783, правая зрительная доля) в формате коннектома FlyVis.

FlyVis (`tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json`): 65 типов, 605 рёбер «тип-источник → тип-приёмник»
со списком смещений колонок `[[du, dv], n]` — среднее число синапсов от клеток-источников при этом смещении на одну
клетку-приёмник (усреднение по 7 колонкам FIB-25 и FIB-19). Здесь то же из полных данных FlyWire:
  * типы — visual_neuron_types (правая сторона), соответствие имён — ALIAS; не найденные типы перечисляются;
  * положение клетки — колонка (p, q) из column_assignment; для типов без привязки — среднее (p, q) столбчатых партнёров,
    взвешенное числом синапсов, округлённое до узла решётки;
  * связи — connections_princeton_no_threshold, области правой зрительной доли (_R);
  * на клетку-приёмник: n(du, dv) = Σ синапсов от источников со смещением / число клеток-приёмников типа.
Оси решётки FlyWire (p, q) и FlyVis (u, v) могут отличаться поворотом/отражением/базисом: перебираются целочисленные
матрицы 2×2 с элементами из {−1, 0, 1} и |det| = 1; берётся та, при которой распределения по смещениям лучше совпадают
с FlyVis (средний косинус по рёбрам с ≥ 5 синапсами на клетку в обоих).
Выход: runs/flywire_flyvis/compare.json (сводка), edges.csv (по рёбрам), flywire_counts.json (n по смещениям в осях FlyVis).

Запуск: python scripts/flywire_to_flyvis.py
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FW = PROJECT_ROOT / "data/flywire/783"
FV_JSON = PROJECT_ROOT / "tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json"
OUT = PROJECT_ROOT / "runs/flywire_flyvis"
ALIAS = {**{f"R{i}": "R1-6" for i in range(1, 7)}, "Am": "Lai", "TmY9": ("TmY9q", "TmY9q__perp")}
MIN_SYN = 5.0


def main() -> int:
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    fv = json.load(open(FV_JSON, encoding="utf-8"))
    fv_types = [n["name"] for n in fv["nodes"]]
    vt = pd.read_csv(FW / "visual_neuron_types.csv.gz"); vt = vt[vt.side == "right"]
    ca = pd.read_csv(FW / "column_assignment.csv.gz"); ca = ca[ca.hemisphere == "right"]
    fw_type_of = dict(zip(vt.root_id, vt.type))
    # FlyVis-тип → набор FlyWire-типов
    want = {}
    for t in fv_types:
        a = ALIAS.get(t, t); a = a if isinstance(a, tuple) else (a,)
        if any((vt.type == x).any() for x in a):
            want[t] = a
    missing = [t for t in fv_types if t not in want]
    fw2fv = {x: t for t, a in want.items() for x in a}
    # R1-6 — один тип FlyWire на шесть FlyVis: связи R1-6 считаются для каждого R1..R6 одинаково (оговорка в сводке)
    cells = {rid: fw2fv[ty] for rid, ty in fw_type_of.items() if ty in fw2fv}
    print(f"[TYPES] найдено {len(want)} из {len(fv_types)} типов FlyVis; нет: {missing}", flush=True)
    pos = {int(r.root_id): (int(r.p), int(r.q)) for r in ca.itertuples()}
    print(f"[POS] привязано к колонкам: {sum(1 for c in cells if c in pos)} из {len(cells)} клеток нужных типов", flush=True)
    # связи правой зрительной доли между нужными клетками
    con = pd.read_csv(FW / "connections_princeton_no_threshold.csv.gz", usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
    con = con[con.neuropil.str.endswith("_R")]
    con = con[con.pre_root_id.isin(cells.keys()) & con.post_root_id.isin(cells.keys())]
    con = con.groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum()
    print(f"[CONN] пар нейронов: {len(con)}, синапсов: {int(con.syn_count.sum())}", flush=True)
    # положение для нестолбчатых: среднее (p, q) столбчатых партнёров, взвешенное синапсами
    acc = defaultdict(lambda: [0.0, 0.0, 0.0])
    for a, b, w in con[["pre_root_id", "post_root_id", "syn_count"]].itertuples(index=False):
        for x, y in ((a, b), (b, a)):
            if x not in pos and y in pos:
                p, q = pos[y]; s = acc[x]; s[0] += w * p; s[1] += w * q; s[2] += w
    inferred = 0
    for x, (sp, sq, sw) in acc.items():
        if sw > 0:
            pos[x] = (int(round(sp / sw)), int(round(sq / sw))); inferred += 1
    n_cells = defaultdict(int)
    for c, t in cells.items():
        if c in pos:
            n_cells[t] += 1
    print(f"[POS] выведено по партнёрам: {inferred}; клеток с положением по типам: мин {min(n_cells.values())}, медиана {int(np.median(list(n_cells.values())))}", flush=True)
    # n(src, tar, dp, dq) на клетку-приёмник
    agg = defaultdict(float)
    for a, b, w in con[["pre_root_id", "post_root_id", "syn_count"]].itertuples(index=False):
        if a in pos and b in pos:
            (pa, qa), (pb, qb) = pos[a], pos[b]
            agg[(cells[a], cells[b], pa - pb, qa - qb)] += w
    fwd = defaultdict(dict)
    for (s, t, dp, dq), w in agg.items():
        fwd[(s, t)][(dp, dq)] = w / max(n_cells[t], 1)
    # FlyVis по рёбрам
    fvd = {}
    for e in fv["edges"]:
        fvd[(e["src"], e["tar"])] = {tuple(o[0]): float(o[1]) for o in e["offsets"]}
    # перебор преобразования осей
    mats = [np.array(m).reshape(2, 2) for m in itertools.product((-1, 0, 1), repeat=4) if abs(round(np.linalg.det(np.array(m).reshape(2, 2)))) == 1]
    common = [k for k in fvd if k in fwd and sum(fvd[k].values()) >= MIN_SYN and sum(fwd[k].values()) >= MIN_SYN and k[0] in want and k[1] in want]
    def cos_for(M):
        cs = []
        for k in common:
            f = defaultdict(float)
            for (dp, dq), w in fwd[k].items():
                u, v = M @ np.array([dp, dq]); f[(int(u), int(v))] += w
            keys = set(f) | set(fvd[k])
            a = np.array([f.get(x, 0.0) for x in keys]); b = np.array([fvd[k].get(x, 0.0) for x in keys])
            cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        return float(np.mean(cs)), cs
    scores = [(cos_for(M)[0], i) for i, M in enumerate(mats)]
    scores.sort(reverse=True)
    best = mats[scores[0][1]]
    mean_cos, per_cos = cos_for(best)
    print(f"[AXES] лучшее преобразование (p, q) → (u, v): {best.tolist()} — средний косинус {mean_cos:.3f} (второе {scores[1][0]:.3f}, тождественное {cos_for(np.eye(2, dtype=int))[0]:.3f})", flush=True)
    # итоговые числа в осях FlyVis
    fw_uv = {}
    for k, d in fwd.items():
        f = defaultdict(float)
        for (dp, dq), w in d.items():
            u, v = best @ np.array([dp, dq]); f[(int(u), int(v))] += w
        fw_uv[k] = dict(f)
    rows = []
    for k in sorted(set(fvd) | set(fw_uv)):
        if k[0] not in want or k[1] not in want:
            continue
        a, b = sum(fvd.get(k, {}).values()), sum(fw_uv.get(k, {}).values())
        rows.append({"src": k[0], "tar": k[1], "flyvis_total": a, "flywire_total": b, "in_flyvis": k in fvd})
    import csv
    with open(OUT / "edges.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    both = [r for r in rows if r["in_flyvis"] and r["flyvis_total"] > 0 and r["flywire_total"] > 0]
    la = np.log10([r["flyvis_total"] for r in both]); lb = np.log10([r["flywire_total"] for r in both])
    lost = [r for r in rows if r["in_flyvis"] and r["flywire_total"] < 0.5]
    new = sorted([r for r in rows if not r["in_flyvis"] and r["flywire_total"] >= MIN_SYN], key=lambda r: -r["flywire_total"])
    summary = {"fv_types": len(fv_types), "mapped_types": len(want), "missing_types": missing, "alias": {k: list(v) if isinstance(v, tuple) else v for k, v in ALIAS.items()},
               "cells_with_position": dict(n_cells), "inferred_positions": inferred,
               "axes_matrix_pq_to_uv": best.tolist(), "axes_mean_cosine": mean_cos, "axes_second_best": scores[1][0], "edges_compared_for_axes": len(common),
               "flyvis_edges_mapped": sum(1 for r in rows if r["in_flyvis"]), "edges_both_nonzero": len(both),
               "log_total_corr": float(np.corrcoef(la, lb)[0, 1]), "median_ratio_flywire_over_flyvis": float(np.median(10 ** (lb - la))),
               "flyvis_edges_absent_in_flywire": [(r["src"], r["tar"], round(r["flyvis_total"], 2)) for r in lost],
               "new_edges_in_flywire_ge5": [(r["src"], r["tar"], round(r["flywire_total"], 2)) for r in new[:40]], "n_new_edges_ge5": len(new),
               "per_edge_offset_cosine_median": float(np.median(per_cos)), "note": "R1-6 FlyWire — один тип на R1..R6 FlyVis; CT1 и не найденные типы — без данных FlyWire"}
    (OUT / "compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    json.dump({f"{k[0]}->{k[1]}": [[list(o), w] for o, w in d.items()] for k, d in fw_uv.items()}, open(OUT / "flywire_counts.json", "w", encoding="utf-8"))
    print(f"[CMP] рёбер FlyVis с данными FlyWire: {len(both)}; корреляция log(всего синапсов на клетку) {summary['log_total_corr']:.3f}; медиана FlyWire/FlyVis {summary['median_ratio_flywire_over_flyvis']:.2f}", flush=True)
    print(f"[CMP] рёбер FlyVis, которых нет в FlyWire (< 0,5 син./клетку): {len(lost)}; новых рёбер в FlyWire ≥ 5 син./клетку: {len(new)}", flush=True)
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
