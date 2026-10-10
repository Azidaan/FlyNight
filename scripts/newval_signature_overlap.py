"""Сколько коротких жизней имеют совпадающие первое и последнее средние мухи и какую долю запаса они дают (карточка newval_signature_overlap_v1).

Разбор автора 10.10: сигнатура мухи — средние первых и последних K = 5 измерений глаза (scripts/build_fly_signatures.py:181);
если у жизни не больше пяти измерений, оба средних строятся из одних данных и порядок изменений теряется. Перед вариантом
«история мухи внутри жизни» — измерить на сохранённых данных 97 новых val-видео (только диагностика):
  * доля жизней с совпадающими средними (n_first = n_last и first_fly == last_fly; n_first/n_last — число измерений глаза
    в окнах, из signatures.npz) — среди всех размеченных, полностью ложных, ошибочно и правильно удаляемых (голова с мухой, 0,7);
  * доля запаса: оракул, удаляющий только полностью ложные жизни с совпадающими средними, против полного оракула
    (runs/newval_oracle) — HOTA ночь / день / все.

Запуск: python scripts/newval_signature_overlap.py
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.life_filter_head import KEYS, apply_task  # noqa: E402

EV = PROJECT_ROOT / "runs/newval_eval"
OUT = PROJECT_ROOT / "runs/newval_signature_overlap"
SIG, CACHE = PROJECT_ROOT / "runs/fly_signatures_newval", PROJECT_ROOT / "runs/newval_cache"


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(EV / "lives.pkl", "rb")); zz = np.load(EV / "probs.npz")
    groups = json.loads((EV / "report.json").read_text(encoding="utf-8"))["groups"]
    full = json.loads((PROJECT_ROOT / "runs/newval_oracle/report.json").read_text(encoding="utf-8"))["pooled"]
    same = {}
    stats = {k: [0, 0] for k in ("labeled", "false", "true_or_mixed", "removed_false", "removed_true", "missed_false")}
    fp_boxes = [0, 0]
    for c in L:
        z = np.load(SIG / c["video_id"] / "signatures.npz")
        s = (z["n_first"] == z["n_last"]) & np.all(np.isclose(z["first_fly"].reshape(len(z["n_first"]), -1).astype(np.float32), z["last_fly"].reshape(len(z["n_first"]), -1).astype(np.float32), atol=1e-4), 1)
        same[c["video_id"]] = s
        p = zz[f"fly|{c['video_id']}"] > 0.7; lab = c["label"]
        for k, m in (("labeled", lab >= 0), ("false", lab == 1), ("true_or_mixed", lab == 0), ("removed_false", p & (lab == 1)), ("removed_true", p & (lab == 0)), ("missed_false", ~p & (lab == 1))):
            stats[k][0] += int((m & s).sum()); stats[k][1] += int(m.sum())
        fp_boxes[0] += int(c["n_fp"][(lab == 1) & s].sum()); fp_boxes[1] += int(c["n_fp"][lab == 1].sum())
    rep = {"share_same": {k: {"same": a, "all": b, "share": a / max(b, 1)} for k, (a, b) in stats.items()},
           "fp_boxes_in_false_lives": {"same": fp_boxes[0], "all": fp_boxes[1], "share": fp_boxes[0] / max(fp_boxes[1], 1)}, "oracle": {}}
    tk = [(c["video_id"], str(SIG), str(CACHE), "val", "newval", np.nonzero((c["label"] == 1) & same[c["video_id"]])[0], str(OUT / "oracle_same"), True) for c in L]
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(apply_task, tk))
    for g, gv in groups.items():
        agg = evaluate_hota(OUT / "oracle_same", gv, split="val")
        none_h, full_h = full[f"none|{g}"]["HOTA"], full[f"oracle|{g}"]["HOTA"]
        rep["oracle"][g] = {"none": none_h, "oracle_same_only": agg["HOTA"], "oracle_full": full_h, "share_of_gain": (agg["HOTA"] - none_h) / (full_h - none_h) if full_h > none_h else None}
    for k, v in rep["share_same"].items():
        print(f"[SAME] {k:15s}: совпадающие средние у {v['same']} из {v['all']} ({v['share']:.1%})", flush=True)
    f = rep["fp_boxes_in_false_lives"]; print(f"[SAME] ложных рамок в ложных жизнях с совпадающими средними: {f['same']} из {f['all']} ({f['share']:.1%})", flush=True)
    for g, o in rep["oracle"].items():
        print(f"[ORACLE] {g}: без фильтра {o['none']:.3f}, оракул только по таким жизням {o['oracle_same_only']:.3f}, полный оракул {o['oracle_full']:.3f}; доля запаса {o['share_of_gain']:.1%}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
