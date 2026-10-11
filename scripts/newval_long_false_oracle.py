"""Адресная проверка основания А2 (решение автора 11.10): точное число измерений глаза и отдельный оракул для группы.

Число измерений глаза за жизнь восстанавливается точно: n = end − start (проверено: n_first = n_last = min(5, n) у всех
235 тыс. жизней dev, 40 видео train и 97 новых val; совпадение двух средних — только при n ≤ 5). На 97 новых val
(runs/newval_eval/lives.pkl, метки v2) — описание полностью ложных жизней по n (≤ 5 / > 5: число, ложные рамки) и
оракул, удаляющий только полностью ложные жизни с n > 5 (тот же путь, что оракул newval_signature_overlap_v1:
life_filter_head.apply_task). HOTA — без фильтра / этот оракул / полный оракул (runs/newval_oracle) по группам ночь /
день / все, каждое число отдельно; приросты разных оракулов не вычитаются и не складываются.
Выход — runs/newval_long_false_oracle/report.json.

Запуск: python scripts/newval_long_false_oracle.py
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

from scripts.life_filter_head import apply_task  # noqa: E402

EV = PROJECT_ROOT / "runs/newval_eval"
OUT = PROJECT_ROOT / "runs/newval_long_false_oracle"
SIG, CACHE = PROJECT_ROOT / "runs/fly_signatures_newval", PROJECT_ROOT / "runs/newval_cache"
K = 5


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(EV / "lives.pkl", "rb"))
    groups = json.loads((EV / "report.json").read_text(encoding="utf-8"))["groups"]
    full = json.loads((PROJECT_ROOT / "runs/newval_oracle/report.json").read_text(encoding="utf-8"))["pooled"]
    cnt = {"false_le5": [0, 0], "false_gt5": [0, 0], "true_le5": [0, 0], "true_gt5": [0, 0]}   # [жизней, ложных рамок]
    drop = {}
    for c in L:
        z = np.load(SIG / c["video_id"] / "signatures.npz"); n = z["end"] - z["start"]
        assert np.all(z["n_first"] == np.minimum(K, n)) and np.all(z["n_last"] == np.minimum(K, n)), c["video_id"]
        lab = c["label"]
        for key, m in (("false_le5", (lab == 1) & (n <= K)), ("false_gt5", (lab == 1) & (n > K)), ("true_le5", (lab == 0) & (n <= K)), ("true_gt5", (lab == 0) & (n > K))):
            cnt[key][0] += int(m.sum()); cnt[key][1] += int(c["n_fp"][m].sum())
        drop[c["video_id"]] = np.nonzero((lab == 1) & (n > K))[0]
    tk = [(c["video_id"], str(SIG), str(CACHE), "val", "newval", drop[c["video_id"]], str(OUT / "oracle_false_gt5"), True) for c in L]
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(apply_task, tk))
    rep = {"counts": {k: {"lives": a, "fp_boxes": b} for k, (a, b) in cnt.items()}, "oracle": {}}
    for g, gv in groups.items():
        agg = evaluate_hota(OUT / "oracle_false_gt5", gv, split="val")
        rep["oracle"][g] = {"none": full[f"none|{g}"]["HOTA"], "oracle_false_gt5": agg["HOTA"], "oracle_full": full[f"oracle|{g}"]["HOTA"], "videos": len(gv)}
    for k, v in rep["counts"].items():
        print(f"[COUNT] {k}: жизней {v['lives']}, ложных рамок {v['fp_boxes']}", flush=True)
    for g, o in rep["oracle"].items():
        print(f"[ORACLE] {g} ({o['videos']}): без фильтра {o['none']:.3f}; удалены только ложные жизни с n > 5: {o['oracle_false_gt5']:.3f}; полный оракул {o['oracle_full']:.3f}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
