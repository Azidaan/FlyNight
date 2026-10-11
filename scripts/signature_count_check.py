"""Сохранённая проверка счётчика измерений глаза для всех наборов с метками жизней (поправка 11.10 к записи о А2).

Утверждение «n_first = n_last = min(5, end − start); совпадение двух средних — только при ≤ 5 измерениях» было
проверено разовой командой без сохранённого результата. Здесь — для train 130, dev 35, test 35, val7
(runs/life_filter_head_v2/lives.pkl) и 97 новых val (runs/newval_eval/lives.pkl), все записи signatures.npz:
число жизней, нарушения n_first / n_last, жизни с совпадающими средними (n_first = n_last и равные средние мухи,
допуск 1e-4) и из них с > 5 измерениями. Выход — runs/signature_overlap_reconcile/count_check.json.

Запуск: python scripts/signature_count_check.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "runs/signature_overlap_reconcile/count_check.json"


def main() -> int:
    L = pickle.load(open(PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl", "rb"))
    sets = {k: L[k] for k in ("train", "dev", "test", "val7")}
    sets["newval"] = pickle.load(open(PROJECT_ROOT / "runs/newval_eval/lives.pkl", "rb"))
    rep = {}
    for name, vids in sets.items():
        r = {"videos": len(vids), "lives": 0, "bad_n_first": 0, "bad_n_last": 0, "same": 0, "same_gt5": 0, "le5": 0}
        for c in vids:
            z = np.load(Path(c["sig_dir"]) / c["video_id"] / "signatures.npz")
            n = z["end"] - z["start"]; k = len(n)
            s = (z["n_first"] == z["n_last"]) & np.all(np.isclose(z["first_fly"].reshape(k, -1).astype(np.float32), z["last_fly"].reshape(k, -1).astype(np.float32), atol=1e-4), 1)
            r["lives"] += k; r["bad_n_first"] += int((z["n_first"] != np.minimum(5, n)).sum()); r["bad_n_last"] += int((z["n_last"] != np.minimum(5, n)).sum())
            r["same"] += int(s.sum()); r["same_gt5"] += int((s & (n > 5)).sum()); r["le5"] += int((n <= 5).sum())
        rep[name] = r
        print(name, r, flush=True)
    rep["total_lives"] = sum(v["lives"] for k, v in rep.items() if isinstance(v, dict))
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}, всего жизней {rep['total_lives']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
