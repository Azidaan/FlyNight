"""Сверка доли жизней с совпадающими средними мухи на одинаковом определении (поправка к пересмотру 7.10, А2).

Противоречие 11.10: «84 % на 40 видео train» (все записи, np.isclose по умолчанию, без условия n_first = n_last) против
«31 % размеченных на 97 новых val» (newval_signature_overlap_v1). Здесь — одно определение для всех наборов:
совпадают = n_first == n_last и first_fly ≈ last_fly (atol 1e-4), как в scripts/newval_signature_overlap.py.
Метки жизней — runs/life_filter_head_v2/lives.pkl (train 130, dev 35, test 35, val7) и runs/newval_eval/lives.pkl
(97 новых val): 1 — полностью ложная, 0 — настоящая / смешанная, < 0 — без метки (нет рамок на размеченных кадрах).
Для групп «все записи / размеченные / ложные / настоящие» — число, доля совпадающих, медиана длительности (кадры
видео, end − start + 1), доля жизней с n_first < 5 (меньше пяти измерений глаза), доля ложных среди размеченных.
Только описание сохранённых данных. Выход — runs/signature_overlap_reconcile/report.json.

Запуск: python scripts/signature_overlap_reconcile.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "runs/signature_overlap_reconcile"


def same_mask(z):
    n = len(z["n_first"])
    return (z["n_first"] == z["n_last"]) & np.all(np.isclose(z["first_fly"].reshape(n, -1).astype(np.float32), z["last_fly"].reshape(n, -1).astype(np.float32), atol=1e-4), 1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl", "rb"))
    sets = {k: L[k] for k in ("train", "dev", "test", "val7")}
    sets["newval"] = pickle.load(open(PROJECT_ROOT / "runs/newval_eval/lives.pkl", "rb"))
    rep = {}
    for name, vids in sets.items():
        acc = {g: {"n": 0, "same": 0, "dur": [], "few": 0} for g in ("all", "labeled", "false", "true_or_mixed")}
        mism = 0
        for c in vids:
            z = np.load(Path(c["sig_dir"]) / c["video_id"] / "signatures.npz")
            if len(z["n_first"]) != len(c["label"]):
                mism += 1; continue
            s = same_mask(z); dur = z["end"] - z["start"] + 1; few = z["n_first"] < 5; lab = c["label"]
            for g, m in (("all", np.ones(len(lab), bool)), ("labeled", lab >= 0), ("false", lab == 1), ("true_or_mixed", lab == 0)):
                a = acc[g]; a["n"] += int(m.sum()); a["same"] += int((m & s).sum()); a["dur"].append(dur[m]); a["few"] += int((m & few).sum())
        r = {"videos": len(vids), "length_mismatch_videos": mism}
        for g, a in acc.items():
            d = np.concatenate(a["dur"]) if a["dur"] else np.zeros(0)
            r[g] = {"n": a["n"], "same_share": round(a["same"] / max(a["n"], 1), 3), "median_frames": float(np.median(d)) if len(d) else None,
                    "share_le5_frames": round(float((d <= 5).mean()), 3) if len(d) else None, "share_lt5_measurements": round(a["few"] / max(a["n"], 1), 3)}
        r["false_share_of_labeled"] = round(acc["false"]["n"] / max(acc["labeled"]["n"], 1), 3)
        rep[name] = r
        print(f"{name} ({len(vids)} видео, несовпадений длины {mism}):")
        for g in ("all", "labeled", "false", "true_or_mixed"):
            x = r[g]; print(f"   {g}: {x['n']} жизней, совпадающих {x['same_share']:.1%}, медиана {x['median_frames']} кадров, ≤ 5 кадров {x['share_le5_frames']:.1%}, < 5 измерений {x['share_lt5_measurements']:.1%}")
        print(f"   ложных среди размеченных {r['false_share_of_labeled']:.1%}")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
