"""З13: проверка режима сборщика без моделирования мухи (--skip-fly) — воспроизводит ли он стенд для сшивки «база + MASA».

Сборка стенда на dev 35 с --skip-fly --cpu (runs/fly_signatures_skipfly_check) против штатных сигнатур runs/fly_signatures:
  * совпадение жизней и признаков, которые читает сшивка «база + MASA» (track_id, start, end, cls, рамки, n_first / n_last,
    миниатюры, гистограммы, вход глаза, выходы трекера);
  * та же сшивка (эмбеддинги DINO / MASA стенда — жизни те же) → HOTA = 61,655 (runs/chain_filter_rank/repro/dev).
Выход — runs/z13/skipfly_check.json.

Запуск: python scripts/z13_check_skipfly.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stitch_eval_v2 as V  # noqa: E402

SIG = PROJECT_ROOT / "runs/fly_signatures_skipfly_check"
REF = PROJECT_ROOT / "runs/fly_signatures"
OUT = PROJECT_ROOT / "runs/z13/skipfly_check"
KEYS = ("track_id", "start", "end", "cls", "first_box", "last_box", "n_first", "n_last", "first_thumb", "last_thumb", "first_hist", "last_hist", "first_hexin", "last_hexin", "outputs")
STAND_MASA_HOTA = 61.65477893761337   # runs/buffer_extend/report.json, pooled.b30_masa|all


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    S = sets(REF, PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]]; vids = [v for v, _, _ in dev]; split, root = dev[0][2], dev[0][1]
    mism = {}
    for v in vids:
        a, b = np.load(SIG / v / "signatures.npz"), np.load(REF / v / "signatures.npz")
        bad = [k for k in KEYS if a[k].shape != b[k].shape or not np.allclose(a[k].astype(np.float64), b[k].astype(np.float64), atol=1e-3)]
        if bad:
            mism[v] = bad
    with ProcessPoolExecutor(max_workers=4, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        tr = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["train"]]))
    idx = [V.NAMES.index(c) for c in V.HEADS["base_masa"]]
    X = np.concatenate([d["X"][d["y"] >= 0] for d in tr]); y = np.concatenate([d["y"][d["y"] >= 0] for d in tr]); del tr
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[:, idx], y); del X, y
    with ProcessPoolExecutor(max_workers=4, initializer=V._set_dirs, initargs=(150, 5.0, str(PROJECT_ROOT / "runs/dino_embeddings"), str(PROJECT_ROOT / "runs/masa_embeddings"))) as pool:
        pairs = list(pool.map(V.video_pairs, [(v, str(SIG), root, split) for v in vids]))
    tasks = [(d["video_id"], str(SIG), root, split, d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1] if len(d["y"]) else np.zeros(0), 0.6, str(OUT / "masa")) for d in pairs]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(stitch_video, tasks))
    h = evaluate_hota(OUT / "masa", vids, split=split)["HOTA"]
    rep = {"videos": len(vids), "mismatched_videos": mism, "hota_masa": h, "reference": STAND_MASA_HOTA, "passed": bool(not mism and abs(h - STAND_MASA_HOTA) < 1e-6)}
    (OUT.parent / "skipfly_check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[SKIPFLY] {rep}\n[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
