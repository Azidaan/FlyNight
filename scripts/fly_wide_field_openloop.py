"""Уточнение вывода fly_wide_field_v1 (одна проверка без GPU): ошибка от несовпадения обучения и применения?

Веса считывания подбирались по признакам из рамок, интерполированных между размеченными кадрами, а при оценке признаки
берутся из последовательно предсказанных рамок. На тех же 6 видео train (кэш runs/fly_wide_field/cache) сравниваются:
  closed — как в оценке: признаки из предсказанной на прошлом шаге рамки;
  open   — признаки из истинной (интерполированной) рамки на каждом шаге, смещение накапливается так же.
Медиана ошибки центра в конце интервала (px), по видео; для t45_pool и dec_pool, справочно — ноль.
Выход — runs/fly_wide_field/openloop.json.

Запуск: python scripts/fly_wide_field_openloop.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_wide_field as W  # noqa: E402


def errs(pts, F, m, vid):
    closed, opened, zero = [], [], []
    for r0, r1, b0, b1 in W.pairs(vid, len(F)):
        gt = (b1[:2] + b1[2:]) / 2; c0 = (b0[:2] + b0[2:]) / 2
        pr = b0.astype(np.float64).copy(); co = c0.astype(np.float64).copy()
        for r in range(r0 + 1, r1 + 1):
            f = F[r].astype(np.float32)
            v = W.predict(m, W.pool(pts, f, pr)); pr = pr + np.concatenate([v, v])
            w = (r - 1 - r0) / (r1 - r0); co = co + W.predict(m, W.pool(pts, f, (1 - w) * b0 + w * b1))
        closed.append(float(np.linalg.norm((pr[:2] + pr[2:]) / 2 - gt))); opened.append(float(np.linalg.norm(co - gt))); zero.append(float(np.linalg.norm(c0 - gt)))
    return float(np.median(closed)), float(np.median(opened)), float(np.median(zero))


def main() -> int:
    tr, ev, cond = W.split_videos()
    data = {}
    for v in tr:
        z = np.load(W.OUT / "cache" / f"{v}.npz"); data[v] = (z["points"], z["dec"], z["t45"])
    rep = {}
    for name, k in (("dec_pool", 1), ("t45_pool", 2)):
        X, Y = [], []
        for v in tr:
            x, y = W.train_rows(data[v][0], data[v][k], v); X += x; Y += y
        m = W.fit(X, Y)
        per = {v: dict(zip(("closed", "open", "zero"), errs(data[v][0], data[v][k], m, v))) for v in tr}
        rep[name] = {"per_video": per, **{s: float(np.mean([per[v][s] for v in tr])) for s in ("closed", "open", "zero")}}
        print(f"{name}: closed {rep[name]['closed']:.2f}, open {rep[name]['open']:.2f}, ноль {rep[name]['zero']:.2f} (среднее по 6 видео train)", flush=True)
        for v in tr:
            print(f"   {v}: closed {per[v]['closed']:.2f} open {per[v]['open']:.2f} ноль {per[v]['zero']:.2f}", flush=True)
    (W.OUT / "openloop.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
