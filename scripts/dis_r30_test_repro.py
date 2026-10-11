"""Сверка для dis_r30_test_v1: r30_all_none (motion_scheme_search) совпадает со стендом без сшивки (class_gate b30) на test 35.

Стенд без сшивки — scripts/class_gate.video с вариантом b30 (XYWHByteTrack, буфер 30, без запрета), вывод в
runs/dis_r30_test/stand_b30; HOTA по видео сравнивается с runs/motion_scheme_search/test.json → r30_all_none.

Запуск: python scripts/dis_r30_test_repro.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.class_gate as G  # noqa: E402

G.OUT = PROJECT_ROOT / "runs/dis_r30_test"


def task(t):
    import scripts.class_gate as g
    g.OUT = PROJECT_ROOT / "runs/dis_r30_test"
    return g.video(t)


def main() -> int:
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]["test"]
    root = str(PROJECT_ROOT / "runs/train_cache")
    with ProcessPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(task, [(v, root, "train", "b30") for v in sel]))
    ms = json.loads((PROJECT_ROOT / "runs/motion_scheme_search/test.json").read_text(encoding="utf-8"))["r30_all_none"]
    d = np.array([m["HOTA"] - ms[v]["HOTA"] for v, _, m in got])
    rep = {"videos": len(d), "max_abs_diff": float(np.abs(d).max()), "mean_stand": float(np.mean([m["HOTA"] for _, _, m in got])), "mean_scheme_none": float(np.mean([ms[v]["HOTA"] for v, _, _ in got]))}
    (G.OUT / "repro.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
