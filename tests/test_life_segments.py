"""Тесты сопоставления рамок с записями жизней (scripts/life_segments.py) и признаков внутри интервала.

Ошибка 09.10: один номер трекера бывает у нескольких жизней; рамки относились к жизни по номеру, а не по записи.
Синтетический случай: номер 8 — две жизни (кадры 0–5 и 10–15); рамки кадров 2 и 12 должны попасть в разные записи.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.life_segments import box_segments  # noqa: E402
from scripts.stitch_realness import life_features  # noqa: E402


def _make(tmp: Path):
    vid = "v"
    sig = tmp / "sig"; (sig / vid).mkdir(parents=True)
    root = tmp / "root"; (root / vid).mkdir(parents=True)
    rows = [[f, 10, 10, 50, 50, 8, 0.9 if f < 10 else 0.3, 2] for f in (0, 2, 5, 10, 12, 15)] + [[2, 100, 100, 150, 150, 9, 0.8, 2]]
    n = 3
    np.savez(sig / vid / "signatures.npz", track_id=np.array([8, 8, 9]), start=np.array([0, 10, 2]), end=np.array([5, 15, 2]), cls=np.array([2, 2, 2]),
             first_box=np.zeros((n, 4), np.float32) + [10, 10, 50, 50], last_box=np.zeros((n, 4), np.float32) + [10, 10, 50, 50],
             n_first=np.full(n, 3), n_last=np.full(n, 3), first_fly=np.zeros((n, 34, 19)), last_fly=np.zeros((n, 34, 19)),
             first_hexin=np.zeros((n, 721)), last_hexin=np.zeros((n, 721)), first_thumb=np.zeros((n, 256)), last_thumb=np.zeros((n, 256)),
             first_hist=np.zeros((n, 40)), last_hist=np.zeros((n, 40)), outputs=np.asarray(rows, np.float32), frame_count=np.asarray(16))
    (root / vid / "alignment.json").write_text(json.dumps({"matches": [{"video_frame_index": 2}, {"video_frame_index": 12}, {"video_frame_index": 7}]}), encoding="utf-8")
    box = {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
    res = [{"labels": [{"id": "8", "box2d": box}, {"id": "9", "box2d": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}}]},
           {"labels": [{"id": "8", "box2d": box}]},
           {"labels": [{"id": "8", "box2d": box}]}]   # кадр 7: строки трекера нет
    return vid, str(sig), str(root), res


def test_same_track_id_two_lives(tmp_path):
    vid, sig, root, res = _make(tmp_path)
    seg, st = box_segments(vid, sig, root, res)
    assert seg[0] == [0, 2]          # кадр 2: номер 8 → запись 0 (кадры 0–5), номер 9 → запись 2
    assert seg[1] == [1]             # кадр 12: номер 8 → запись 1 (кадры 10–15), не запись 0
    assert seg[2] == [None]          # кадр 7: строки трекера нет
    assert st["ok"] == 3 and st["no_row"] == 1


def test_life_features_by_interval(tmp_path):
    vid, sig, root, res = _make(tmp_path)
    old = life_features(str(Path(sig) / vid / "signatures.npz"))
    new = life_features(str(Path(sig) / vid / "signatures.npz"), by_interval=True)
    # число строк (log1p) и средняя уверенность: прежнее поведение смешивает обе жизни номера 8, исправленное — нет
    assert np.isclose(old["common"][0, 1], np.log1p(6)) and np.isclose(old["common"][1, 1], np.log1p(6))
    assert np.isclose(new["common"][0, 1], np.log1p(3)) and np.isclose(new["common"][1, 1], np.log1p(3))
    assert np.isclose(new["common"][0, 2], 0.9) and np.isclose(new["common"][1, 2], 0.3)
