"""Тесты З13 (fly_object_readout_z13_v1): метки по исходной разметке, геометрия решётки, оценка центра, правило τ, модули."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.z13_data import label_frame, raw_index  # noqa: E402
from scripts.z13_train import choose_tau  # noqa: E402
from tracking.z13_readout import HexGeometry, center_estimate, to_eye, to_frame  # noqa: E402


def hex_points(radius=4, spacing=10.0):
    pts = []
    for q in range(-radius, radius + 1):
        for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
            pts.append((spacing * (q + r / 2.0) + 50, spacing * (r * np.sqrt(3) / 2) + 50))
    return np.asarray(pts, np.float32)


@pytest.fixture(scope="module")
def geom():
    return HexGeometry(hex_points(), 100)


def test_neighbors_interior(geom):
    c = geom.nearest((50, 50))
    nb = geom.neighbors[c]
    assert nb[0] == c and len(set(nb.tolist())) == 7
    assert np.allclose(np.linalg.norm(geom.points[nb[1:]] - geom.points[c], axis=1), geom.spacing, rtol=0.05)


def test_eye_frame_roundtrip():
    crop = (100, 200, 300, 400)
    p = (150.0, 260.0)
    assert np.allclose(to_frame(to_eye(p, crop, 64), crop, 64), p)


def test_raw_index_no_interpolation_and_filters():
    truth = [{"labels": [{"id": "a", "category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
                         {"id": "c", "category": "car", "attributes": {"crowd": True}, "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
                         {"id": "s", "category": "traffic sign", "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}]},
             {"labels": []},
             {"labels": [{"id": "a", "category": "car", "box2d": {"x1": 20, "y1": 0, "x2": 30, "y2": 10}}]}]
    R = raw_index(truth, [0, 6, 12], ("car",))
    assert [o[0] for o in R[0]] == ["a"] and R[6] == [] and [o[0] for o in R[12]] == ["a"]
    assert set(R) == {0, 6, 12}


def test_label_kinds(geom):
    crop = (0, 0, 100, 100)
    objs = [("a", "car", np.array([40.0, 40.0, 60.0, 60.0]))]
    assert label_frame("a", 3, objs, crop, geom, 100)[0] == 1
    assert label_frame("a", 3, [], crop, geom, 100)[0] == 0                       # нет в разметке кадра — отказ
    assert label_frame("a", 3, [], crop, geom, 100, around=True)[0] == -1         # дыра в разметке — неизвестно
    assert label_frame(None, 3, objs, crop, geom, 100)[0] == -1                   # объект образца не установлен
    assert label_frame("a", 13, objs, crop, geom, 100)[0] == -1                   # установлен слишком давно
    far = [("a", "car", np.array([200.0, 200.0, 220.0, 220.0]))]
    assert label_frame("a", 3, far, crop, geom, 100)[0] == 0                      # вне области
    corner = [("a", "car", np.array([94.0, 94.0, 98.0, 98.0]))]
    assert label_frame("a", 3, corner, crop, geom, 100)[0] == 0                   # угол вырезки вне шестиугольника колонок


def test_center_estimate_does_not_average_two_peaks(geom):
    p = np.full(len(geom.points), 1e-4)
    a, b = geom.nearest((20, 50)), geom.nearest((80, 50))
    p[a], p[b] = 0.5, 0.4
    c = center_estimate(p, geom)
    assert np.linalg.norm(c - geom.points[a]) < geom.spacing


def _row(kind, p, ok=True):
    r = {"kind": kind, "p_present": p, "center": [0, 0], "pred": [1, 1]}
    if kind == 1:
        r.update(err=1.0 if ok else 50.0, err_pred=10.0, scale=20.0)
    return r


def test_choose_tau_none_when_too_few_interventions():
    rows = [_row(1, 0.9) for _ in range(40)]
    assert choose_tau(rows)["tau"] is None


def test_choose_tau_precision_rule():
    rows = [_row(1, 0.9) for _ in range(95)] + [_row(0, 0.5) for _ in range(30)] + [_row(-1, 0.95) for _ in range(10)]
    rep = choose_tau(rows)
    assert rep["tau"] is not None and rep["tau"] > 0.5 and rep["precision"] >= 0.9 and rep["on_negative"] == 0
    assert rep["unknown_intervention_share"] == 1.0


def test_modules_shapes(geom):
    torch = pytest.importorskip("torch")
    from tracking.z13_readout import make_modules
    Readout, HexGRU = make_modules(torch)
    n = len(geom.points)
    gru = HexGRU(geom.neighbors); ro = Readout(gru.out_dim)
    hx = torch.rand(3, 6, n)
    st = gru.init(hx[:, 0:1]); last = hx[:, 0]
    for j in range(1, 4):
        st, f = gru.step(st, last, hx[:, j]); last = hx[:, j]
    ex = ro.exemplar(f, torch.zeros(3, n, dtype=torch.bool).index_fill_(1, torch.tensor([0, 1]), True))
    lg = ro.logits(ex, f)
    assert f.shape == (3, 34, n) and lg.shape == (3, n + 1)
    lg.sum().backward()
    assert gru.out.w.grad is not None
