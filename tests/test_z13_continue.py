"""Тесты z13_core_continue_v1 без FlyVis: перенос Adam, равные пакеты, заморозка ядра, точное продолжение, одинаковые знаменатели."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.z13_continue import coverage_table, migrate_optimizer, paired, per_video_useful, train_steps  # noqa: E402


class Core:
    """Заменитель FlyBackbone: params, penalty, clamp."""

    def __init__(self):
        self.lin = torch.nn.Linear(5, 5)
        self.params = list(self.lin.parameters()); self.p0 = [p.detach().clone() for p in self.params]

    def penalty(self):
        return sum(((p - q) ** 2).mean() for p, q in zip(self.params, self.p0))

    def clamp(self):
        with torch.no_grad():
            self.lin.weight.clamp_(-5, 5)


class Data:
    def __init__(self, n=40, seed=0):
        g = np.random.default_rng(seed)
        self.x = torch.tensor(g.normal(size=(n, 5)), dtype=torch.float32)
        self.labels = [(i, 0, 1 if i % 3 else 0, int(g.integers(0, 721))) for i in range(n)]

    def batch(self, torch_, dev, eps, pos):
        return self.x[eps], None


def forward(torch_, bb, ro, hx, masks):
    return ro(torch.tanh(bb.lin(hx))).unsqueeze(1)   # (B, кадры поиска = 1, 722), как run_sequence


def setup(seed=0):
    torch.manual_seed(seed)
    bb, ro = Core(), torch.nn.Linear(5, 722)
    opt = torch.optim.Adam([{"params": list(ro.parameters()), "lr": 3e-3}, {"params": bb.params, "lr": 1e-3}])
    return bb, ro, opt


def warm(bb, ro, opt, data, steps=3):
    rng = np.random.default_rng(0); hist = []
    train_steps(torch, "D-train", data, bb, ro, "cpu", opt, rng, 0, steps, hist, forward, lambda s: None, batch=8, micro=4)
    return rng


def test_migrate_optimizer_keeps_readout_moments_and_drops_core():
    data = Data(); bb, ro, opt = setup(); warm(bb, ro, opt, data)
    sd = opt.state_dict(); m = migrate_optimizer(sd, len(list(ro.parameters())))
    assert len(m["param_groups"]) == 1 and set(m["state"]) == {0, 1}
    for i in (0, 1):
        assert torch.equal(m["state"][i]["exp_avg"], sd["state"][i]["exp_avg"]) and torch.equal(m["state"][i]["step"], sd["state"][i]["step"])
    with pytest.raises(ValueError):
        migrate_optimizer(sd, 3)


def test_frozen_core_unchanged_trained_core_changes_identical_batches():
    data = Data(); bb, ro, opt = setup(); rng0 = warm(bb, ro, opt, data)
    base = (copy.deepcopy(bb.lin.state_dict()), copy.deepcopy(ro.state_dict()), opt.state_dict(), copy.deepcopy(rng0.bit_generator.state))
    runs = {}
    for variant in ("D-frozen", "D-train"):
        b2, r2 = Core(), torch.nn.Linear(5, 722)
        b2.lin.load_state_dict(base[0]); r2.load_state_dict(base[1])
        if variant == "D-train":
            o2 = torch.optim.Adam([{"params": list(r2.parameters()), "lr": 3e-3}, {"params": b2.params, "lr": 1e-3}]); o2.load_state_dict(base[2])
        else:
            for p in b2.params:
                p.requires_grad_(False)
            o2 = torch.optim.Adam([{"params": list(r2.parameters()), "lr": 3e-3}]); o2.load_state_dict(migrate_optimizer(base[2], 2))
        rng = np.random.default_rng(0); rng.bit_generator.state = copy.deepcopy(base[3]); hist = []
        train_steps(torch, variant, data, b2, r2, "cpu", o2, rng, 3, 8, hist, forward, lambda s: None, batch=8, micro=4)
        runs[variant] = (hist, b2)
    assert [h["batch"] for h in runs["D-frozen"][0]] == [h["batch"] for h in runs["D-train"][0]]
    assert torch.equal(runs["D-frozen"][1].lin.weight, base[0]["weight"])
    assert not torch.equal(runs["D-train"][1].lin.weight, base[0]["weight"])
    assert all(h["core_object_gradient"] > 0 for h in runs["D-train"][0]) and all(h["core_object_gradient"] == 0 for h in runs["D-frozen"][0])


def test_frozen_rejects_core_gradients():
    data = Data(); bb, ro, opt = setup()
    with pytest.raises(RuntimeError, match="Frozen core"):   # ядро не заморожено, а вариант объявлен замороженным
        train_steps(torch, "D-frozen", data, bb, ro, "cpu", opt, np.random.default_rng(0), 0, 1, [], forward, lambda s: None, batch=8, micro=4)


def test_resume_equals_uninterrupted():
    data = Data()
    bb, ro, opt = setup(); rng = np.random.default_rng(0)
    train_steps(torch, "D-train", data, bb, ro, "cpu", opt, rng, 0, 6, [], forward, lambda s: None, batch=8, micro=4)
    full = (copy.deepcopy(ro.state_dict()), copy.deepcopy(bb.lin.state_dict()))
    bb2, ro2, opt2 = setup(); rng2 = np.random.default_rng(0)
    train_steps(torch, "D-train", data, bb2, ro2, "cpu", opt2, rng2, 0, 3, [], forward, lambda s: None, batch=8, micro=4)
    snap = (copy.deepcopy(ro2.state_dict()), copy.deepcopy(bb2.lin.state_dict()), copy.deepcopy(opt2.state_dict()), copy.deepcopy(rng2.bit_generator.state))
    bb3, ro3, opt3 = setup(seed=5)   # другое начальное — должно быть перезаписано восстановлением
    ro3.load_state_dict(snap[0]); bb3.lin.load_state_dict(snap[1]); opt3.load_state_dict(snap[2])
    rng3 = np.random.default_rng(123); rng3.bit_generator.state = snap[3]
    bb3.p0 = [p.detach().clone() for p in bb2.p0]
    train_steps(torch, "D-train", data, bb3, ro3, "cpu", opt3, rng3, 3, 6, [], forward, lambda s: None, batch=8, micro=4)
    assert all(torch.allclose(full[0][k], ro3.state_dict()[k], atol=1e-6) for k in full[0])
    assert all(torch.allclose(full[1][k], bb3.lin.state_dict()[k], atol=1e-6) for k in full[1])


def test_coverage_same_denominator_and_unknown_separate():
    rows = [{"kind": 1, "p_present": .9, "useful": True}, {"kind": -1, "p_present": .8}, {"kind": 0, "p_present": .7, "useful": False}] + \
           [{"kind": 1, "p_present": .1, "useful": False} for _ in range(97)]
    t = coverage_table(rows, (0.02, 0.03))
    assert t["0.02"] == {"selected": 2, "known": 1, "unknown": 1, "useful": 1, "harmful": 0, "useful_share_known": 1.0}
    assert t["0.03"]["harmful"] == 1 and t["0.03"]["unknown"] == 1


def test_paired_by_video():
    a = per_video_useful([{"kind": 1, "video": "v1", "useful": True}, {"kind": 1, "video": "v2", "useful": False}, {"kind": 0, "video": "v1", "useful": False}])
    b = per_video_useful([{"kind": 1, "video": "v1", "useful": False}, {"kind": 1, "video": "v2", "useful": False}])
    assert a == {"v1": 1.0, "v2": 0.0}
    r = paired(a, b, n_boot=200)
    assert r["mean_per_video"] == pytest.approx(0.5) and r["n"] == 2 and r["better"] == 1
