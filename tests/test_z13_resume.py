"""Regression checks: episode ownership and exact CPU continuation after pause."""
import random

import numpy as np
import pytest

from scripts import z13_train as training
from tracking.z13_checkpoint import RunControl, TrainingPaused, restore_checkpoint


def write_video(path, episodes, base):
    off = np.arange(episodes + 1) * 5
    values = np.repeat(np.arange(base, base + episodes), 5)
    np.savez_compressed(path, ep_offset=off, ep_pert=np.zeros(episodes, bool),
        fr_hex=np.repeat(values[:, None], 721, axis=1).astype(np.float16),
        ex_mask=np.ones((episodes, 721), bool), lab_ep=np.arange(episodes),
        lab_pos=np.zeros(episodes, int), lab_kind=np.ones(episodes, int),
        lab_col=np.arange(episodes), lab_center=np.zeros((episodes, 2)),
        lab_gtbox=np.ones((episodes, 4)), lab_pred=np.zeros((episodes, 2)),
        lab_crop=np.ones((episodes, 4)))


def test_labels_keep_video_episode_ownership(tmp_path, monkeypatch):
    write_video(tmp_path / "a.npz", 3, 0)
    write_video(tmp_path / "b.npz", 2, 3)
    write_video(tmp_path / "c.npz", 4, 5)
    monkeypatch.setattr(training, "DATA", tmp_path)
    data = training.Episodes(["a", "b", "c"])
    assert [row[0] for row in data.labels] == list(range(9))
    for row in data.labels:
        assert data.hex[data.off[row[0]], 0] == row[0]
    assert len(data.mask) == len(data.off) - 1 == 9


def test_bad_episode_label_rejected(tmp_path, monkeypatch):
    write_video(tmp_path / "a.npz", 2, 0)
    with np.load(tmp_path / "a.npz") as z:
        arrays = dict(z)
    arrays["lab_ep"][0] = 2
    np.savez_compressed(tmp_path / "a.npz", **arrays)
    monkeypatch.setattr(training, "DATA", tmp_path)
    with pytest.raises(ValueError, match="outside"):
        training.Episodes(["a"])


def test_training_resume_equals_uninterrupted(tmp_path):
    import torch
    from tracking.z13_readout import make_modules
    torch.set_num_threads(1)
    Readout, _ = make_modules(torch)

    class SmallRecurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 3)
            self.drop = torch.nn.Dropout(0.1)

        def init(self, hx):
            return torch.zeros(hx.shape[0], hx.shape[-1], 3)

        def step(self, state, last, current):
            # Exercise Python, global numpy and torch RNG restoration as well.
            scale = 0.9 + 0.01 * (random.random() + np.random.random())
            state = state * 0.4 + self.drop(self.linear(current.unsqueeze(-1))) * scale
            return state, state.transpose(1, 2)

    def setup(seed):
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        return SmallRecurrent(), Readout(3)

    data = training.Episodes.__new__(training.Episodes)
    data.hex = np.random.default_rng(21).random((100, 721)).astype(np.float16)
    data.mask = np.ones((20, 721), bool)
    data.off = np.arange(21) * 5
    data.labels = [(e, 0, 1, e, None, None, None, None, False) for e in range(20)]
    identity = {"data": "fixture", "code": "fixture", "iters": 8}
    bb0, ro0 = setup(7)
    uninterrupted = training.train("B", torch, data, bb0, ro0, "cpu", 8)

    bb1, ro1 = setup(7)
    ctl = RunControl(tmp_path / "pause", every=1)
    original_status = ctl.status

    def pause_after_three(**values):
        original_status(**values)
        if values.get("step") == 3:
            ctl.stop_file = tmp_path / "stop"
            ctl.stop_file.write_text("stop")

    ctl.status = pause_after_three
    path = tmp_path / "checkpoint.pt"
    with pytest.raises(TrainingPaused):
        training.train("B", torch, data, bb1, ro1, "cpu", 8,
                       control=ctl, checkpoint=path, identity=identity)
    ck = torch.load(path, weights_only=False)
    assert ck["step"] == 3 and ck["optimizer"]["state"]
    assert all(k in ck for k in ("generator", "rng_python", "rng_numpy", "rng_torch", "rng_cuda"))
    bb2, ro2 = setup(99)
    resumed = training.train("B", torch, data, bb2, ro2, "cpu", 8,
        control=RunControl(tmp_path / "resume", resume=True, every=1),
        checkpoint=path, identity=identity)
    assert resumed == uninterrupted
    for name, value in bb0.state_dict().items():
        assert torch.equal(value, bb2.state_dict()[name])
    for name, value in ro0.state_dict().items():
        assert torch.equal(value, ro2.state_dict()[name])
    opt = torch.optim.Adam([{"params": list(ro2.parameters()), "lr": training.LR_READOUT}]
        + training.backbone_groups("B", bb2))
    with pytest.raises(ValueError, match="changed"):
        restore_checkpoint(torch, path, "B", bb2, ro2, opt, np.random.default_rng(0), {"data": "other"})


def test_aware_deadline_and_stop_file(tmp_path):
    with pytest.raises(ValueError, match="UTC offset"):
        RunControl(tmp_path, deadline="2026-10-13T07:35:00")
    ctl = RunControl(tmp_path, deadline="2000-01-01T00:00:00+03:00")
    with pytest.raises(TrainingPaused):
        ctl.check()


def test_fly_anchor_and_optimizer_restoration(tmp_path):
    import torch
    from tracking.z13_checkpoint import save_checkpoint

    class FlyLike:
        def __init__(self):
            self.net = torch.nn.Linear(2, 2)
            self.p0 = [p.detach().clone() for p in self.net.parameters()]

    bb = FlyLike(); ro = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(list(ro.parameters()) + list(bb.net.parameters()))
    rng = np.random.default_rng(31)
    ((ro(torch.ones(2)) + bb.net(torch.ones(2))) ** 2).sum().backward()
    opt.step()
    expected = [v.detach().clone() for v in bb.net.parameters()]
    anchors = [v.clone() for v in bb.p0]
    path = tmp_path / "fly.pt"
    save_checkpoint(torch, path, "D", bb, ro, opt, rng, 1, [{"it": 1}], {"fixture": "D"})
    fresh = FlyLike(); ro2 = torch.nn.Linear(2, 2)
    opt2 = torch.optim.Adam(list(ro2.parameters()) + list(fresh.net.parameters()))
    restore_checkpoint(torch, path, "D", fresh, ro2, opt2, np.random.default_rng(0), {"fixture": "D"})
    assert all(torch.equal(a, b) for a, b in zip(expected, fresh.net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(anchors, fresh.p0))
    assert len(opt2.state) == len(opt.state)


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows process-tree guard")
def test_stop_guard_terminates_only_owned_child():
    import subprocess
    import sys
    from scripts.z13_window_runner import stop_owned_child
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    own = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=flags)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=flags)
    try:
        stop_owned_child(own)
        assert own.wait(timeout=5) is not None
        assert other.poll() is None
    finally:
        # This test owns both fixtures; production passes only its actual child.
        stop_owned_child(own)
        stop_owned_child(other)


def test_t3_acceptance_checks_frames_events_and_current_budget(tmp_path, monkeypatch):
    import json
    from scripts import z13_window_runner as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run = tmp_path / "run"
    (run / "t2").mkdir(parents=True)
    (run / "t2/report.json").write_text(json.dumps({a: {"seconds": 100} for a in ("B", "C", "D")}))
    for video in ("019efe88-bbc2a4ac", "023b022b-723860c9"):
        output = run / "t3/signatures" / video
        baseline = tmp_path / "runs/fly_signatures" / video
        output.mkdir(parents=True); baseline.mkdir(parents=True)
        np.savez_compressed(output / "signatures.npz", outputs=np.ones((2, 8)), frame_count=1200)
        np.savez_compressed(baseline / "signatures.npz", frame_count=1200)
        (output / "signatures.json").write_text(json.dumps({"z13_events": {
            "steps": 10, "births": 2, "interventions": 3, "refusals": 7}, "seconds": 100}))
    assert runner.accept_t3(run)["passed"]
    target = run / "t3/signatures/019efe88-bbc2a4ac/signatures.json"
    values = json.loads(target.read_text())
    values["z13_events"]["refusals"] = 8
    target.write_text(json.dumps(values))
    assert not runner.accept_t3(run)["passed"]
    values["z13_events"]["refusals"] = 7
    values["seconds"] = 1100
    target.write_text(json.dumps(values))
    assert not runner.accept_t3(run)["within_price_limit"]
