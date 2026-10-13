"""Atomic, resumable Z13 training state and cooperative stop control."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path


class TrainingPaused(RuntimeError):
    pass


def replace_with_retry(source, destination, attempts=40, delay=0.25):
    """Windows readers may temporarily deny replacement; preserve old file until success."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def provenance(root, data_root, videos, config):
    paths = [root / p for p in (
        "scripts/z13_train.py", "tracking/z13_readout.py", "tracking/z13_checkpoint.py",
        "scripts/z13_data.py", "tracking/foveated_eyes.py", "tracking/flyvis_motion.py",
        "tracking/retina.py", "runs/flyvis/results/flow/0000/000/best_chkpt",
    )] + [data_root / (v + ".npz") for v in sorted(videos)]
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path.relative_to(root))] = digest.hexdigest()
    return {"files_sha256": hashes, "videos": sorted(videos), "config": config}


class RunControl:
    def __init__(self, directory, stop_file=None, deadline=None, resume=False, every=10):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop_file = Path(stop_file) if stop_file else None
        self.deadline = datetime.fromisoformat(deadline) if deadline else None
        if self.deadline is not None and self.deadline.tzinfo is None:
            raise ValueError("Deadline must include an explicit UTC offset")
        self.resume = resume
        self.every = every
        self.current = {}

    def stopped(self):
        return bool((self.stop_file and self.stop_file.exists()) or
                    (self.deadline and datetime.now().astimezone() >= self.deadline))

    def status(self, **values):
        self.current.update(values)
        atomic_json(self.directory / "status.json", {
            **self.current, "updated": datetime.now().astimezone().isoformat(),
        })

    def check(self):
        if self.stopped():
            self.status(state="paused", reason="stop requested or deadline reached")
            raise TrainingPaused("Paused; resume from the saved training checkpoint")


def save_checkpoint(torch, path, arm, bb, ro, opt, rng, step, hist, identity):
    cuda = torch.cuda.is_initialized()
    state = {
        "format": 1, "arm": arm, "step": step, "history": hist,
        "identity": identity, "readout": ro.state_dict(),
        "backbone": (bb if arm == "B" else bb.net).state_dict(),
        "anchors": [p.detach().cpu().clone() for p in getattr(bb, "p0", [])],
        "optimizer": opt.state_dict(), "generator": rng.bit_generator.state,
        "rng_python": random.getstate(), "rng_numpy": __import__("numpy").random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if cuda else None,
        "torch_version": str(torch.__version__),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Validate the file before replacing the previous usable checkpoint.
    verified = torch.load(temporary, map_location="cpu", weights_only=False)
    if verified["step"] != step or verified["identity"] != identity:
        raise RuntimeError("Checkpoint verification failed")
    replace_with_retry(temporary, path)


def restore_checkpoint(torch, path, arm, bb, ro, opt, rng, identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != 1 or state["arm"] != arm or state["identity"] != identity:
        raise ValueError("Resume rejected: code, data, split, arm or settings changed")
    if state["torch_version"] != str(torch.__version__):
        raise ValueError("Resume rejected: torch version changed")
    ro.load_state_dict(state["readout"])
    (bb if arm == "B" else bb.net).load_state_dict(state["backbone"])
    anchors = getattr(bb, "p0", [])
    if len(anchors) != len(state["anchors"]):
        raise ValueError("Resume rejected: anchor configuration changed")
    for destination, value in zip(anchors, state["anchors"]):
        destination.copy_(value.to(destination.device))
    opt.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["generator"]
    random.setstate(state["rng_python"])
    __import__("numpy").random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    if state["rng_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state["step"], state["history"]
