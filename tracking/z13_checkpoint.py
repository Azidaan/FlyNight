"""Atomic, resumable Z13 training state and cooperative stop control."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import math
from collections.abc import Mapping
from copy import copy, deepcopy
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


def _finite_values(torch, value):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Resume rejected: nonfinite optimizer value')
    elif isinstance(value, Mapping):
        for item in value.values():
            _finite_values(torch, item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_values(torch, item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Resume rejected: nonfinite optimizer value')


def _validate_weights(torch, module, state, label):
    from tracking.flyvis_checkpoint import validate_state
    validate_state(module, state, label)
    for key, target in module.state_dict().items():
        if not target.is_floating_point() and state[key].dtype != target.dtype:
            raise ValueError(f'Resume rejected: {label}.{key} non-floating dtype mismatch')


def _capture_rng(torch, rng):
    import numpy as np
    return (deepcopy(rng.bit_generator.state), random.getstate(), np.random.get_state(),
            torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng(torch, rng, saved):
    import numpy as np
    rng.bit_generator.state = saved[0]
    random.setstate(saved[1])
    np.random.set_state(saved[2])
    torch.set_rng_state(saved[3])
    if saved[4] is not None:
        torch.cuda.set_rng_state_all(saved[4])


def restore_checkpoint(torch, path, arm, bb, ro, opt, rng, identity, *, restore_cuda_rng=True):
    """Restore a complete training transaction or leave the live state unchanged.

    Exact GPU continuation needs already initialized CUDA generators. An explicit
    ``restore_cuda_rng=False`` permits CPU transfer of a GPU checkpoint, without
    claiming exact GPU continuation or queuing RNG changes for unrelated GPU work.
    """
    import numpy as np

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("format") != 1 or state.get("arm") != arm or state.get("identity") != identity:
        raise ValueError("Resume rejected: code, data, split, arm or settings changed")
    if state.get("torch_version") != str(torch.__version__):
        raise ValueError("Resume rejected: torch version changed")
    required = ('readout', 'backbone', 'anchors', 'optimizer', 'generator',
                'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda', 'step', 'history')
    if any(key not in state for key in required):
        raise ValueError('Resume rejected: incomplete checkpoint')
    if isinstance(state['step'], bool) or not isinstance(state['step'], int) or state['step'] < 0:
        raise ValueError('Resume rejected: invalid training step')
    if not isinstance(state['history'], list):
        raise ValueError('Resume rejected: training history must be a list')
    core = bb if arm == 'B' else bb.net
    _validate_weights(torch, ro, state['readout'], 'readout')
    _validate_weights(torch, core, state['backbone'], 'backbone')
    anchors = getattr(bb, "p0", [])
    if not isinstance(state['anchors'], (list, tuple)) or len(anchors) != len(state["anchors"]):
        raise ValueError("Resume rejected: anchor configuration changed")
    for destination, value in zip(anchors, state["anchors"]):
        if not torch.is_tensor(value) or value.shape != destination.shape or value.is_complex() != destination.is_complex():
            raise ValueError('Resume rejected: incompatible anchor tensor')
        if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(value.to(device='cpu', dtype=destination.dtype)).all()):
            raise ValueError('Resume rejected: nonfinite or overflowing anchor')

    # Stage the optimizer: load_state_dict accepts malformed moment shapes and
    # can cast finite float64 moments to Inf in a float32 optimizer.
    _finite_values(torch, state['optimizer'])
    old_rng = _capture_rng(torch, rng)
    try:
        staged = deepcopy(opt)
        staged.load_state_dict(deepcopy(state['optimizer']))
    finally:
        # A user optimizer hook can draw random values even on the private copy.
        # Validation is not a training step and must not consume any RNG stream.
        _restore_rng(torch, rng, old_rng)
    _finite_values(torch, staged.state_dict())
    for parameter, values in staged.state.items():
        for key, value in values.items():
            if torch.is_tensor(value) and key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq', 'momentum_buffer'):
                if value.shape != parameter.shape:
                    raise ValueError(f'Resume rejected: optimizer moment shape {key}')
                if key in ('exp_avg_sq', 'max_exp_avg_sq') and bool((value < 0).any()):
                    raise ValueError(f'Resume rejected: negative optimizer second moment {key}')

    # Validate RNG payloads with private generators, not the global live ones.
    random.Random().setstate(state['rng_python'])
    np.random.RandomState().set_state(state['rng_numpy'])
    torch.Generator(device='cpu').set_state(state['rng_torch'])
    private_generator = deepcopy(rng.bit_generator)
    private_generator.state = deepcopy(state['generator'])
    cuda_states = state['rng_cuda'] if restore_cuda_rng else None
    if cuda_states is not None:
        if not torch.cuda.is_initialized():
            raise ValueError('Resume rejected: CUDA RNG requires initialized CUDA; use restore_cuda_rng=False for explicit CPU transfer, not exact GPU continuation')
        if not isinstance(cuda_states, (list, tuple)) or len(cuda_states) != torch.cuda.device_count():
            raise ValueError('Resume rejected: CUDA RNG device count changed')
        for index, value in enumerate(cuda_states):
            torch.Generator(device=f'cuda:{index}').set_state(value)

    old_weights = [(module, {key: value.detach().clone() for key, value in module.state_dict().items()})
                   for module in (core, ro)]
    old_anchors = [value.detach().clone() for value in anchors]
    old_optimizer_state = copy(opt.state)
    old_optimizer_state.update({p: deepcopy(value) for p, value in opt.state.items()})
    old_optimizer_groups = [{key: list(value) if key == 'params' else deepcopy(value)
                             for key, value in group.items()} for group in opt.param_groups]
    try:
        core.load_state_dict(state['backbone'], strict=True)
        ro.load_state_dict(state['readout'], strict=True)
        with torch.no_grad():
            for destination, value in zip(anchors, state['anchors']):
                destination.copy_(value)
        opt.load_state_dict(deepcopy(state['optimizer']))
        rng.bit_generator.state = deepcopy(state['generator'])
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    except BaseException:
        # Do not invoke load_state_dict again: the hook that failed could fail
        # again. Tensor copies and original Parameter keys restore the live state.
        with torch.no_grad():
            for module, saved in old_weights:
                for key, target in module.state_dict().items():
                    target.copy_(saved[key])
            for target, saved in zip(anchors, old_anchors):
                target.copy_(saved)
        opt.state, opt.param_groups = old_optimizer_state, old_optimizer_groups
        _restore_rng(torch, rng, old_rng)
        raise
    return state["step"], state["history"]
