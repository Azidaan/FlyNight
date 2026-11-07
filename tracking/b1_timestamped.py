"""Б1/З13: версия с реальным временем наблюдений, отдельно от старых опытов.

У всех ветвей одинаковые наблюдаемые кадры. Пропуски заполняются линейной
интерполяцией только между уже полученными наблюдениями. FlyVis интегрирует
реальную длительность; дискретные контроли делают шаг на каждый исходный кадр.
Повтор кадра при дополнении короткой истории не продвигает состояние во времени.
Старые контрольные точки без явной версии времени здесь не загружаются.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from tracking.b1_learning import ReceptorHistory
from tracking.b1_real_inputs import corrected_identity_probability
from tracking.flyvis_temporal_residual import TemporalResidualAssociation
from tracking.z13_readout import FlyBackbone, make_modules

SCHEMA = "b1_observed_frame_time_v1"


@dataclass(frozen=True)
class TimingPolicy:
    max_dt: float = .02

    def __post_init__(self):
        if not math.isfinite(self.max_dt) or not 0 < self.max_dt <= .02:
            raise ValueError("max_dt must be finite and in (0, 0.02]")

    def metadata(self):
        return dict(schema=SCHEMA, max_dt=self.max_dt, missing_frames="linear",
                    repeated_frame="zero_elapsed", controls="one_step_per_source_frame",
                    source_clock="frame_delta / actual_source_fps")


def _frame(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("Frame numbers must be nonnegative integers")
    return int(value)


def _fps(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Actual source FPS must be finite and positive")
    return value


def _cursor(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Valid checkpoint cursor required")
    return value


def _finite_optimizer_values(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite optimizer state")
    elif isinstance(value, (np.generic, np.ndarray)) and np.issubdtype(value.dtype, np.number):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite optimizer NumPy value")
    elif isinstance(value, Mapping):
        for item in value.values():
            _finite_optimizer_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_optimizer_values(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite optimizer value")


def _concrete_device(value):
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type == "cpu":
        return torch.device("cpu")
    return device


def _seeded_module(factory, device, seed):
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.device(device):
        # torch.manual_seed also seeds every CUDA device. Seed only the generator
        # that construction uses; fork_rng restores that generator on exit.
        if device.type == "cuda":
            torch.cuda.default_generators[device.index].manual_seed(seed)
        else:
            torch.random.default_generator.manual_seed(seed)
        return factory().to(device)


class TimestampedBackbone:
    """Общий причинный путь времени для ядра и согласованных контролей."""
    def __init__(self, backbone, arm: str, policy=TimingPolicy()):
        if arm not in ("C", "D", "B", "R1"):
            raise ValueError("Expected C, D, B or R1")
        self.backend, self.arm, self.policy = backbone, arm, policy
        self.dev = backbone.dev

    def _features(self, state, current):
        if self.arm in ("C", "D"):
            return state.nodes.activity[:, self.backend.index]
        if self.arm == "B":
            return torch.tanh(self.backend.out(state["h"]))
        return torch.cat((state, current[:, None] - state), 1)

    def _candidate(self, state, current, activity=None):
        """Validate the whole recurrent state before publishing or reusing it.

        Backends in this module create new states. This boundary does not undo
        arbitrary in-place mutations by a replacement backend.
        """
        batch, columns = current.shape
        if self.arm in ('C', 'D'):
            values = state.nodes.activity
            state_shape = (batch, self.backend.net.stimulus.n_nodes)
            channels = self.backend.index.shape[0]
        elif self.arm == 'B':
            if not isinstance(state, Mapping) or 'h' not in state:
                raise ValueError('B recurrent state must contain h')
            values = state['h']
            state_shape = (batch, self.backend.out.w.shape[1], columns)
            channels = self.backend.out.w.shape[0]
        else:
            values = state
            state_shape = (batch, self.backend.alpha.shape[1], columns)
            channels = 2 * self.backend.alpha.shape[1]
        if (not torch.is_tensor(values) or tuple(values.shape) != tuple(state_shape)
                or values.device != current.device or not torch.isfinite(values).all()):
            raise ValueError(f'Expected finite recurrent state with shape {tuple(state_shape)} on the observation device')
        if activity is None:
            activity = self._features(state, current)
        shape = (batch, channels, columns)
        if (not torch.is_tensor(activity) or tuple(activity.shape) != tuple(shape)
                or activity.device != current.device or not torch.isfinite(activity).all()):
            raise ValueError(f'Expected finite recurrent activity with shape {shape} on the observation device')
        return state, activity

    def init(self, first):
        if (first.ndim != 3 or first.shape[1] != 1 or not first.shape[0] or not first.shape[2]
                or not torch.isfinite(first).all()):
            raise ValueError('Finite nonempty B,1,N initial observations required')
        state = self.backend.init(first)
        current = first.reshape(first.shape[0], -1)
        return self._candidate(state, current)

    def step(self, state, previous, current, *, previous_frame, current_frame, source_fps):
        before, after = _frame(previous_frame), _frame(current_frame)
        fps = _fps(source_fps)
        if previous.shape != current.shape or previous.ndim != 2:
            raise ValueError("Matching B,N receptor tensors required")
        if previous.device != current.device or previous.dtype != current.dtype:
            raise ValueError("Observation dtype/device must match")
        if not torch.isfinite(previous).all() or not torch.isfinite(current).all():
            raise ValueError("Finite receptor observations required")
        gap = after - before
        if gap < 0:
            raise ValueError("Observed frames must not go backwards")
        if not gap:
            if not torch.equal(previous, current):
                raise ValueError("Repeated frame ID must identify the same receptor input")
            return self._candidate(state, current)
        if self.arm in ("C", "D"):
            duration = gap / fps
            count = max(1, math.ceil(duration / self.policy.max_dt - 1e-12))
            dt = duration / count
            alpha = torch.arange(1, count + 1, device=current.device, dtype=current.dtype)
            alpha = alpha.view(1, count, 1, 1) / count
            stimulus = previous[:, None, None] * (1 - alpha) + current[:, None, None] * alpha
            from tracking.b1_index_copy_stimulus import indexed_stimulus
            net = self.backend.net
            inputs = indexed_stimulus(torch, stimulus, np.asarray(net.stimulus.input_index), net.stimulus.n_nodes)
            states = net.forward(inputs, dt, state, as_states=True)
            # Validate every internal state: a later bounded readout must not
            # conceal a nonfinite neuron outside the selected output types.
            for candidate in states[:-1]:
                self._candidate(candidate, current)
            return self._candidate(states[-1], current)
        last = previous
        for step in range(1, gap + 1):
            # Endpoint must be bit-identical for the adjacent-frame control.
            value = current if step == gap else previous * (1 - step / gap) + current * (step / gap)
            state, features = self.backend.step(state, last, value)
            state, features = self._candidate(state, value, features)
            last = value
        return state, features

    def sequence(self, hexes, frame_numbers, source_fps):
        if hexes.ndim != 3 or not hexes.shape[1] or len(frame_numbers) != hexes.shape[1]:
            raise ValueError("B,T,N input with one frame number per observation required")
        frames = [_frame(f) for f in frame_numbers]
        _fps(source_fps)
        if any(b < a for a, b in zip(frames, frames[1:])):
            raise ValueError("Observed frames must not go backwards")
        if not torch.isfinite(hexes).all():
            raise ValueError("Finite receptor observations required")
        state, activity = self.init(hexes[:, :1])
        activities = [activity]
        for i in range(1, len(frames)):
            state, activity = self.step(state, hexes[:, i-1], hexes[:, i],
                previous_frame=frames[i-1], current_frame=frames[i], source_fps=source_fps)
            activities.append(activity)
        return state, activities


def timestamped_pair_activities(backbone, hexes, frame_numbers, source_fps, reference_index):
    if not isinstance(reference_index, int) or not 0 <= reference_index < hexes.shape[1] - 1:
        raise ValueError("Reference observation must precede the current one")
    if _frame(frame_numbers[reference_index]) >= _frame(frame_numbers[-1]):
        raise ValueError("Reference time must precede current time")
    _, activity = backbone.sequence(hexes, frame_numbers, source_fps)
    return activity[reference_index], activity[-1]


def _checked_identity_probability(base, delta, real):
    if delta.shape != base.shape or not torch.isfinite(delta).all():
        raise ValueError('Expected finite residual head correction aligned with base probabilities')
    updated = corrected_identity_probability(base, delta.double(), real)
    # NULL and padded slots intentionally retain their original NaN. Only real
    # candidate probabilities must be finite for association and training loss.
    if not torch.isfinite(updated[real]).all():
        raise ValueError('Residual head produced nonfinite real-candidate probabilities')
    return updated


def forward_timestamped_residual(backbone, head, hexes, row, inputs, *, frame_numbers, source_fps, train_backbone):
    """Обучающий вызов с теми же масками/базой, но с явными временами."""
    with torch.set_grad_enabled(torch.is_grad_enabled() and train_backbone):
        reference, current = timestamped_pair_activities(backbone, hexes, frame_numbers,
                                                        source_fps, row["reference_index"])
    base = inputs["base"]
    logits = torch.logit(torch.nan_to_num(base, nan=.5).clamp(1e-8, 1-1e-8)).float()
    _, delta = head(logits, reference, current, inputs["reference_mask"], inputs["masks"],
                    inputs["valid"], inputs["has_history"])
    return _checked_identity_probability(base, delta, inputs["real"]), delta


def load_timestamped_observation(root, row, device):
    """Читает времена из того же архива, что изображения, и проверяет привязку."""
    root = Path(root).resolve()
    path = (root / row["retinal_source"]).resolve()
    path.relative_to(root)
    with np.load(path) as data:
        start, current = int(row["prefix_start"]), int(row["current_array_index"])
        if not 0 <= start <= current < len(data["fr_frame"]):
            raise ValueError("Invalid observation slice")
        values = data["fr_hex"][start:current+1].astype(np.float32)
        frames = data["fr_frame"][start:current+1].copy()
    reference = row["reference_index"]
    if not 0 <= reference < len(frames)-1:
        raise ValueError("Invalid reference observation")
    if int(frames[reference]) != row["reference_frame"] or int(frames[-1]) != row["current_frame"]:
        raise ValueError("Saved frame binding changed")
    return torch.as_tensor(values[None], device=device), frames


def make_timestamped_modules(arm, *, device, estimator=None, neighbors=None, trainable=True, seed=26, policy=TimingPolicy()):
    """Фабрика новых обучения и исполнения; старые контрольные точки не использует."""
    device = _concrete_device(device)
    if arm in ("C", "D"):
        if estimator is None or any(p.device != device for p in estimator.network.parameters()):
            raise ValueError("Explicit estimator on the requested device required")
        if _concrete_device(estimator.flyvis.device) != device:
            raise ValueError("Estimator FlyVis device and network device must agree")
        with torch.device(device):
            raw = FlyBackbone(estimator, trainable=bool(trainable and arm == "D"))
    elif arm == "B":
        if neighbors is None:
            raise ValueError("Geometry neighbors required for matched B control")
        _, cls = make_modules(torch)
        raw = _seeded_module(lambda: cls(neighbors), device, seed)
        raw.dev = device
        raw.requires_grad_(trainable)
    elif arm == "R1":
        raw = ReceptorHistory(device)
    else:
        raise ValueError("Expected C, D, B or R1")
    head = _seeded_module(TemporalResidualAssociation, device, seed)
    head.requires_grad_(trainable)
    head.train(trainable)
    if arm == "B":
        raw.train(trainable)
    return TimestampedBackbone(raw, arm, policy), head


def timestamped_checkpoint(backbone, head, *, cursor, optimizer=None, provenance=None):
    """Состояние с обязательной версией; отдельно от прежних моделей Б1."""
    raw, arm = backbone.backend, backbone.arm
    state = raw.net.state_dict() if arm in ("C", "D") else raw.state_dict() if arm == "B" else {}
    # Clone: subsequent optimizer steps must not mutate an in-memory checkpoint.
    result = dict(arm=arm, temporal_semantics=backbone.policy.metadata(), cursor=_cursor(cursor),
                  backbone={k:v.detach().clone() for k,v in state.items()},
                  head={k:v.detach().clone() for k,v in head.state_dict().items()},
                  provenance=dict(provenance or {}))
    if optimizer is not None:
        result["optimizer"] = deepcopy(optimizer.state_dict())
    return result


def restore_timestamped_checkpoint(checkpoint, backbone, head, *, optimizer=None):
    from tracking.flyvis_checkpoint import validate_state
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint mapping required")
    if checkpoint.get("temporal_semantics") != backbone.policy.metadata():
        raise ValueError("Legacy or incompatible timing checkpoint; train a separately versioned model")
    if checkpoint.get("arm") != backbone.arm:
        raise ValueError("Checkpoint arm mismatch")
    cursor = _cursor(checkpoint.get("cursor"))
    for name in ("head", "backbone"):
        state = checkpoint.get(name)
        if not isinstance(state, Mapping) or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError(f"Invalid or nonfinite {name} checkpoint")
    if backbone.arm == "R1" and checkpoint["backbone"]:
        raise ValueError("R1 has no learned backbone weights")
    if optimizer is not None and "optimizer" not in checkpoint:
        raise ValueError("Optimizer state required for resume")
    raw = backbone.backend
    core = raw.net if backbone.arm in ("C", "D") else raw if backbone.arm == "B" else None
    expected = core.state_dict() if core is not None else {}
    # Validate both components before mutating either. Strict load_state_dict
    # alone can copy earlier tensors before reporting a later missing key.
    for name, target in (("backbone", expected), ("head", head.state_dict())):
        source = checkpoint[name]
        if set(source) != set(target) or any(source[k].shape != target[k].shape for k in target):
            raise ValueError(f"Checkpoint {name} architecture mismatch")
        # Integer connectivity buffers must not be silently truncated by a cast.
        if any(not target[k].is_floating_point() and source[k].dtype != target[k].dtype for k in target):
            raise ValueError(f"Checkpoint {name} non-floating dtype mismatch")
        if name == "backbone" and backbone.arm == "B":
            for key in ("zr.nb", "hc.nb", "out.nb"):
                if not torch.equal(source[key].cpu(), target[key].cpu()):
                    raise ValueError("Checkpoint receptor geometry differs from runtime geometry")
    validate_state(head, checkpoint["head"], "head")
    if core is not None:
        validate_state(core, checkpoint["backbone"], "backbone")
    if optimizer is not None:
        # PyTorch may accept a bad moment tensor and fail only at optimizer.step.
        # Stage its load on private parameter/state copies before touching live weights.
        _finite_optimizer_values(checkpoint["optimizer"])
        staged = deepcopy(optimizer)
        staged.load_state_dict(deepcopy(checkpoint["optimizer"]))
        for parameter, state in staged.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Nonfinite optimizer state: {key}")
                    if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer") and value.shape != parameter.shape:
                        raise ValueError(f"Optimizer state shape mismatch: {key}")
    # A load hook or custom optimizer can still fail after validation. Roll back
    # all live tensors and optimizer state before propagating that failure.
    old_head = {k:v.detach().clone() for k,v in head.state_dict().items()}
    old_core = {k:v.detach().clone() for k,v in expected.items()}
    if optimizer is not None:
        # Keep original Parameter identities, while copying mutable state values.
        old_optimizer_state = type(optimizer.state)(dict) if hasattr(optimizer.state, "default_factory") else {}
        old_optimizer_state.update({p:deepcopy(state) for p,state in optimizer.state.items()})
        old_optimizer_groups = [{k:list(v) if k == "params" else deepcopy(v)
                                 for k,v in group.items()} for group in optimizer.param_groups]
    try:
        if core is not None:
            core.load_state_dict(checkpoint["backbone"], strict=True)
        head.load_state_dict(checkpoint["head"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(deepcopy(checkpoint["optimizer"]))
    except Exception:
        with torch.no_grad():
            for target, saved in ((head.state_dict(), old_head), (expected, old_core)):
                for key, value in target.items():
                    value.copy_(saved[key])
        if optimizer is not None:
            # Calling load_state_dict again would re-run the hook that just failed.
            optimizer.state = old_optimizer_state
            optimizer.param_groups = old_optimizer_groups
        raise
    return cursor


class TimestampedResidualEngine:
    """Реальная интеграция с Б1 через TimestampedB1Tracker; без меток и будущих кадров."""
    temporal_semantics = SCHEMA

    def __init__(self, backbone, head, checkpoint, *, source_fps, allow_diagnostic=False):
        self.bb, self.head, self.arm = backbone, head, backbone.arm
        self.source_fps = _fps(source_fps)
        if checkpoint.get("provenance", {}).get("diagnostic_only") and not allow_diagnostic:
            raise ValueError("Engineering smoke checkpoint is not a trained experimental model")
        self.cursor = restore_timestamped_checkpoint(checkpoint, backbone, head)
        self.head.eval().requires_grad_(False)
        raw = self.bb.backend
        if hasattr(raw, "net"):
            raw.net.eval().requires_grad_(False)
        elif hasattr(raw, "eval"):
            raw.eval().requires_grad_(False)
        self.last_pooled = None

    def start(self, history):
        if not history:
            raise ValueError("Observed history required")
        frames = [_frame(h["frame"]) for h in history]
        hx = torch.stack([h["hex"].reshape(-1) for h in history])[None]
        with torch.no_grad():
            state, activity = self.bb.sequence(hx, frames, self.source_fps)
        return dict(state=state, last=hx[:, -1], reference=activity[-1], last_frame=frames[-1])

    def advance_at(self, eye, current, frame):
        frame = _frame(frame)
        if frame <= eye["last_frame"]:
            raise ValueError("Live observations must advance time")
        current = current.reshape(1, -1)
        with torch.no_grad():
            state, activity = self.bb.step(eye["state"], eye["last"], current,
                previous_frame=eye["last_frame"], current_frame=frame, source_fps=self.source_fps)
        eye.update(state=state, current=activity, last=current, last_frame=frame)

    def correction(self, eye, masks, reference_mask, probability, zero=False):
        device = self.bb.dev
        count = len(masks)
        p = torch.as_tensor(probability[:count][None], device=device, dtype=torch.float64)
        real = torch.ones_like(p, dtype=torch.bool); real[:, 0] = False
        refs = torch.as_tensor(reference_mask[None], device=device)
        candidates = torch.as_tensor(masks[None], device=device)
        with torch.no_grad():
            safe = torch.nan_to_num(p, nan=.5).clamp(1e-8,1-1e-8).float()
            _, delta = self.head(torch.logit(safe),eye["reference"],eye["current"],refs,candidates,
                torch.ones_like(real),torch.tensor([True],device=device))
            updated = _checked_identity_probability(p, delta, real)
            if zero:
                delta = torch.zeros_like(delta)
                updated = p.clone()
            updated = updated[0].cpu().numpy()
            self.last_pooled = dict(reference=self.head.pool(eye["reference"],refs[:,None])[0,0].cpu().tolist(),
                current=self.head.pool(eye["current"],candidates)[0].cpu().tolist())
        return updated,delta[0].cpu().numpy()


class _FrameBoundEngine:
    """Часы принадлежат трекеру, даже когда вычислитель общий для двух контролей."""
    def __init__(self, engine):
        self.engine, self.frame = engine, None

    def start(self, history):
        return self.engine.start(history)

    def advance(self, eye, current):
        if self.frame is None:
            raise RuntimeError("Call tracker.begin before association")
        return self.engine.advance_at(eye, current, self.frame)

    def correction(self, *args, **kwargs):
        return self.engine.correction(*args, **kwargs)

    @property
    def last_pooled(self):
        return self.engine.last_pooled


from tracking.b1_tracker_runtime import B1Tracker


class TimestampedB1Tracker(B1Tracker):
    """Б1 с тем же сопоставлением; begin(frame, image) передаёт истинное время ядру."""
    def __init__(self, args, bank=None, geom=None, server=None, base=None, engine=None,
                 disabled=False, zero=False):
        if engine is not None and getattr(engine,"temporal_semantics",None) != SCHEMA:
            raise ValueError("TimestampedResidualEngine required; legacy engines are not interchangeable")
        bound = _FrameBoundEngine(engine) if engine is not None else None
        from ultralytics.trackers.basetrack import BaseTrack
        external = BaseTrack._count
        try:
            super().__init__(args,bank,geom,server,base,bound,disabled,zero)
        finally:
            BaseTrack._count = external
        self._observed_frame = None

    def begin(self, frame, eye_frame):
        frame = _frame(frame)
        if self._observed_frame is not None and frame <= self._observed_frame:
            raise ValueError("Tracker observations must increase strictly")
        super().begin(frame,eye_frame)
        self._observed_frame = frame
        if self.engine is not None:
            self.engine.frame = frame

    def reset(self):
        """Сбрасывает собственную историю; общий извлекатель/сервер не затрагивает."""
        from ultralytics.trackers.basetrack import BaseTrack
        external = BaseTrack._count
        try:
            super().reset()
        finally:
            BaseTrack._count = external
        self.hist.clear(); self.eyes.clear(); self.previous.clear(); self.events.clear()
        self._id_count = 0
        self._first = False
        self._observed_frame = None
        for name in ("frame", "eye_frame", "size"):
            self.__dict__.pop(name, None)
        if self.engine is not None:
            self.engine.frame = None
