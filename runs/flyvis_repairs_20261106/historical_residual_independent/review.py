"""Independent bounded CPU check of the historical engine repair."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast
import copy
import hashlib
import importlib.util
import json
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SAVED = ROOT / 'runs/flyvis_system_fixes_20261106/historical_residual_transaction'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from tracking.b1_tracker_runtime import ResidualEngine
from tracking.b1_learning import ReceptorHistory
from tracking.b1_deterministic_runtime import deterministic_call
from tracking.b1_fast_stimulus import fast_init
from tracking.b1_index_copy_stimulus import index_copy_step
from tracking.z13_readout import FlyBackbone
from tracking.flyvis_motion import FlyVisMotionEstimator

assert not (OUT / 'report.json').exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_path = SAVED / 'before_b1_tracker_runtime.py'
spec = importlib.util.spec_from_file_location('independent_old_residual', old_path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

def strip_changed(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'ResidualEngine':
            node.body = [part for part in node.body if not (isinstance(part, ast.FunctionDef)
                and part.name in ('start', 'advance', '_check_candidate'))]
    return ast.dump(tree, include_attributes=False)

assert strip_changed(old_path) == strip_changed(ROOT / 'tracking/b1_tracker_runtime.py')
saved_arrays = []
with np.load(SAVED / 'native_before_outputs.npz') as a, np.load(SAVED / 'native_after_outputs.npz') as b:
    assert set(a.files) == set(b.files)
    for key in sorted(a.files):
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
        assert np.array_equal(a[key], b[key]) and np.isfinite(b[key]).all()
        saved_arrays.append(dict(key=key, exact=True, shape=list(b[key].shape), dtype=str(b[key].dtype)))
assert len(saved_arrays) == 12
old_report = json.loads((SAVED / 'native_before_report.json').read_text())
new_report = json.loads((SAVED / 'native_after_report.json').read_text())
assert old_report['checkpoints'] == new_report['checkpoints']
for arm, expected in new_report['checkpoints'].items():
    assert sha(ROOT / f'runs/b123_20261027/b1_training/{arm}/checkpoint.pt') == expected
assert new_report['current_source_sha256'] == sha(ROOT / 'tracking/b1_tracker_runtime.py')

def engine(cls, backend, arm):
    obj = cls.__new__(cls)
    obj.bb, obj.arm = backend, arm
    return obj

def tensors(eye, arm):
    return {key: (value if arm == 'R1' else value.nodes.activity) if key == 'state' else value
            for key, value in eye.items()}

def exact(a, b, arm):
    assert set(a) == set(b)
    for key, value in tensors(a, arm).items():
        other = tensors(b, arm)[key]
        assert value.dtype == other.dtype and value.device == other.device
        assert torch.equal(value, other), key

def snapshot(eye, arm):
    return dict(eye), {key: value.clone() for key, value in tensors(eye, arm).items()}

def unchanged(eye, snap, arm):
    refs, values = snap
    assert set(eye) == set(refs) and all(eye[key] is refs[key] for key in refs)
    assert all(torch.equal(tensors(eye, arm)[key], value) for key, value in values.items())

checks = []
initial_rng = torch.random.get_rng_state().clone()
initial_flags = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
try:
    for dtype in (torch.float32, torch.float64):
        raw = ReceptorHistory('cpu')
        values = torch.linspace(.15, .81, 42, dtype=dtype).reshape(6, 7)
        history = [dict(hex=row) for row in values[:4]]
        before, after = engine(old.ResidualEngine, raw, 'R1'), engine(ResidualEngine, raw, 'R1')
        left, right = before.start(history), after.start(history)
        exact(left, right, 'R1')
        for row in values[4:]:
            before.advance(left, row); after.advance(right, row)
            exact(left, right, 'R1')
        assert right['state'].dtype == dtype and right['last'].dtype == dtype
        unchanged_input = after.start(history)
        snap = snapshot(unchanged_input, 'R1')
        step = raw.step
        raw.step = lambda *args: (_ for _ in ()).throw(AssertionError('dtype mismatch reached backend'))
        try:
            try:
                after.advance(unchanged_input, values[4].to(torch.float64 if dtype == torch.float32 else torch.float32))
            except ValueError:
                pass
            else:
                raise AssertionError('dtype mismatch accepted')
        finally:
            raw.step = step
        unchanged(unchanged_input, snap, 'R1')
        after.advance(unchanged_input, values[4])
        clean = before.start(history); before.advance(clean, values[4])
        exact(unchanged_input, clean, 'R1')
        checks.append(dict(arm='R1', dtype=str(dtype), valid_exact=True, rejected_before_backend=True, retry_exact=True))

    estimator = FlyVisMotionEstimator(device='cpu', source_fps=30., dt=.02,
        fade_in_seconds=.02, flow_gain=1., geometry_version='legacy_v1')
    for arm in ('C', 'D'):
        ck = torch.load(ROOT / f'runs/b123_20261027/b1_training/{arm}/checkpoint.pt', map_location='cpu', weights_only=False)
        raw = FlyBackbone(estimator, trainable=False)
        raw.net.load_state_dict(ck['backbone'], strict=True)
        raw.init = deterministic_call(MethodType(fast_init, raw))
        normal_step = MethodType(index_copy_step, raw)
        raw.step = deterministic_call(normal_step)
        raw.net.eval().requires_grad_(False)
        columns = raw.index.shape[1]
        observations = .45 + .08 * torch.cos(torch.arange(columns)[None] / 11 + torch.arange(6)[:, None] / 7)
        history = [dict(hex=row) for row in observations[:4]]
        before, after = engine(old.ResidualEngine, raw, arm), engine(ResidualEngine, raw, arm)
        torch.use_deterministic_algorithms(False, warn_only=True)
        expected_flags = (False, True)
        left, right = before.start(history), after.start(history)
        exact(left, right, arm)
        assert (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()) == expected_flags
        for fault in ('unselected_neuron_nan', 'wrapped_exception'):
            eye = after.start(history)
            snap = snapshot(eye, arm)
            selected = set(raw.index.reshape(-1).tolist())
            unseen = next(i for i in range(raw.net.stimulus.n_nodes) if i not in selected)
            def broken(*args):
                assert torch.are_deterministic_algorithms_enabled()
                state, activity = normal_step(*args)
                if fault == 'wrapped_exception':
                    raise RuntimeError('failure after native step')
                state.nodes.activity[0, unseen] = float('nan')
                assert torch.isfinite(activity).all(), 'Fault must be outside selected readout'
                return state, activity
            raw.step = deterministic_call(broken)
            try:
                after.advance(eye, observations[4])
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError('Native failure accepted')
            assert (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()) == expected_flags
            unchanged(eye, snap, arm)
            raw.step = deterministic_call(normal_step)
            control = before.start(history)
            for row in observations[4:]:
                after.advance(eye, row); before.advance(control, row)
                exact(eye, control, arm)
            checks.append(dict(arm=arm, fault=fault, wrapped=True, old_eye_refs_and_values_preserved=True,
                warn_only_flags_preserved=True, retry_and_next_exact=True))
        print(arm, 'independent native wrapped faults/retry passed', flush=True)
finally:
    torch.random.set_rng_state(initial_rng)
    torch.use_deterministic_algorithms(initial_flags[0], warn_only=initial_flags[1])
assert not torch.cuda.is_initialized()
report = dict(source_sha256=sha(ROOT / 'tracking/b1_tracker_runtime.py'), baseline_sha256=sha(old_path),
    deterministic_source_sha256=sha(ROOT / 'tracking/b1_deterministic_runtime.py'),
    independent_probe_sha256=sha(__file__), unchanged_AST_outside_three_methods=True,
    saved_native_arrays=saved_arrays, actual_checkpoint_hashes=new_report['checkpoints'],
    independent_checks=checks, cuda_initialized=False,
    limitations='Checks atomic eye publication for functional official backends only; internal stimulus buffers are not rolled back; no complete tracker transaction, no GPU or training, no HOTA claim.')
(OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
