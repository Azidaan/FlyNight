"""Independent current-vs-frozen CPU R1 transaction and valid-output checks."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import tracking.b1_timestamped as current
from tracking.b1_learning import ReceptorHistory

spec = importlib.util.spec_from_file_location('frozen_timestamped_transaction_review', OUT / 'before_b1_timestamped.py')
before = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = before
spec.loader.exec_module(before)
frames = torch.tensor([[.10, .20, .30, .40], [.15, .25, .35, .45],
                       [.20, .30, .40, .50], [.25, .35, .45, .55],
                       [.30, .40, .50, .60], [.35, .45, .55, .65]], dtype=torch.float32)
history = [dict(frame=i, hex=frames[i]) for i in range(4)]


def make_engine(module):
    raw = ReceptorHistory('cpu')
    engine = module.TimestampedResidualEngine.__new__(module.TimestampedResidualEngine)
    engine.bb = module.TimestampedBackbone(raw, 'R1')
    engine.source_fps = 30.
    return engine


def same_eye(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]) if torch.is_tensor(left[key]) else left[key] == right[key]


results = []
for defect in ('state_nan', 'state_shape', 'activity_nan', 'activity_shape', 'backend_error'):
    engine, frozen = make_engine(current), make_engine(before)
    eye, control = engine.start(history), frozen.start(history)
    same_eye(eye, control)
    saved = dict(eye)
    saved_values = {k:v.clone() for k,v in eye.items() if torch.is_tensor(v)}
    raw_step = engine.bb.backend.step
    def broken(*args):
        if defect == 'backend_error':
            raise RuntimeError('injected backend error')
        state, activity = raw_step(*args)
        if defect == 'state_nan':
            state = torch.full_like(state, float('nan'))
        elif defect == 'state_shape':
            state = state[:, :, :-1]
        elif defect == 'activity_nan':
            activity = torch.full_like(activity, float('nan'))
        elif defect == 'activity_shape':
            activity = activity[:, :, :-1]
        return state, activity
    engine.bb.backend.step = broken
    try:
        engine.advance_at(eye, frames[4], 4)
    except (ValueError, RuntimeError) as error:
        failure = str(error)
    else:
        raise AssertionError('Invalid backend output was committed')
    assert eye.keys() == saved.keys() and all(eye[k] is saved[k] for k in eye)
    assert all(torch.equal(eye[k], v) for k,v in saved_values.items())
    assert eye['last_frame'] == 3
    engine.bb.backend.step = raw_step
    engine.advance_at(eye, frames[4], 4)
    frozen.advance_at(control, frames[4], 4)
    same_eye(eye, control)
    engine.advance_at(eye, frames[5], 5)
    frozen.advance_at(control, frames[5], 5)
    same_eye(eye, control)
    results.append(dict(defect=defect, error=failure, rejected=True,
                        failed_eye_object_identity_preserved=True, failed_eye_tensor_values_preserved=True,
                        failed_clock=3, retry_exact_to_frozen_control=True, next_frame_exact_to_frozen_control=True))

engine = make_engine(current)
raw_init = engine.bb.backend.init
engine.bb.backend.init = lambda x: torch.full_like(raw_init(x), float('nan'))
try:
    engine.start(history)
except ValueError as error:
    initial_error = str(error)
else:
    raise AssertionError('Nonfinite initial state accepted')
engine.bb.backend.init = raw_init
same_eye(engine.start(history), make_engine(before).start(history))
source = ROOT / 'tracking/b1_timestamped.py'
report = dict(scope='CPU inference-only; real R1 state; no learned weights constructed or loaded',
              before_source_sha256=hashlib.sha256((OUT/'before_b1_timestamped.py').read_bytes()).hexdigest(),
              reviewed_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              cases=results, initial_nonfinite_error=initial_error, initial_clean_retry_exact=True,
              cuda_initialized=torch.cuda.is_initialized())
assert not report['cuda_initialized']
(OUT/'reviewed_b1_timestamped_after.py').write_bytes(source.read_bytes())
(OUT/'timestamped_after_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
