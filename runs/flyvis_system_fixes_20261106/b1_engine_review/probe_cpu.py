"""Real CPU R1 recurrence, one injected backend failure; no learned weights."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import json
import hashlib
import importlib.util
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tracking.b1_learning import ReceptorHistory
from tracking.flyvis_temporal_residual import TemporalResidualAssociation

def snapshot_module(name, file):
    spec = importlib.util.spec_from_file_location(name, OUT / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

timestamped = snapshot_module('b1_timestamped_before_review', 'before_b1_timestamped.py')
legacy = snapshot_module('b1_runtime_before_review', 'before_b1_tracker_runtime.py')
frames = torch.tensor([[.10, .20, .30, .40], [.15, .25, .35, .45],
                       [.20, .30, .40, .50], [.25, .35, .45, .55],
                       [.30, .40, .50, .60], [.35, .45, .55, .65]], dtype=torch.float32)
history = [dict(frame=i, hex=frames[i]) for i in range(4)]
report = {}

def head_guard(reference, current):
    # Invoke the real stateless input guard before it accesses trained layers.
    # No head weights are constructed, loaded, or modified in this probe.
    return TemporalResidualAssociation.forward(None, torch.zeros(1, 2), reference, current,
        torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 2, 4, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool), torch.ones(1, dtype=torch.bool))

for name in ('timestamped', 'legacy'):
    raw = ReceptorHistory('cpu')
    if name == 'timestamped':
        engine = timestamped.TimestampedResidualEngine.__new__(timestamped.TimestampedResidualEngine)
        engine.bb = timestamped.TimestampedBackbone(raw, 'R1')
        engine.source_fps = 30.
    else:
        engine = legacy.ResidualEngine.__new__(legacy.ResidualEngine)
        engine.bb = raw
    eye = engine.start(history)
    previous_state = eye['state']
    previous_last = eye['last'].clone()
    normal_step = raw.step
    def broken_step(state, previous, current):
        state, activity = normal_step(state, previous, current)
        return torch.full_like(state, float('nan')), torch.full_like(activity, float('nan'))
    raw.step = broken_step
    try:
        if name == 'timestamped':
            engine.advance_at(eye, frames[4], 4)
        else:
            engine.advance(eye, frames[4])
        accepted = True
    except Exception as error:
        accepted = False
        raise AssertionError('This before-source probe expects the confirmed defect') from error
    raw.step = normal_step
    try:
        head_guard(eye['reference'], eye['current'])
        downstream_error = None
    except ValueError as error:
        downstream_error = str(error)
    record = dict(finite_original_input=bool(torch.isfinite(frames).all()),
                  finite_original_state=bool(torch.isfinite(previous_state).all()),
                  bad_backend_output_accepted=accepted,
                  state_reference_changed=eye['state'] is not previous_state,
                  current_state_finite=bool(torch.isfinite(eye['state']).all()),
                  last_input_changed=not torch.equal(eye['last'], previous_last),
                  downstream_actual_head_error=downstream_error)
    if name == 'timestamped':
        record['committed_last_frame'] = eye['last_frame']
        try:
            engine.advance_at(eye, frames[4], 4)
            record['same_frame_retry_error'] = None
        except ValueError as error:
            record['same_frame_retry_error'] = str(error)
        engine.advance_at(eye, frames[5], 5)
    else:
        engine.advance(eye, frames[5])
    record['next_clean_step_state_finite'] = bool(torch.isfinite(eye['state']).all())
    assert accepted and not record['current_state_finite'] and downstream_error == 'Non-finite FlyVis activity'
    assert not record['next_clean_step_state_finite']
    report[name] = record

result = dict(scope='CPU inference-only R1 recurrence, no learned head/core construction, no training or data files',
              cuda_initialized=torch.cuda.is_initialized(), frames=frames.tolist(), cases=report,
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('before_*.py')})
assert not result['cuda_initialized']
(OUT / 'before_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
