"""Independent small failure checks against the captured pre-fix source."""
from pathlib import Path
import importlib.util
import sys
import json
import hashlib
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

before = load('observation_pre_fix', OUT / 'before_flyvis_motion.py')
fixture = load('observation_captured_fixture', OUT / 'before_test_flyvis_observation_transaction.py')
fixture.FlyVisMotionEstimator = before.FlyVisMotionEstimator
cases = {}
for phase, failure in [('first', 'raise'), ('later', 'raise'), ('later', 'wrong_shape')]:
    model = fixture.model()
    if phase == 'later':
        model.observe(np.array([[1.]], np.float32))
    previous = model._state
    old_stats = dict(model.stats)
    model.decoder_mode = failure
    try:
        model.observe(np.array([[8.]], np.float32))
        exception = None
    except Exception as error:
        exception = type(error).__name__
    cases[phase + '_' + failure] = dict(exception=exception, same_state_reference=model._state is previous,
                                      previous_stats=old_stats, current_stats=model.stats)

model = fixture.model()
model.mode = 'full'
model._frame_size = model._bounds = None
render = model._render
def render_layout(frame):
    model._prepare_layout(frame.shape[1], frame.shape[0])
    return render(frame)
model._render = render_layout
model.decoder_mode = 'raise'
try:
    model.observe(np.ones((3, 4), np.float32))
except RuntimeError:
    pass
after_failed = dict(frame_size=model._frame_size, bounds=model._bounds)
model.decoder_mode = 'valid'
try:
    model.observe(np.ones((3, 6), np.float32))
    retry = 'accepted'
except Exception as error:
    retry = repr(error)
cases['first_layout_failure'] = dict(after_failed=after_failed, retry_new_shape=retry)
report = dict(captured_source_sha256=hashlib.sha256((OUT / 'before_flyvis_motion.py').read_bytes()).hexdigest(),
              cases=cases, scope='No runtime network/model initialization; bounded injected backend failures')
(OUT / 'additional_before_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
