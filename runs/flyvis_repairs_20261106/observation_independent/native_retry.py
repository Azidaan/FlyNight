"""Native CPU exception/retry with official functional recurrent state."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from tracking.flyvis_motion import FlyVisMotionEstimator

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
source=ROOT/'tracking/flyvis_motion.py';source_hash=sha(source)
before_model=OUT/'native_protocol.json'
assert not before_model.exists()
before_model.write_text(json.dumps(dict(source_sha256=source_hash,script_sha256=sha(__file__),
    scope='One native CPU estimator; decoder exceptions before publication, retry compared exactly with reset clean replay. No training or HOTA.',
    geometry_version='legacy_v1',source_fps=30.,flow_gain=1.,criteria=['First failure restores uninitialized layout/state','Later decoder failure after simulate preserves published tensor and NumPy objects and values','Correct retry equals clean reset replay exactly']),indent=2),encoding='utf-8')
e=FlyVisMotionEstimator(source_fps=30.,flow_gain=1.,device='cpu')
assert e.geometry_version=='legacy_v1'
rng=np.random.default_rng(61);image=rng.integers(0,256,(80,128),dtype=np.uint8)
frames=[np.roll(image,i,axis=1) for i in range(3)]
decoder=e.decoder
def fail(activity):raise RuntimeError('independent injected decoder failure')

e.decoder=fail;before=e.stats.copy()
try:e.observe(frames[0])
except RuntimeError as exc:assert 'injected decoder' in str(exc)
else:raise AssertionError('first fault did not raise')
assert e._state is None and e._last_hex is None and e._baseline_flow is None
assert e._bounds is None and e._frame_size is None and e.baseline_activity is None
assert e.stats==before
e.decoder=decoder
for frame in frames[:2]:e.observe(frame)
names=['_state','_last_hex','_baseline_flow','baseline_activity','last_activity','last_decoded']
objects={name:getattr(e,name) for name in names}
values={name:(value.nodes.activity.detach().clone() if name=='_state' else value.detach().clone() if isinstance(value,torch.Tensor) else value.copy()) for name,value in objects.items()}
before=e.stats.copy();e.decoder=fail
try:e.observe(frames[2])
except RuntimeError as exc:assert 'injected decoder' in str(exc)
else:raise AssertionError('later fault did not raise')
assert e.stats==before
for name,value in objects.items():
    assert getattr(e,name) is value,name
    actual=value.nodes.activity if name=='_state' else value
    if isinstance(actual,torch.Tensor):assert torch.equal(actual,values[name]),name
    else:assert np.array_equal(actual,values[name]),name
e.decoder=decoder
actual=e.observe(frames[2]);actual_activity=e.last_activity.copy();actual_state=e._state.nodes.activity.detach().clone()
arrays={'actual_points':actual.points_xy.copy(),'actual_vectors':actual.vectors_xy.copy(),'actual_activity':actual_activity}
for agg in ['mean','last','sum']:arrays['actual_'+agg]=e.field_from_last(agg).vectors_xy.copy()
e.reset()
for frame in frames:expected=e.observe(frame)
assert np.array_equal(actual.points_xy,expected.points_xy)
assert np.array_equal(actual.vectors_xy,expected.vectors_xy)
assert np.array_equal(actual_activity,e.last_activity)
assert torch.equal(actual_state,e._state.nodes.activity)
arrays.update(expected_points=expected.points_xy,expected_vectors=expected.vectors_xy,expected_activity=e.last_activity)
for agg in ['mean','last','sum']:
    arrays['expected_'+agg]=e.field_from_last(agg).vectors_xy
    assert np.array_equal(arrays['actual_'+agg],arrays['expected_'+agg])
assert sha(source)==source_hash and not torch.cuda.is_initialized()
np.savez_compressed(OUT/'native_retry_outputs.npz',**arrays)
report=dict(passed=True,source_sha256=source_hash,first_decoder_exception_preserves_layout_and_state=True,
    later_decoder_exception_preserves_objects_values_stats=True,retry_equals_clean_reset_replay_exact=True,
    state_exact=True,activity_exact=True,aggregations_exact=['mean','last','sum'],cuda_initialized=False,
    outputs_sha256=sha(OUT/'native_retry_outputs.npz'),protocol_sha256=sha(before_model),script_sha256=sha(__file__),
    boundary='Official backend and standard decoder, CPU one synthetic image sequence. Custom in-place backends and external side effects are outside the atomic publication claim.')
(OUT/'native_retry_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
