"""Distinct current detections recover a stale pointer; ties remain unknown."""
from common import *
start('cpu',timeout=600)
import numpy as np
from repair_public_provenance import repair
raw=dict(frame_idx=np.array([0,1,1,2,2]),conf=np.array([.9,.8,.7,.6,.6],np.float32),cls=np.array([2]*5))
flat=np.array([[1,0,0,1,1,1,.8,2],[1,0,0,1,1,2,.7,2],[2,0,0,1,1,1,.6,2]],np.float32)
assign=[dict(frame=1,tid=1,raw_id=0,start_frame=1,emitted=True),
    dict(frame=1,tid=2,raw_id=2,start_frame=1,emitted=True),
    dict(frame=2,tid=1,raw_id=0,start_frame=1,emitted=True),
    dict(frame=2,tid=3,raw_id=1,start_frame=2,emitted=False)]
result,counts,witnesses=repair(raw,flat,assign)
assert result[0]['raw_id']==1 and result[0]['logged_raw_id']==0
assert result[1]['raw_id']==2 and 'logged_raw_id' not in result[1]
assert result[2]['raw_id'] is None and result[3]['raw_id'] is None
assert counts['unknown_pointer']==2 and counts['recovered_unique_public_pointer']==1
assert assign[0]['raw_id']==0,'Original log mutated'
write('provenance_math_test.json',dict(passed=True,unique_exact_recovered=True,tie_and_nonpublic_remain_unknown=True,
    original_log_preserved=True,no_GT_or_nearest_box=True,zero_model_or_tracker_calls=True))
print('Public provenance invariant tests passed',flush=True)
