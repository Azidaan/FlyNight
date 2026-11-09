"""Verify the corrected checkpoint decoder before any real-ID intervention."""
from common import *
start('cpu')
import numpy as np
import torch
from sequential_views import SequentialViews
torch.set_num_threads(2)
p=read(OUT/'protocol.json');verify(p);recovery=read(OUT/'input_recovery_protocol.json');verify(recovery)
assert not (OUT/'input_witness_recovery.json').exists()
records=[];used={};first=p['witnesses'][0]['video']
for video in p['videos']:
    checkpoint=SequentialViews(video);initial=SequentialViews(video,force_initial=True) if video==first else None
    cases=[w for w in recovery['witnesses'] if w['video']==video]
    if video==first:
        cases=p['witnesses']+cases
        unique={tuple([w['frame']]+w['crop']):w for w in cases};cases=sorted(unique.values(),key=lambda w:(w['frame'],w['crop']))
    try:
        for w in cases:
            expected=np.load(ROOT/w['cached'],allow_pickle=False);actual=checkpoint.get(w['frame'],w['crop'],w['frame'],force_restore=True)
            error=float(np.max(np.abs(actual.astype(float)-expected.astype(float))));start_error=None
            if initial is not None:
                fresh=initial.get(w['frame'],w['crop'],w['frame'],force_restore=True);start_error=float(np.max(np.abs(fresh.astype(float)-expected.astype(float))));assert np.array_equal(fresh,expected),(w,start_error)
            records.append(dict(video=video,frame=w['frame'],crop=w['crop'],checkpoint_error=error,initial_error=start_error,cached_sha256=sha(ROOT/w['cached']),dtype=str(expected.dtype)))
            assert np.array_equal(actual,expected),(w,error)
            print('EXACT INPUT',video,w['frame'],error,start_error,flush=True)
        used.update(checkpoint.frames.sources)
        if initial is not None:used.update(initial.frames.sources)
    finally:
        checkpoint.close()
        if initial is not None:initial.close()
    status('input_recovery',video=video,done=len({r['video'] for r in records}),total=20)
assert len({r['video'] for r in records})==20 and not torch.cuda.is_initialized()
verify(p);verify(recovery)
write('input_witness_recovery.json',dict(passed=True,videos=20,records=records,all_arrays_exact=True,used_sources=used,
    correction='Sequential raw prefix decoding with checkpoint retina state, no seek equivalence assumed.',no_new_huge_frame_array=True,no_new_core_or_rollout=True))
status('input_recovery_passed',videos=20)
print('All20 corrected native-precision input witnesses passed',flush=True)
