"""Exact replay of cached eye input through two causal recovery routes."""
from common import *
start('cpu')
import numpy as np
import torch
from sparse_views import SparseViews
torch.set_num_threads(2)
p=read(OUT/'protocol.json');verify(p)
assert not (OUT/'input_witness.json').exists()
video=p['witnesses'][0]['video'];checkpoint=SparseViews(video);initial=SparseViews(video,force_initial=True);records=[]
try:
    for w in p['witnesses']:
        expected=np.load(ROOT/w['cached'],allow_pickle=False)
        a=checkpoint.get(w['frame'],w['crop'],w['frame'],force_restore=True)
        b=initial.get(w['frame'],w['crop'],w['frame'],force_restore=True)
        ea=float(np.max(np.abs(a.astype(float)-expected.astype(float))));eb=float(np.max(np.abs(b.astype(float)-expected.astype(float))))
        records.append(dict(frame=w['frame'],crop=w['crop'],checkpoint_max_error=ea,initial_max_error=eb,checkpoint_exact=np.array_equal(a,expected),initial_exact=np.array_equal(b,expected),cached_sha256=sha(ROOT/w['cached']),dtype=str(expected.dtype)))
        print('INPUT WITNESS',w['frame'],ea,eb,flush=True)
        assert np.array_equal(a,expected) and np.array_equal(b,expected),(w,ea,eb)
finally:
    checkpoint.close();initial.close()
assert not torch.cuda.is_initialized()
verify(p)
write('input_witness.json',dict(passed=True,records=records,checkpoint_decoded_frames=checkpoint.frames.decoded,initial_decoded_frames=initial.frames.decoded,
    used_sources={**checkpoint.frames.sources,**initial.frames.sources},no_new_huge_frame_array=True,no_core_import=not any(k=='flyvis' or k.startswith('flyvis.') for k in sys.modules),no_new_rollout_or_HOTA=True))
status('input_witness_passed',views=len(records))
print('Exact native-precision input recovery passed',flush=True)
