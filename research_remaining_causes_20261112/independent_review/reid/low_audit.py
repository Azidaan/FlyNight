"""CPU scalar independent audit, preserving all raw and unknown candidates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from common import *
import numpy as np,math
p=read(OUT/'low_protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
verify();r=read(OUT/'low_result.json');orig=read(OUT/'result.json');assert sha(OUT/'low_descriptors.npz')==r['descriptors_sha256'];maximum=0.;checked=0;summary=[]
def cosine(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))/math.sqrt(math.fsum(float(x)**2 for x in a)*math.fsum(float(y)**2 for y in b))
with np.load(OUT/'low_descriptors.npz') as z,np.load(OUT/'descriptors.npz') as old:
    for c in p['cases']:
        cid=str(c['old_id']);cr=next(v for v in r['records'] if v['old_id']==c['old_id']);slot=orig['slots'][cid]['reference'];ref=old[slot[0]][slot[1]];rows=cr['candidates'];values=[]
        with np.load(ROOT/p['videos'][c['video']]['detections']) as raw:expected=np.flatnonzero((raw['frame_idx']==c['current_frame'])&(raw['cls']==2));assert [x['raw_id'] for x in rows]==expected.tolist()
        for i,row in enumerate(rows):
            value=cosine(ref,z[cid][i]);maximum=max(maximum,abs(value-row['cosine']));values.append(value);checked+=1
            assert row['hits']==hits(np.asarray(row['box'],np.float32),objects(c['video'],c['current_frame']))
            ids=[x['identity'] for x in hits(row['box'],objects(c['video'],c['current_frame']))];assert ids==[x['identity'] for x in row['hits']];assert row['own']==(ids==[c['identity']])
            assert row['native_first']==(row['confidence']>=.25) and row['native_second']==(.1<row['confidence']<.25)
        for row,value in zip(rows,values):assert row['rank_all_raw']==1+sum(v>value for v in values)
        own=[v for row,v in zip(rows,values) if row['own']];others=[v for row,v in zip(rows,values) if not row['own']]
        assert cr['own_available_all_raw']==bool(own)
        if own and others:maximum=max(maximum,abs(max(own)-max(others)-cr['own_minus_all_other']))
        summary.append(dict(old_id=c['old_id'],raw_candidates=len(rows),own_candidates=[dict(raw_id=x['raw_id'],confidence=x['confidence'],cosine=x['cosine'],rank=x['rank_all_raw'],native_first=x['native_first'],native_second=x['native_second']) for x in rows if x['own']],top=sorted([dict(raw_id=x['raw_id'],cosine=x['cosine'],rank=x['rank_all_raw'],own=x['own'],unmatched=x['annotation_unmatched']) for x in rows],key=lambda x:x['rank'])[:3]))
assert maximum<1e-6
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
verify();save('low_audit.json',dict(passed=True,checked_scores=checked,max_scalar_error=maximum,summary=summary,model_calls=0,threshold_changed=False,scope='Independent scalar cosines/ranks and complete raw-car candidate inventory. Identity IoUsets checked in original float32 and independent float64; no new fulltracker or rejectionpolicy.'))
print('LOW_AUDIT_PASS',checked,maximum)
