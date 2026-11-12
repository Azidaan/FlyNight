"""CPU-only independent scalar scoring of saved descriptors and availability."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from common import *
import numpy as np,math
recovery=read(OUT/'audit_recovery.json');assert sha(__file__)==recovery['new_auditor_sha256'];p=verify();r=read(OUT/'result.json');assert r['passed'] and r['duplicate_roi_exact'] and r['cpu_gpu_preprocess_exact'];assert sha(OUT/'descriptors.npz')==r['descriptors_sha256']
maximum=0.;checked=0
with np.load(OUT/'descriptors.npz') as z:
    def cosine(a,b):
        return math.fsum(float(x)*float(y) for x,y in zip(a,b))/math.sqrt(math.fsum(float(x)**2 for x in a)*math.fsum(float(y)**2 for y in b))
    def vector(slot):return z[slot[0]][slot[1]]
    for c in p['cases']:
        cr=next(x for x in r['records'] if x['old_id']==c['old_id']);s=r['slots'][str(c['old_id'])];ref=vector(s['reference']);assert [x['raw_id'] for x in cr['operational_candidates']]==[x['raw_id'] for x in c['candidates']]
        independent=[]
        for row,slot in zip(cr['operational_candidates'],s['candidates']):
            value=cosine(ref,vector(slot['slot']));maximum=max(maximum,abs(value-row['cosine']));checked+=1;independent.append(value)
            assert row['hits']==hits(np.asarray(row['box'],np.float32),objects(c['video'],c['current_frame']))
            independently64=hits(row['box'],objects(c['video'],c['current_frame']));assert [h['identity'] for h in independently64]==[h['identity'] for h in row['hits']];assert [h['box'] for h in independently64]==[h['box'] for h in row['hits']]
            assert row['own']==([h['identity'] for h in row['hits']]==[c['identity']])
        for row,value in zip(cr['operational_candidates'],independent):assert row['rank']==1+sum(v>value for v in independent)
        own=[v for row,v in zip(cr['operational_candidates'],independent) if row['own']];other=[v for row,v in zip(cr['operational_candidates'],independent) if len(row['hits'])==1 and not row['own']]
        assert cr['own_available']==bool(own)
        if own and other:maximum=max(maximum,abs((max(own)-max(other))-cr['own_minus_foreign_margin']))
        value=cosine(ref,vector(s['desired']));maximum=max(maximum,abs(value-cr['desired']['cosine']));checked+=1
        for row,slot in zip(cr['oracle'],s['oracle']):maximum=max(maximum,abs(cosine(ref,vector(slot['slot']))-row['cosine']));checked+=1
    for frame in r['frames']:
        assert frame['pixel_sha256']==p['expected_raw_frame_hashes'][frame['key']]
        arr=z[frame['key']];assert arr.shape==(frame['count'],256) and np.isfinite(arr).all() and np.max(abs(np.linalg.norm(arr.astype(float),axis=1)-1))<1e-5
assert maximum<1e-6
verify();save('audit_v2.json',dict(passed=True,checked_scores=checked,max_scalar_error=maximum,new_model_calls=0,source_checksums_verified=True,operational_candidates_complete=True,dtype_recovery='Original float32 detector box arithmetic reproduced exactly; independent float64 IoU produces identical identity sets and availability. Original failed auditor and results preserved.',scope='Independent scalar cosine/ranks/availability and recorded frame hashes; stored descriptors shared. No independent neural forward or physical correspondence proof.'))
print('AUDIT_PASSED',checked,maximum)
