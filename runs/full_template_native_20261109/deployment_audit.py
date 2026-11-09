"""Independently verify deployed parameters and both full zero passes."""
from common import *
start('cpu',timeout=600)
import numpy as np
from scipy.special import expit
p=read(OUT/'execution_protocol.json');verify(p)
data=read(TRAIN/'features.json')['records'];deployed=read(OUT/'deployment.json')
x=np.array([r['geometry'][:7] for r in data]);y=np.array([r['label']=='own' for r in data],int)
mean=x.mean(0);scale=x.std(0);scale[np.ptp(x,axis=0)==0]=1
assert np.array_equal(mean,deployed['geometry_normalization']['mean'])
assert np.array_equal(scale,deployed['geometry_normalization']['scale'])
g=(x-mean)/scale;weights=len(y)/(2*np.bincount(y)[y]);checks={}
for arm,model in deployed['models'].items():
    matrix=g if arm=='G7' else np.column_stack([g,np.array([r['variants'][arm]['full']['channels'] for r in data])])
    theta=np.array(model['theta']);design=np.column_stack([matrix,np.ones(len(y))]);z=design@theta
    penalty=theta.copy();penalty[-1]=0
    gradient=design.T@(weights*(expit(z)-y))/len(y)+penalty
    loss=float(np.mean(weights*(np.logaddexp(0,z)-y*z))+.5*(penalty@penalty))
    residual=float(abs(gradient).max());assert residual<1e-7 and abs(loss-model['loss'])<1e-12
    checks[arm]=dict(gradient_max=residual,loss=loss)
zero=read(OUT/'zero_complete.json');assert zero['runs']==40 and zero['all_full_raw_arrays_exact'] and zero['all_annotated_JSON_exact']
receipts=[]
for arm in ['zero_disabled','zero_empty']:
    for video in read(OUT/'protocol.json')['videos']:
        r=read(OUT/'rollouts'/arm/(video+'.receipt.json'))
        assert r['passed'] and r['exact_full_raw_and_JSON'] and not r['core_imported']
        assert r['execution_sha256']==sha(OUT/'execution_protocol.json')
        for name,digest in r['sources'].items():assert sha(ROOT/name)==digest
        receipts.append(dict(video=video,arm=arm,output_rows=r['output_rows'],active_empty_solver_frames=r['active_empty_solver_frames']))
write('deployment_audit.json',dict(passed=True,models=checks,zero_receipts=receipts,
    zero_runs=40,active_empty_solver_frames=sum(r['active_empty_solver_frames'] for r in receipts),
    no_refit=True,no_new_maps=True,no_tracker_rerun=True))
print('INDEPENDENT DEPLOYMENT AND ZERO RECEIPTS PASSED',flush=True)
