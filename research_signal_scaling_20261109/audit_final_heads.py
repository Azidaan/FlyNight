"""Independently recompute final objective, gradients and all fifteen new logits."""
import json,hashlib,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
for n,h in read(OUT/'deployment_screen_protocol.json')['sources'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
d=read(OUT/'deployment.json');s=read(OUT/'deployment_screen.json');old=read(ROOT/'runs/domain_matched_native_20261109/deployment.json');rows=read(ROOT/'research_domain_matched_reader_20261109/features.json')['records'];cases=read(ROOT/'research_identity_training_stability_20261109/comparison.json')['cases']
assert d['geometry_normalization']==old['geometry_normalization'] and d['models']['G7']==old['models']['G7']
G=np.array([r['geometry'] for r in rows]);stats=d['geometry_normalization'];g=(G-stats['mean'])/stats['scale'];y=np.array([r['label']=='own' for r in rows],int);gradmax=losserr=foldmax=scoremax=0.
for arm in ['C34','R134_spatial','R134_temporal']:
    raw=np.array([r['variants'][arm]['channels'] for r in rows]);mu=raw.mean(0);sd=raw.std(0);constant=np.ptp(raw,axis=0)==0;scale=np.where(constant,1.,np.sqrt(np.mean(sd[~constant]**2)))
    np.testing.assert_array_equal(d['signal_normalization'][arm]['mean'],mu);np.testing.assert_array_equal(d['signal_normalization'][arm]['scale'],scale)
    X=np.c_[g,(raw-mu)/scale];t=np.array(d['training_original'][arm]['theta']);z=X@t[:-1]+t[-1];w=len(y)/(2*np.bincount(y)[y]);res=w*(np.exp(-np.logaddexp(0,-z))-y);grad=np.r_[X.T@res/len(y)+t[:-1],res.mean()];gradmax=max(gradmax,float(abs(grad).max()));loss=float(np.mean(w*np.logaddexp(0,np.where(y==1,-z,z)))+.5*(t[:-1]@t[:-1]));losserr=max(losserr,abs(loss-d['training_original'][arm]['loss']))
    effective=np.array(d['models'][arm]['theta']);foldmax=max(foldmax,float(abs(np.c_[g,raw]@effective[:-1]+effective[-1]-z).max()))
    for c,r in zip(cases,s['records']):
        assert all(c[k]==r[k] for k in ['video','frame','old_id','raw_id','kind'])
        channels=c['ready_C34_channels'] if arm=='C34' else c['new_image_features'][arm]['channels'];geo=(np.array(c['geometry'])-stats['mean'])/stats['scale'];x=np.r_[geo,channels];score=math.fsum(float(a)*float(b) for a,b in zip(x,effective[:-1]))+float(effective[-1]);scoremax=max(scoremax,abs(score-r['scores'][arm]))
assert gradmax<1e-7 and max(losserr,foldmax,scoremax)<1e-12
with (OUT/'final_head_audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,final_fits=3,new_logits=15,gradient_max=gradmax,loss_error=losserr,raw_folding_error=foldmax,independent_scalar_logit_error=scoremax,no_refits=True,ready_vectors_only=True,actual_IDs_evaluated=False),f,indent=2)
print('SCALE_FINAL_HEAD_AUDIT_PASSED',gradmax,scoremax,flush=True)
