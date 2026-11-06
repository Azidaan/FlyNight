"""Read-only real-artifact replay and isolated malformed-input counterexamples."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import warnings

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tracking.learned_recovery_admission import AdmissionModel,head,raw_input
from tracking.reliability_head import LogisticHead,ReliabilityFunction,BASE_FEATURES,FLY_FEATURES,base_features

torch.set_num_threads(2)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
sources=['tracking/learned_recovery_admission.py','tracking/reliability_head.py']
hashes={name:sha(ROOT/name) for name in sources}
(OUT/'original').mkdir(exist_ok=True)
for name in sources:
    dest=OUT/'original'/Path(name).name
    if not dest.exists():shutil.copy2(ROOT/name,dest)
    assert sha(dest)==hashes[name]

run=ROOT/'runs/recovery_research_20261027/admission_learning'
registered=json.loads((run/'report.json').read_text(encoding='utf-8'))
v=registered['video_rows']['dev'][0]
features=ROOT/'runs/recovery_research_20261027/appearance_features/dev'/(v['video']+'.npz')
assert sha(features)==registered['source_features_sha256'][features.relative_to(ROOT).as_posix()]
with np.load(features) as z:
    X=z['X'][:64].copy();E=z['E'][:64].copy()
actual=[]
for item in registered['models']:
    cp=run/item['checkpoint'];norm=run/item['normalization']
    assert sha(cp)==item['checkpoint_sha256'] and sha(norm)==item['normalization_sha256']
    model=AdmissionModel(cp,norm,device='cpu')
    p=model.probabilities(X,E)
    ref=np.load(run/f"{item['arm']}_seed{item['seed']}_dev_probabilities.npy",mmap_mode='r')[v['start']:v['start']+len(X)]
    error=float(np.max(np.abs(ref-p)))
    assert error<3e-5 and np.isfinite(p).all()
    assert all(bool(torch.isfinite(t).all()) for t in model.model.state_dict().values())
    assert all(np.isfinite(a).all() for a in model.norm.values())
    assert (model.norm['numeric_std']>0).all() and (model.norm['input_std']>0).all()
    actual.append(dict(arm=item['arm'],seed=item['seed'],rows=len(X),cpu_vs_saved_max_abs=error,
                       checkpoint_sha256=sha(cp),normalization_sha256=sha(norm),valid_contract=True))

actual_reliability=[]
for name in ['head_none.json','head_lamina.json']:
    path=ROOT/'runs/e2e_motion_source'/name
    spec=json.loads(path.read_text(encoding='utf-8'))
    for part,features_list in [('base',BASE_FEATURES),('fly',BASE_FEATURES+FLY_FEATURES)]:
        s=spec[part]
        assert s['features']==list(features_list)
        for key in ['mean','scale','coef']:
            a=np.asarray(s[key]);assert a.shape==(len(features_list),) and np.isfinite(a).all()
        assert (np.asarray(s['scale'])>0).all() and np.isfinite(s['intercept'])
    actual_reliability.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),valid_contract=True))

# Small, synthetic checkpoints; no original checkpoint is modified.
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(13);state=head(10).state_dict()
saved=dict(arm='numeric',input_dim=10,state_dict=state)
norm=dict(numeric_mean=np.zeros(2,np.float32),numeric_std=np.ones(2,np.float32),
          projection=np.arange(16,dtype=np.float32).reshape(2,8)/16,
          input_mean=np.zeros(10,np.float32),input_std=np.ones(10,np.float32))
x=np.array([[1.,2.],[3.,4.]],np.float32);emb=np.arange(8,dtype=np.float32).reshape(2,2,2)+1
def admission_case(label,cs,ns):
    cp=OUT/(label+'.pt');npz=OUT/(label+'.npz');torch.save(cs,cp);np.savez(npz,**ns)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            m=AdmissionModel(cp,npz,device='cpu');p=m.probabilities(x,emb)
            return dict(accepted=True,finite_output=bool(np.isfinite(p).all()),
                        output=[float(z) if np.isfinite(z) else str(z) for z in p])
        except Exception as e:return dict(accepted=False,error=type(e).__name__,message=str(e))
cases={}
for bad in ['nan','overflow']:
    cs=copy.deepcopy(saved);cs['state_dict']['1.weight']=cs['state_dict']['1.weight'].to(torch.float64)
    cs['state_dict']['1.weight'][0,0]=float('nan') if bad=='nan' else 1e300
    cases[bad+'_weight']=admission_case(bad+'_weight',cs,norm)
for key in ['numeric_std','input_std']:
    ns=copy.deepcopy(norm);ns[key]=np.zeros_like(ns[key])
    cases['zero_'+key]=admission_case('zero_'+key,saved,ns)
cs=copy.deepcopy(saved);cs['arm']='numerci'
cases['unknown_arm']=admission_case('unknown_arm',cs,norm)
cases['unknown_arm']['raw_equals_appearance']=bool(np.array_equal(
    raw_input(x,emb,'numerci',norm['numeric_mean'],norm['numeric_std'],norm['projection']),
    raw_input(x,emb,'appearance',norm['numeric_mean'],norm['numeric_std'],norm['projection'])))
cases['unknown_arm']['raw_equals_numeric']=bool(np.array_equal(
    raw_input(x,emb,'numerci',norm['numeric_mean'],norm['numeric_std'],norm['projection']),
    raw_input(x,emb,'numeric',norm['numeric_mean'],norm['numeric_std'],norm['projection'])))
cp=OUT/'valid.pt';npz=OUT/'valid.npz';torch.save(saved,cp);np.savez(npz,**norm)
with torch.device('meta'):
    try:AdmissionModel(cp,npz,device='cpu')
    except Exception as e:cases['ambient_meta_cpu_request']=dict(error=type(e).__name__,message=str(e))
    else:cases['ambient_meta_cpu_request']=dict(accepted=True)

spec=dict(features=list(BASE_FEATURES),mean=np.zeros(8).tolist(),scale=np.ones(8).tolist(),
          coef=np.array([1.,0,0,0,0,0,0,0]).tolist(),intercept=0.)
pred=np.array([[300.,200.,128.,144.]])
reliability={}
bad=copy.deepcopy(spec);bad['features'][0],bad['features'][1]=bad['features'][1],bad['features'][0]
p=ReliabilityFunction(LogisticHead(bad))(pred,None,0)
expected=1/(1+np.exp(-.2))
reliability['permuted_feature_contract']=dict(accepted=True,actual=float(p[0]),expected_by_saved_feature_order=float(expected),difference=float(p[0]-expected))
assert not np.allclose(p,expected)
for name,field,value in [('nan_weight','coef',float('nan')),('zero_scale','scale',0.),('negative_scale','scale',-1.)]:
    bad=copy.deepcopy(spec);bad[field][0]=value
    with np.errstate(all='ignore'):
        m=LogisticHead(bad);p=ReliabilityFunction(m)(pred,None,0)
    reliability[name]=dict(accepted=True,finite_output=bool(np.isfinite(p).all()),output=[float(z) if np.isfinite(z) else str(z) for z in p])
bad=copy.deepcopy(spec);bad['mean']=0.;bad['scale']=1.
p=ReliabilityFunction(LogisticHead(bad))(pred,None,0)
reliability['scalar_normalization_broadcast']=dict(accepted=True,output=p.tolist())
assert all(sha(ROOT/k)==v for k,v in hashes.items()) and not torch.cuda.is_initialized()
report=dict(actual_admission=actual,actual_reliability=actual_reliability,admission_malformed=cases,
            reliability_malformed=reliability,source_sha256=hashes,script_sha256=sha(__file__),
            boundary='Malformed artifacts are isolated synthetic counterexamples; no evidence that the inspected historical artifacts are malformed. CPU replay is engineering validation, not tracker/HOTA benefit.')
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
