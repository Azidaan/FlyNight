"""Independent arithmetic on four saved final heads; never fits or runs a model."""
import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
stop=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
guard_entrypoint(timeout=min(600,(stop-datetime.now(stop.tzinfo)).total_seconds()))
import numpy as np, psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[OUT/n for n in ['deployment.json','deployment_screen.json','features.json','deployment_screen_protocol.json']]
paths += [ROOT/'research_domain_matched_reader_20261109'/n for n in ['features.json','predictions.json']]
before={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
protocol=read(OUT/'deployment_screen_protocol.json')
for n,h in protocol['sources'].items(): assert sha(ROOT/n)==h,n
d=read(OUT/'deployment.json'); rows=read(OUT/'features.json')['records']
old=read(ROOT/'research_domain_matched_reader_20261109/features.json')['records']
key=lambda r:tuple(r[k] for k in ['video','frame','old_id','raw_id'])
assert [key(r) for r in rows]==[key(r) for r in old]
y=np.array([r['label']=='own' for r in rows],float); counts=np.bincount(y.astype(int))
assert counts.tolist()==[6,20]
sw=np.array([len(y)/(2*counts[int(v)]) for v in y]); stats=d['geometry_normalization']
g=(np.array([r['geometry'] for r in rows])-stats['mean'])/stats['scale']
checks=[]
for arm in ['C34','R134_spatial','R134_temporal','C34_scale']:
    if arm=='C34_scale':
        raw=np.array([r['variants']['C34']['channels'] for r in old])
        features=raw*np.array(d['scale_ratio'])
    else: features=np.array([r['variants'][arm]['mask_only']['channels'] for r in rows])
    x=np.column_stack([g,features]); model=d['training_original'][arm]; t=np.array(model['theta'])
    z=x@t[:-1]+t[-1]
    # This implements the equation independently, without importing the fitted objective.
    loss=float(np.mean(sw*np.logaddexp(0,np.where(y==1,-z,z)))+np.dot(t[:-1],t[:-1])/2)
    probability=np.exp(-np.logaddexp(0,-z)); residual=sw*(probability-y)
    gradient=np.r_[x.T@residual/len(y)+t[:-1],residual.mean()]
    loss_error=abs(loss-model['loss']); gradient_error=float(abs(gradient).max())
    assert loss_error<1e-12 and gradient_error<1e-7,(arm,loss_error,gradient_error)
    inference=np.column_stack([g,raw]) if arm=='C34_scale' else x
    deployed=np.array(d['models'][arm]['theta']); predicted=inference@deployed[:-1]+deployed[-1]
    prediction_error=float(abs(predicted-z).max()); assert prediction_error<1e-12
    checks.append(dict(arm=arm,loss=loss,loss_error=loss_error,gradient_max=gradient_error,prediction_error=prediction_error))
previous=read(ROOT/'research_domain_matched_reader_20261109/predictions.json')['records']
current=read(OUT/'predictions.json')['records']; assert [key(r) for r in current]==[key(r) for r in previous]
changes=[]
for a,b in zip(previous,current):
    old_accept=a['scores']['C34']>0; new_accept=b['scores']['C34__mask_only']>0
    if old_accept!=new_accept: changes.append(dict(**{k:a[k] for k in ['video','frame','old_id','raw_id','label']},old_accept=old_accept,new_accept=new_accept,old_logit=a['scores']['C34'],new_logit=b['scores']['C34__mask_only']))
for n,h in before.items(): assert sha(ROOT/n)==h,n
out=dict(passed=True,checks=checks,cv_decision_changes=changes,no_refits=True,no_core_or_video=True,
    sources=before,script_sha256=sha(Path(__file__)),registered_after_outputs=True,
    purpose='Additional independent final-loss and inference audit; not a new scientific acceptance criterion.',goal_achieved=False)
with (OUT/'final_head_audit.json').open('x',encoding='utf8') as f: json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print('FINAL_HEAD_AUDIT_PASSED',json.dumps(changes),flush=True)
