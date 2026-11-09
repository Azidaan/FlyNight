"""Independent equations for the120newfits; ready oldmodels are not refit."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(900,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
import psutil,numpy as np
from scipy.special import expit
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(OUT/'protocol.json')
for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
data=read(OUT/'profiles.json');rows=data['records'];old=read(TRAIN/'features.json')['records'];pred=read(OUT/'predictions.json');summary=read(OUT/'summary.json');oldpred=read(TRAIN/'predictions.json')
assert len(rows)==26 and len(data['diagnostic'])==3 and all(x['not_used_for_fit_or_gate'] for x in data['diagnostic'])
groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);norms={f['video']:f['normalization'] for f in oldpred['folds'] if f['model']=='G7'}
max_gradient=max_loss=max_prediction=0.;scores={}
for f in pred['folds']:
    tr=np.flatnonzero(groups!=f['video']);te=np.flatnonzero(groups==f['video']);assert sorted(set(groups[tr]))==f['training_videos'] and f['video'] not in f['training_videos']
    assert f['normalization']==norms[f['video']];g=(G-np.array(f['normalization']['mean']))/np.array(f['normalization']['scale'])
    name=f['model'].removesuffix('_contrast').removesuffix('_scale_only');delta=np.array([r['variants'][name]['contrast'] for r in rows]);raw=np.array([r['variants'][name]['channels'] for r in old]);sd=raw[tr].std(0);ratio=np.divide(delta[tr].std(0),sd,out=np.ones_like(sd),where=sd>0)
    if f['model'].endswith('_scale_only'):assert np.array_equal(ratio,np.array(f['scale_ratio']));features=raw*ratio
    else:assert f['scale_ratio'] is None;features=delta
    X=np.c_[g,features,np.ones(len(rows))];theta=np.array(f['fit']['theta']);weights=len(tr)/(2*np.bincount(y[tr])[y[tr]]);z=X[tr]@theta;penalty=theta.copy();penalty[-1]=0
    gradient=X[tr].T@(weights*(expit(z)-y[tr]))/len(tr)+penalty;loss=float(np.mean(weights*(np.logaddexp(0,z)-y[tr]*z))+.5*(penalty@penalty))
    error=float(abs(gradient).max());assert error<1e-7 and abs(loss-f['fit']['loss'])<1e-12;max_gradient=max(max_gradient,error);max_loss=max(max_loss,abs(loss-f['fit']['loss']))
    actual=np.array([pred['records'][i]['scores'][f['model']] for i in te]);error=float(abs(X[te]@theta-actual).max());assert error<1e-12;max_prediction=max(max_prediction,error)
    scores.setdefault(f['model'],np.full(len(rows),np.nan))[te]=actual
assert len(pred['folds'])==120 and all(np.isfinite(z).all() for z in scores.values())
for key,z in scores.items():
    own=int(((y==1)&(z>0)).sum());foreign=int(((y==0)&(z>0)).sum());m=summary['metrics'][key];assert m['own_accept']==own and m['foreign_accept']==foreign and abs(m['balanced_accuracy']-.5*(own/20+(6-foreign)/6))<1e-15
for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,audited_fits=120,independent_gradient_max=max_gradient,independent_loss_max_error=max_loss,prediction_max_error=max_prediction,heldout_video_exclusion_checked=True,geometry_normalization_reused=True,no_refits=True,no_core_or_tracking=True,goal_achieved=False),f,indent=2)
print('CONTRAST_AUDIT_PASSED',max_gradient,flush=True)
