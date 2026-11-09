"""Independent all240fit audit with no training or neural execution."""
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
for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
rows=read(OUT/'features.json')['records'];old=read(TRAIN/'features.json')['records'];pred=read(OUT/'predictions.json');summary=read(OUT/'summary.json');norms={f['video']:f['normalization'] for f in read(TRAIN/'predictions.json')['folds'] if f['model']=='G7'}
groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);maximum=0.;maxprediction=0.
for f in pred['folds']:
    tr=np.flatnonzero(groups!=f['video']);te=np.flatnonzero(groups==f['video']);assert sorted(set(groups[tr]))==f['training_videos'] and f['video'] not in f['training_videos'] and f['normalization']==norms[f['video']]
    name,kind=f['model'].split('__');target='mask_only' if kind.startswith('mask') else 'aligned';value=np.array([r['variants'][name][target]['channels'] for r in rows]);raw=np.array([r['variants'][name]['channels'] for r in old])
    if kind.endswith('scale_only'):
        sd=raw[tr].std(0);ratio=np.divide(value[tr].std(0),sd,out=np.ones_like(sd),where=sd>0);assert np.array_equal(ratio,np.array(f['scale_ratio']));value=raw*ratio
    else:assert f['scale_ratio'] is None
    g=(G-np.array(f['normalization']['mean']))/np.array(f['normalization']['scale']);X=np.c_[g,value,np.ones(len(rows))];theta=np.array(f['fit']['theta']);weights=len(tr)/(2*np.bincount(y[tr])[y[tr]]);z=X[tr]@theta;penalty=theta.copy();penalty[-1]=0
    gradient=X[tr].T@(weights*(expit(z)-y[tr]))/len(tr)+penalty;loss=float(np.mean(weights*(np.logaddexp(0,z)-y[tr]*z))+.5*(penalty@penalty));error=float(abs(gradient).max())
    assert error<1e-7 and abs(loss-f['fit']['loss'])<1e-12;maximum=max(maximum,error)
    score=np.array([pred['records'][i]['scores'][f['model']] for i in te]);error=float(abs(X[te]@theta-score).max());assert error<1e-12;maxprediction=max(maxprediction,error)
assert len(pred['folds'])==240
for key,m in summary['metrics'].items():
    z=np.array([r['scores'][key] for r in pred['records']]);own=int(((y==1)&(z>0)).sum());foreign=int(((y==0)&(z>0)).sum());assert m['own_accept']==own and m['foreign_accept']==foreign and abs(m['balanced_accuracy']-.5*(own/20+(6-foreign)/6))<1e-15
for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,audited_fits=240,independent_gradient_max=maximum,prediction_max_error=maxprediction,video_exclusion_checked=True,no_refits=True,no_core_or_tracker=True,goal_achieved=False),f,indent=2)
print('VISIBLE_GEOMETRY_AUDIT_PASSED',maximum,flush=True)
