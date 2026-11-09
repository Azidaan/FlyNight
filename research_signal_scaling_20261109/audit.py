"""Independently check all new scale fits and their raw-feature deployment."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
stop=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(600,(stop-datetime.now(stop.tzinfo)).total_seconds()))
import psutil,numpy as np
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(p.read_text(encoding='utf8'))
p=read(OUT/'protocol.json')
for n,h in p['sources'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
rows=read(TRAIN/'features.json')['records'];pred=read(OUT/'predictions.json');groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);gradmax=losserr=prederr=deployerr=0.
for f in pred['folds']:
    tr=np.flatnonzero(groups!=f['video']);te=np.flatnonzero(groups==f['video']);arm,kind=f['model'].split('__');raw=np.array([r['variants'][arm]['channels'] for r in rows]);mu=raw[tr].mean(0);sd=raw[tr].std(0);constant=np.ptp(raw[tr],axis=0)==0;rms=np.sqrt(np.mean(sd[~constant]**2));scale=np.where(constant,1.,sd if kind=='unit' else rms)
    np.testing.assert_array_equal(mu,f['signal_mean']);np.testing.assert_array_equal(scale,f['signal_scale']);assert sorted(set(groups[tr]))==f['training_videos']
    g=(G-f['normalization']['mean'])/f['normalization']['scale'];x=np.c_[g,(raw-mu)/scale];t=np.array(f['fit']['theta']);z=x[tr]@t[:-1]+t[-1];w=len(tr)/(2*np.bincount(y[tr])[y[tr]]);res=w*(np.exp(-np.logaddexp(0,-z))-y[tr]);grad=np.r_[x[tr].T@res/len(tr)+t[:-1],res.mean()];loss=float(np.mean(w*np.logaddexp(0,np.where(y[tr]==1,-z,z)))+.5*(t[:-1]@t[:-1]))
    gradmax=max(gradmax,float(abs(grad).max()));losserr=max(losserr,abs(loss-f['fit']['loss']));actual=x[te]@t[:-1]+t[-1];expected=np.array([pred['records'][i]['scores'][f['model']] for i in te]);prederr=max(prederr,float(abs(actual-expected).max()))
    effective=np.array(f['effective_raw_theta']);deployerr=max(deployerr,float(abs(np.c_[g,raw]@effective[:-1]+effective[-1]-x@t[:-1]-t[-1]).max()))
assert len(pred['folds'])==120 and gradmax<1e-7 and max(losserr,prederr,deployerr)<1e-12
with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,audited_fits=120,gradient_max=gradmax,loss_error=losserr,prediction_error=prederr,raw_deployment_error=deployerr,train_only_statistics_verified=True,no_refits=True,no_core_or_video=True,goal_achieved=False),f,indent=2)
print('SCALE_AUDIT_PASSED',gradmax,deployerr,flush=True)
