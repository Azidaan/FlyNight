"""Independent objective, training-only scale and held-out score audit; no fitting."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
stop=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(600,(stop-datetime.now(stop.tzinfo)).total_seconds()))
import numpy as np,psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(p.read_text(encoding='utf8'))
p=read(OUT/'protocol.json')
for n,h in p['sources'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
rows=read(OUT/'features.json')['records'];old=read(TRAIN/'features.json')['records'];pred=read(OUT/'predictions.json');groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);gradient_max=loss_error=prediction_error=0.
for fold in pred['folds']:
    tr=np.flatnonzero(groups!=fold['video']);te=np.flatnonzero(groups==fold['video']);arm,kind=fold['model'].split('__');assert sorted(set(groups[tr]))==fold['training_videos']
    norm=fold['normalization'];g=(G-norm['mean'])/norm['scale'];new=np.array([r['variants'][arm]['channels'] for r in rows]);full=np.array([r['variants'][arm]['channels'] for r in old])
    if kind=='scale_only':
        sd=full[tr].std(0);target=np.sqrt((new[tr].std(0).reshape(3,34)**2).sum(0));ratio=np.divide(target,sd,out=np.ones_like(sd),where=sd>0);np.testing.assert_array_equal(ratio,fold['scale_ratio']);value=full*ratio
    elif kind=='mixed_parts':value=np.array([r['variants'][arm]['mixed']['channels'] for r in rows])
    else:value=new
    x=np.c_[g,value];t=np.array(fold['fit']['theta']);z=x[tr]@t[:-1]+t[-1];w=len(tr)/(2*np.bincount(y[tr])[y[tr]]);res=w*(np.exp(-np.logaddexp(0,-z))-y[tr]);gradient=np.r_[x[tr].T@res/len(tr)+t[:-1],res.mean()]
    loss=float(np.mean(w*np.logaddexp(0,np.where(y[tr]==1,-z,z)))+.5*(t[:-1]@t[:-1]));gradient_max=max(gradient_max,float(abs(gradient).max()));loss_error=max(loss_error,abs(loss-fold['fit']['loss']))
    actual=x[te]@t[:-1]+t[-1];expected=np.array([pred['records'][i]['scores'][fold['model']] for i in te]);prediction_error=max(prediction_error,float(abs(actual-expected).max()))
assert len(pred['folds'])==180 and gradient_max<1e-7 and loss_error<1e-12 and prediction_error<1e-12
with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,audited_fits=180,independent_gradient_max=gradient_max,loss_error=loss_error,prediction_error=prediction_error,training_only_scaling=True,no_refits=True,no_core_or_tracker=True,goal_achieved=False),f,indent=2)
print('LOCAL_PART_AUDIT_PASSED',gradient_max,flush=True)
