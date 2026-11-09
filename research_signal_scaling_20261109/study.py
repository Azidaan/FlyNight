"""Two fixed signal-scale treatments on saved correlations; no feature extraction."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
stop=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(1800,(stop-datetime.now(stop.tzinfo)).total_seconds()))
import psutil,numpy as np
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
from tracking.channel_identity_head import fit_ridge_logistic,decision_function
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cast(x):
    if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cast(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x
def save(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
p=read(OUT/'protocol.json')
def verify():
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
verify();assert read(OUT/'preflight.json')['passed']
rows=read(TRAIN/'features.json')['records'];pred=read(TRAIN/'predictions.json');groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);norms={f['video']:f['normalization'] for f in pred['folds'] if f['model']=='G7'};arms=['C34','R134_spatial','R134_temporal']
scores={arm+'__'+kind:np.full(len(rows),np.nan) for arm in arms for kind in ['unit','common']};fits=[]
for video in sorted(set(groups)):
    tr=np.flatnonzero(groups!=video);te=np.flatnonzero(groups==video);assert set(y[tr])=={0,1};stats=norms[video];g=(G-stats['mean'])/stats['scale']
    for arm in arms:
        raw=np.array([r['variants'][arm]['channels'] for r in rows]);mu=raw[tr].mean(0);sd=raw[tr].std(0);constant=np.ptp(raw[tr],axis=0)==0;assert (~constant).any();rms=float(np.sqrt(np.mean(sd[~constant]**2)))
        for kind,scale in [('unit',np.where(constant,1.,sd)),('common',np.where(constant,1.,rms))]:
            assert (scale>0).all();value=(raw-mu)/scale;x=np.c_[g,value];model=fit_ridge_logistic(x[tr],y[tr]);name=arm+'__'+kind;scores[name][te]=decision_function(model,x[te])
            theta=np.array(model['theta']);effective=theta.copy();effective[7:-1]/=scale;effective[-1]-=mu@effective[7:-1]
            error=float(abs(np.c_[g,raw]@effective[:-1]+effective[-1]-x@theta[:-1]-theta[-1]).max());assert error<1e-12
            fits.append(dict(video=video,model=name,fit=model,effective_raw_theta=effective,normalization=stats,signal_mean=mu,signal_scale=scale,constant_channels=constant,training_videos=sorted(set(groups[tr])),folded_prediction_error=error))
    print('SCALE_FOLD',len(fits)//6,'/20',flush=True)
def metric(z):
    own=y==1;yes=z>0;a=int((own&yes).sum());b=int((~own&yes).sum());return dict(own_accept=a,foreign_accept=b,balanced_accuracy=.5*(a/20+(6-b)/6))
assert all(np.isfinite(z).all() for z in scores.values());metrics={n:metric(z) for n,z in scores.items()};baseline=read(TRAIN/'summary.json')['metrics']
def better(a,b):return a['own_accept']>=b['own_accept'] and a['foreign_accept']<=b['foreign_accept'] and a['balanced_accuracy']>b['balanced_accuracy']
gates={arm+'__'+kind:better(metrics[arm+'__'+kind],baseline[arm]) for arm in arms for kind in ['unit','common']};specific={arm:better(metrics[arm+'__unit'],metrics[arm+'__common']) for arm in arms};selection={}
for arm in arms:
    u=gates[arm+'__unit'];c=gates[arm+'__common'];selection[arm]='unit' if u and (not c or specific[arm]) else ('common' if c else None)
verify();save('predictions.json',dict(passed=True,records=[dict(**{k:r[k] for k in ['video','frame','old_id','raw_id','label']},scores={n:z[i] for n,z in scores.items()}) for i,r in enumerate(rows)],folds=fits))
save('summary.json',dict(passed=True,metrics=metrics,ready_baselines=baseline,gates=gates,channel_specific=specific,selection=selection,any_gate=any(gates.values()),fits=len(fits),old_models_not_refit=True,actual_IDs_evaluated=False,goal_achieved=False))
print('SIGNAL_SCALING_RESULT',json.dumps(metrics),json.dumps(selection),flush=True)
