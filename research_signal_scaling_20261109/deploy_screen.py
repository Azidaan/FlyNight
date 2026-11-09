"""Fit the three preselected common-scale heads and use ready five-case vectors."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109';NATIVE=ROOT/'runs/domain_matched_native_20261109'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
stop=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(600,(stop-datetime.now(stop.tzinfo)).total_seconds()))
import psutil,numpy as np
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
from tracking.channel_identity_head import fit_ridge_logistic
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cast(x):
    if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cast(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x
def save(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
p=read(OUT/'deployment_screen_protocol.json')
def verify():
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
verify();s=read(OUT/'summary.json');assert s['selection']=={a:'common' for a in ['C34','R134_spatial','R134_temporal']} and read(OUT/'audit.json')['passed']
rows=read(TRAIN/'features.json')['records'];old=read(NATIVE/'deployment.json');stats=old['geometry_normalization'];G=np.array([r['geometry'] for r in rows]);g=(G-stats['mean'])/stats['scale'];y=np.array([r['label']=='own' for r in rows],int)
models={'G7':old['models']['G7']};original={};norm={};fit_receipts={}
for arm in ['C34','R134_spatial','R134_temporal']:
    raw=np.array([r['variants'][arm]['channels'] for r in rows]);mu=raw.mean(0);constant=np.ptp(raw,axis=0)==0;sd=raw.std(0);rms=float(np.sqrt(np.mean(sd[~constant]**2)));scale=np.where(constant,1.,rms);X=np.c_[g,(raw-mu)/scale]
    fit=fit_ridge_logistic(X,y);t=np.array(fit['theta']);effective=t.copy();effective[7:-1]/=scale;effective[-1]-=mu@effective[7:-1]
    error=float(abs(np.c_[g,raw]@effective[:-1]+effective[-1]-X@t[:-1]-t[-1]).max());assert error<1e-12
    original[arm]=fit;models[arm]=dict(fit,theta=effective);norm[arm]=dict(mean=mu,scale=scale,kind='common');fit_receipts[arm]=dict(folded_error=error)
    save('final_model_'+arm+'.json',dict(model=models[arm],training_original=fit,signal_normalization=norm[arm]))
save('deployment.json',dict(passed=True,models=models,training_original=original,geometry_normalization=stats,signal_normalization=norm,folded_errors=fit_receipts,threshold=0,regularization=1,pairs=26,own=20,foreign=6,actual_IDs_evaluated=False))
cases=[]
for c in read(ROOT/'research_identity_training_stability_20261109/comparison.json')['cases']:
    geo=(np.array(c['geometry'])-stats['mean'])/stats['scale'];values={}
    for arm in ['C34','R134_spatial','R134_temporal']:
        channels=c['ready_C34_channels'] if arm=='C34' else c['new_image_features'][arm]['channels'];t=np.array(models[arm]['theta']);z=float(np.r_[geo,channels]@t[:-1]+t[-1]);values[arm]=z
    cases.append(dict(**{k:c[k] for k in ['video','frame','old_id','raw_id','kind']},baseline=c['baseline'],scores=values,ready_feature_vectors_reused=True))
gates={};counts={}
for arm in ['C34','R134_spatial','R134_temporal']:
    old_errors=sum((c['baseline'][arm]>0)!=(c['kind']=='own') for c in cases);new_errors=sum((c['scores'][arm]>0)!=(c['kind']=='own') for c in cases);retained=all((c['scores'][arm]>0)==(c['kind']=='own') for c in cases if (c['baseline'][arm]>0)==(c['kind']=='own'))
    gates[arm]=bool(retained and new_errors<old_errors);counts[arm]=dict(old_errors=old_errors,new_errors=new_errors,previous_correct_retained=retained)
verify();save('deployment_screen.json',dict(passed=True,records=cases,gates=gates,counts=counts,any_gate=any(gates.values()),new_final_fits=3,ready_maps_and_features_reused=True,no_core_or_video=True,actual_IDs_evaluated=False,goal_achieved=False))
print('COMMON_SCALE_FIVECASE_SCREEN',json.dumps(gates),json.dumps(counts),flush=True)
