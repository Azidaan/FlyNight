"""Check the exact scale/penalty relation before a new fitting protocol."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(ROOT))
import numpy as np
from tracking.channel_identity_head import objective
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[ROOT/'research_domain_matched_reader_20261109/features.json',ROOT/'runs/domain_matched_native_20261109/deployment.json',ROOT/'tracking/channel_identity_head.py',Path(__file__)]
source={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
rows=read(paths[0])['records'];d=read(paths[1]);G=np.array([r['geometry'] for r in rows]);g=(G-d['geometry_normalization']['mean'])/d['geometry_normalization']['scale'];y=np.array([r['label']=='own' for r in rows],int);sw=len(y)/(2*np.bincount(y)[y]);records=[]
for arm in ['C34','R134_spatial','R134_temporal']:
    x=np.array([r['variants'][arm]['channels'] for r in rows]);mean=x.mean(0);sd=x.std(0);constant=np.ptp(x,axis=0)==0;sd[constant]=1
    assert (sd>0).all();t=np.array(d['models'][arm]['theta']);X=np.c_[g,x];z=X@t[:-1]+t[-1];loss=objective(t,X,y,sw)[0]
    centered=np.c_[g,x-mean];tc=t.copy();tc[-1]+=mean@t[7:-1]
    center_error=float(abs(centered@tc[:-1]+tc[-1]-z).max());center_loss_error=abs(objective(tc,centered,y,sw)[0]-loss)
    normalized=np.c_[g,(x-mean)/sd];tn=t.copy();tn[7:-1]*=sd;tn[-1]+=mean@t[7:-1]
    normalize_error=float(abs(normalized@tn[:-1]+tn[-1]-z).max());expected_delta=.5*np.dot(sd**2-1,t[7:-1]**2);penalty_error=abs(objective(tn,normalized,y,sw)[0]-loss-expected_delta)
    effective=tn.copy();effective[7:-1]/=sd;effective[-1]-=mean@effective[7:-1];deploy_error=float(abs(effective-t).max())
    rms=float(np.sqrt(np.mean((x.std(0))**2)));assert rms>0;trace=float(np.sum(np.var((x-mean)/rms,axis=0)));unit_trace=float(np.sum(np.var((x-mean)/sd,axis=0)))
    # With constant channels, compare trace over the varying channels, not34.
    active_rms=float(np.sqrt(np.mean((x.std(0)[~constant])**2)));active_trace=float(np.sum(np.var((x[:,~constant]-mean[~constant])/active_rms,axis=0)))
    assert max(center_error,center_loss_error,normalize_error,penalty_error,deploy_error)<1e-12
    assert abs(unit_trace-active_trace)<1e-12
    records.append(dict(arm=arm,constant_channels=int(constant.sum()),signal_SD_min=float(x.std(0)[~constant].min()),signal_SD_median=float(np.median(x.std(0)[~constant])),signal_SD_max=float(x.std(0)[~constant].max()),geometry_varying_SD=1.,center_prediction_error=center_error,center_loss_error=center_loss_error,normalized_prediction_error=normalize_error,anisotropic_penalty_identity_error=penalty_error,folded_deployment_error=deploy_error,unit_variance_trace=unit_trace,common_scale_active_variance_trace=active_trace))
for n,h in source.items():assert sha(ROOT/n)==h,n
with (OUT/'math_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,records=records,sources=source,new_fits=0,no_core_or_video=True,
    identity='For x_standard=(x-mu)/sigma and raw coefficient beta=w_standard/sigma, penalty ||w_standard||^2/2 equals sum(sigma^2*beta^2)/2. Mean centering alone is exactly absorbed by the unpenalized intercept. Common RMS scaling matches total variance but not channel-specific penalty.',
    scope='Proves the training objectives differ because of feature scale; does NOT prove this explains any wrong ID, that raw scales are a coding error, or that standardization helps. Statistics are from the old training cohort only.',goal_achieved=False),f,ensure_ascii=False,indent=2)
print('SIGNAL_SCALE_MATH_PASSED',json.dumps(records),flush=True)
