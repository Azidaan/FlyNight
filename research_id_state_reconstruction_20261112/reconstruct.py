from common import *
import gzip,importlib.util
import numpy as np
p=verify();native=ROOT/'runs/domain_matched_native_20261109';np_=read(native/'protocol.json')
spec=importlib.util.spec_from_file_location('frozen_kalman',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/kalman_filter.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);kf=mod.KalmanFilterXYWH()
def measurement(b):
    xywh=np.r_[(b[:2]+b[2:])/2,b[2:]-b[:2]].astype(np.float64)
    xywh[:2]-=xywh[2:]/2;tlwh=xywh.astype(np.float32);tlwh[:2]+=tlwh[2:]/2
    return tlwh
def box(m):return np.r_[m[:2]-m[2:4]/2,m[:2]+m[2:4]/2]
records=[];failures=[];maxmean=0.;maxcov=0.;snapshots=0;outputs=0;states={};costs=[]
for case in p['cases']:
    v=case['video']
    with gzip.open(native/'rollouts/C34'/(v+'.events.json.gz'),'rt',encoding='utf8') as f:logs=json.load(f)
    with np.load(native/'rollouts/C34'/(v+'.npz'),allow_pickle=False) as z:public=z['outputs']
    with np.load(ROOT/np_['videos'][v]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    for tid in case['ids']:
        aa=[a for a in logs['assignments'] if a['tid']==tid];by={a['frame']:a for a in aa};assert len(by)==len(aa)
        saved={e['frame']:r for e in logs['events'] for r in e['rows'] if r['tid']==tid};end=max(max(by),max(saved,default=0));start=min(by);confirmed=False;state=1;mean=None;cov=None;history=[];ownstates=[]
        for f in range(start,end+1):
            observation=by.get(f)
            if mean is not None and confirmed:
                pred=mean.copy()
                if state!=1:pred[6:8]=0
                mm,cc=kf.multi_predict(pred[None],cov[None]);mean,cov=mm[0],cc[0]
            if mean is not None:
                states[(v,tid,f)]=dict(mean=mean.copy(),covariance=cov.copy(),state=state,confirmed=confirmed)
                if f in saved:
                    s=saved[f];em=float(np.max(abs(mean-np.array(s['mean']))));ec=float(np.max(abs(cov-np.array(s['covariance']))));maxmean=max(maxmean,em);maxcov=max(maxcov,ec);snapshots+=1
                    if em>1e-9 or ec>1e-9 or s['state']!=state:failures.append(dict(video=v,tid=tid,frame=f,mean_error=em,covariance_error=ec,state=state,saved_state=s['state']))
            if observation:
                rid=observation['raw_id'];assert rid>=0 and int(raw['frame_idx'][rid])==f;z=measurement(raw['xyxy'][rid]);history.append(dict(frame=f,raw_id=rid,measurement=z.tolist(),pre_state=state,pre_confirmed=confirmed))
                if mean is None:mean,cov=kf.initiate(z)
                else:mean,cov=kf.update(mean,cov,z)
                state=1;confirmed=bool(observation['emitted'])
                if confirmed:
                    rows=public[(public[:,0]==f)&(public[:,5]==tid)];assert len(rows)==1;actual=box(mean).astype(np.float32);outputs+=1
                    if not np.array_equal(actual,rows[0][1:5]):failures.append(dict(video=v,tid=tid,frame=f,public_error=float(np.max(abs(actual-rows[0][1:5])))))
            elif mean is not None:state=2
            ownstates.append(dict(frame=f,mean=mean.tolist(),covariance=cov.tolist(),state=state,confirmed=confirmed))
        records.append(dict(video=v,tid=tid,start=start,end=end,observations=history,post_states=ownstates))
    for birth in case['births']:
        f=birth['frame'];new=next(a for a in logs['assignments'] if a['tid']==birth['new'] and a['frame']==f);rid=new['raw_id'];entries=[]
        for tid in birth['old']:
            pred=states[(v,tid,f)];assigned=next((a for a in logs['assignments'] if a['tid']==tid and a['frame']==f),None)
            entries.append(dict(tid=tid,**{k:cast(x) for k,x in pred.items()},box=box(pred['mean']).tolist(),actual_assigned_raw=None if assigned is None else assigned['raw_id']))
        costs.append(dict(video=v,**birth,new_raw_id=rid,new_box=raw['xyxy'][rid].tolist(),confidence=float(raw['conf'][rid]),older_states=entries,matrix_saved=any(e['frame']==f for e in logs['events'])))
verify();save('reconstruction.json',dict(passed=not failures,failures=failures,mean_max_error=maxmean,covariance_max_error=maxcov,snapshot_checks=snapshots,public_checks=outputs,records=records,birth_states=costs,protocol_sha256=sha(OUT/'protocol.json')))
print('RECONSTRUCTION',not failures,'SNAPSHOTS',snapshots,'PUBLIC',outputs,'ERRORS',maxmean,maxcov,'FAILURES',failures[:5])
assert not failures,'Reconstruction rejected; do not interpret birth costs.'
