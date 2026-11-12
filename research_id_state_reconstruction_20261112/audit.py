from common import *
import gzip
import numpy as np
from ultralytics.trackers.bot_sort import BOTrack
from ultralytics.trackers.utils import matching
p=verify();rec=read(OUT/'reconstruction_v2.json');assert rec['passed'];native=ROOT/'runs/domain_matched_native_20261109';np_=read(native/'protocol.json');maxmean=0.;maxcov=0.;steps=0;births=[];sources={}
for case in p['cases']:
    v=case['video']
    with np.load(ROOT/np_['videos'][v]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    for tid in case['ids']:
        rr=next(x for x in rec['records'] if x['video']==v and x['tid']==tid);obs={x['frame']:x for x in rr['observations']};confirmed=False;state=1;x=None;P=None;last=None
        for expected in rr['post_states']:
            f=expected['frame'];o=obs.get(f)
            if x is not None and confirmed:
                x=x.copy()
                if state!=1:x[6:8]=0
                dims=[x[2],x[3],x[2],x[3]];xn=np.empty(8);Pn=np.zeros((8,8));F=np.array([[1.,1.],[0.,1.]])
                for j in range(4):
                    idx=[j,j+4];xn[idx]=F@x[idx];Pn[np.ix_(idx,idx)]=F@P[np.ix_(idx,idx)]@F.T+np.diag([(dims[j]/20)**2,(dims[j]/160)**2])
                x,P=xn,Pn
            if o:
                rid=o['raw_id'];b=raw['xyxy'][rid];xywh=np.r_[(b[:2]+b[2:])/2,b[2:]-b[:2]];det=BOTrack(np.r_[xywh,0].astype(np.float64),raw['conf'][rid],raw['cls'][rid]);z=det.convert_coords(det.tlwh)
                assert np.array_equal(z,np.array(o['measurement'],np.float32))
                if x is None:
                    x=np.r_[z,np.zeros(4,dtype=z.dtype)];std=np.array([.1*z[2],.1*z[3],.1*z[2],.1*z[3],.0625*z[2],.0625*z[3],.0625*z[2],.0625*z[3]]);P=np.diag(std**2)
                else:
                    dims=[x[2],x[3],x[2],x[3]];xn=np.empty(8);Pn=np.zeros((8,8))
                    for j in range(4):
                        idx=[j,j+4];block=P[np.ix_(idx,idx)].astype(float);R=(.05*dims[j])**2;S=block[0,0]+R;K=block[:,0]/S;xn[idx]=x[idx]+K*(z[j]-x[j]);Pn[np.ix_(idx,idx)]=block-np.outer(K,K)*S
                    x,P=xn,Pn
                confirmed=expected['confirmed'];state=1;last=f
            else:state=3 if f-last>np_['native_config']['track_buffer'] else 2
            em=float(np.max(abs(x-np.array(expected['mean']))));ec=float(np.max(abs(P-np.array(expected['covariance']))));maxmean=max(maxmean,em);maxcov=max(maxcov,ec);steps+=1
            assert em<1e-9 and ec<1e-9 and state==expected['state'],(v,tid,f,em,ec,state,expected['state'])
    for b in [x for x in rec['birth_states'] if x['video']==v]:
        result=dict(video=v,frame=b['frame'],new=b['new'],new_raw_id=b['new_raw_id'],new_confidence=b['confidence'],old=[])
        for old in b['older_states']:
            edges=[]
            for kind,rid in [('birth_detection',b['new_raw_id']),('actual_assigned_detection',old['actual_assigned_raw'])]:
                if rid is None:continue
                bb=raw['xyxy'][rid];xywh=np.r_[(bb[:2]+bb[2:])/2,bb[2:]-bb[:2]];det=BOTrack(np.r_[xywh,0].astype(np.float64),raw['conf'][rid],raw['cls'][rid]);dist=matching.iou_distance([np.array(old['box'])],[det.xyxy]);cost=float(matching.fuse_score(dist,[det])[0,0]);iou=float(1-dist[0,0])
                # Independent float32 geometry including bbox_ioa's epsilon.
                aa=np.array(old['box'],np.float32);bb=det.xyxy.astype(np.float32);dx=max(np.float32(0),min(aa[2],bb[2])-max(aa[0],bb[0]));dy=max(np.float32(0),min(aa[3],bb[3])-max(aa[1],bb[1]));inter=dx*dy;union=(aa[2]-aa[0])*(aa[3]-aa[1])+(bb[2]-bb[0])*(bb[3]-bb[1])-inter;iv=inter/(union+np.float32(1e-7));dv=np.float32(1)-iv;cv=np.float32(1)-(np.float32(1)-dv)*raw['conf'][rid]
                assert abs(cost-float(cv))<1e-7,(cost,cv)
                edges.append(dict(kind=kind,raw_id=rid,confidence=float(raw['conf'][rid]),iou=iou,cost=cost,allowed_at_unchanged_threshold=cost<=.8,independent_cost=float(cv)))
            result['old'].append(dict(tid=old['tid'],state=old['state'],confirmed=old['confirmed'],edges=edges))
        births.append(result)
verify();save('audit.json',dict(passed=True,independent_steps=steps,mean_max_error=maxmean,covariance_max_error=maxcov,births=births,reconstruction_sha256=sha(OUT/'reconstruction_v2.json'),script_sha256=sha(__file__),limitations='Independent per-coordinate Kalman algebra and native detection constructor; same measurements and observed assignment schedule. Pair costs do not reconstruct the missing full competing pool. No tracker object or global update instantiated.'))
print('INDEPENDENT_PASS',steps,maxmean,maxcov);print(json.dumps(births))
