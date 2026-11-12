from common import *
import numpy as np
from tracking.foveated_eyes import crop_box
from tracking.z13_readout import HexGeometry
from tracking.z13_region_assignment import regions
from tracking.spatial_identity_assignment import geometric_match
p=verify();rec=read(OUT/'reconstruction_v2.json');assert rec['passed'] and read(OUT/'audit.json')['passed'];native=ROOT/'runs/domain_matched_native_20261109';np_=read(native/'protocol.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.array(g['points'],np.float32),g['eye_size']);archive=read(native/'execution_protocol.json')['sources'];checks={}
for n in ['tracking/foveated_eyes.py','tracking/z13_readout.py','tracking/z13_region_assignment.py','tracking/spatial_identity_assignment.py','runs/z13_local_peak_20261021/geometry.json']:
    h=sha(ROOT/n);checks[n]=dict(current=h,archived=archive.get(n));assert n not in archive or archive[n]==h,n
save('eligibility_protocol.json',dict(registered=now(),posthoc=True,reason='After exact state reconstruction, determine which original pre-readout gate prevented birth detections from being considered by FlyVis.',sources=checks,script_sha256=sha(__file__),fixed='Recompute existing gates literally; no map, head, tracker or threshold change. Stop at original first failing gate.'))
results=[]
for birth in rec['birth_states']:
    v=birth['video'];f=birth['frame']
    with np.load(ROOT/np_['videos'][v]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    ids=np.flatnonzero((raw['frame_idx']==f)&(raw['conf']>=np_['native_config']['track_high_thresh']))
    for row in birth['older_states']:
        tid=row['tid'];r=dict(video=v,frame=f,new=birth['new'],old=tid,state=row['state'],new_raw_id=birth['new_raw_id'],readout_evaluated=False)
        if row['state']!=2:r['first_failing_gate']='not_lost';results.append(r);continue
        old=next(x for x in rec['records'] if x['video']==v and x['tid']==tid);by={x['frame']:x for x in old['post_states']};history=[]
        for o in [x for x in old['observations'] if x['frame']<f][-4:]:
            m=np.array(by[o['frame']]['mean']);b=np.r_[m[:2]-m[2:4]/2,m[:2]+m[2:4]/2];history.append(dict(frame=o['frame'],box=b.tolist(),crop=list(crop_box(m[:4],(1280,720)))))
        if history:history=[history[0]]*max(0,4-len(history))+history
        crop=crop_box(row['mean'][:4],(1280,720));choices,_,_=regions(geom,crop,row['mean'][:2],raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids],2);choices=choices[1:];r.update(history=history,crop=crop,regional_raw_ids=ids[choices].tolist(),regional_count=len(choices))
        if len(choices)!=2:r['first_failing_gate']='regional_candidate_count_not_two';results.append(r);continue
        checked=[dict(raw_id=int(ids[j]),result=geometric_match(row['mean'],row['covariance'],history,crop,raw['xyxy'][ids[j]],g)) for j in choices];eligible=[x['raw_id'] for x in checked if x['result']['valid']];r.update(geometry=checked,eligible_raw_ids=eligible)
        if len(eligible)!=1:r['first_failing_gate']='geometrically_eligible_count_not_one'
        elif eligible[0]!=birth['new_raw_id']:r['first_failing_gate']='birth_detection_not_unique_geometric_candidate'
        else:
            gap=f-history[-1]['frame'];r['gap']=gap;r['first_failing_gate']='outside_observed_training_gap' if gap not in [5,6] else 'requires_full_native_unmatched_check_before_head'
        results.append(r)
save('eligibility.json',dict(passed=True,records=results,warning='Literal original gate audit using reconstructed target states and actual full high detections. No new proposal or permissive gate. Native unmatched test not reconstructed if not reached.'))
print(json.dumps([{k:r[k] for k in ['frame','old','new','first_failing_gate','regional_count','regional_raw_ids','eligible_raw_ids'] if k in r} for r in results]))
