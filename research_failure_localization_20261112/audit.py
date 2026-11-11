"""Independent direct-label and saved-result consistency checks."""
from common import *
import numpy as np
from collections import Counter
p=verify();s1=read(OUT/'stage1.json');s2=read(OUT/'stage2.json');s3=read(OUT/'stage3.json');checked=0
native=read(NATIVE/'protocol.json')
for v in sorted({r['video'] for r in p['selected']}):
    gt=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');clock=read(LEGACY/'runtime_inputs'/(v+'.json'));indexed={int(x.get('frameIndex',i)):x for i,x in enumerate(gt)}
    with np.load(ROOT/native['videos'][v]['detections'],allow_pickle=False) as z:boxes=z['xyxy'];frames=z['frame_idx']
    for r in [r for r in p['selected'] if r['video']==v]:
        ids=[]
        for fr,rid in [(r['reference_frame'],r['reference_raw_id']),(r['frame'],r['raw_id'])]:
            anchor=clock['anchors'][str(fr)];a=indexed[anchor['frameIndex']];assert a['name']==anchor['name'] and int(frames[rid])==fr
            objs=[o for o in a['labels'] if o['category']=='car' and not (o.get('attributes') or {}).get('crowd',False)]
            b=np.array([[o['box2d'][k] for k in ['x1','y1','x2','y2']] for o in objs]);x=np.array(boxes[rid],float)
            wh=np.maximum(0,np.minimum(b[:,2:],x[2:])-np.maximum(b[:,:2],x[:2]));inter=wh[:,0]*wh[:,1]
            area=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]);iou=inter/(area+(x[2]-x[0])*(x[3]-x[1])-inter)
            candidates=np.flatnonzero(iou>=.5);assert len(candidates)==1;ids.append(str(objs[candidates[0]]['id']))
        assert ('own' if ids[0]==ids[1] else 'foreign')==r['kind'];checked+=1
assert checked==len(s1['records'])==len(s2['records'])==29
assert s2['independent_objective_max_error']<1e-9
assert s3['max_saved_score_error']<1e-12 and len(s3['records'])==5
assert all(r['original_score']==r['identity_control_score'] for r in s3['records'])
assert not s3['physical_intervention_completed']
assert all(not r['causal_intervention_executed'] for r in s3['records'])
assert p['fixed']==dict(head_threshold=0,alpha=.2,beta=0,match_thresh=.8,new_maps=0,new_fits=0,tracker_calls=0)
for name in ['stage2_execution.json','stage3_readiness.json']:
    for n,h in read(OUT/name)['sources'].items():assert sha(OUT/n)==h,(name,n)
for name,script in [('stage3_execution.json','inspect_frames.py'),('boundary_execution.json','boundary_check.py')]:assert read(OUT/name)['source']==sha(OUT/script)
for n in ['stage1.json','stage2.json','stage3.json','ownership.json','boundary_check.json']:
    assert read(OUT/n)['script_sha256']==sha(OUT/(n.replace('.json','.py')))
receipt=dict(passed=True,checked=now(),independent_endpoint_labels=checked,assignment_identity_controls=29,
 independent_assignment_objective_max_error=s2['independent_objective_max_error'],five_saved_map_identity_controls=True,
 all_original_inputs_unchanged=True,thresholds_unchanged=True,full_tracker_calls=0,new_maps=0,new_fits=0,
 stage3_physical_intervention_completed=False,all_root_causes_established=False,
 limitations='Independent label arithmetic shares original annotations and frame clock. LAP and SciPy verify one-step assignment, not public lifecycle. Stage1 source digest was recorded on completion, not falsely claimed as a pre-execution script snapshot.',
 results={n:sha(OUT/n) for n in ['protocol.json','stage1.json','stage2.json','stage3.json','ownership.json','boundary_check.json','frame_manifest.json']})
save('audit.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
