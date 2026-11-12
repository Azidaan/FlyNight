"""Descriptive annotation-box membership; never labels pixels as segmented objects."""
from common import *
import numpy as np
p=verify();source=ROOT/'research_failure_localization_20261112/stage1.json';stage=read(source)['records'];flow=read(OUT/'correspondence.json');inputs={source.relative_to(ROOT).as_posix():sha(source),Path(__file__).relative_to(ROOT).as_posix():sha(__file__), (OUT/'correspondence.json').relative_to(ROOT).as_posix():sha(OUT/'correspondence.json')}
for c in p['cases']:
    video=c['case']['video']
    for n in [f'data/bdd100k/labels/box_track_20/val/{video}.json',f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json']:inputs[n]=sha(ROOT/n)
save('membership_protocol.json',dict(created=now(),inputs=inputs,method='Count where already saved comparison points fall relative to annotated bounding boxes of old and candidate identities. Descriptive post-experiment audit, no score optimization/intervention, no segmentation or physical-ground-truth claim.'))
records=[]
for c,f in zip(p['cases'],flow['records']):
    r=c['case'];s=next(x for x in stage if x['old_id']==r['old_id'] and x['video']==r['video'] and x['frame']==r['frame']);video=r['video']
    data=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json');by={int(x.get('frameIndex',i)):x for i,x in enumerate(data)};anchor=read(ROOT/f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json')['anchors'][str(r['frame'])];frame=by[anchor['frameIndex']];assert frame['name']==anchor['name']
    oldid=s['reference_hits'][0]['identity'];candidate=s['current_hits'][0]['identity'];old=next(x for x in frame['labels'] if str(x['id'])==oldid);new=next(x for x in frame['labels'] if str(x['id'])==candidate)
    box=lambda x:np.array([x['box2d'][k] for k in ['x1','y1','x2','y2']]);oldbox,newbox=box(old),box(new);inside=lambda pts,bb:((pts>=bb[:2])&(pts<=bb[2:])).all(1);counts=[]
    for name,pts in [('original_C34',f['originals']['C34']),('lk',f['lk']),('dis',f['dis']),('opposite_lk',f['opposite_lk'])]:
        pts=np.array(pts);a=inside(pts,oldbox);b=inside(pts,newbox)
        for support,keep in [('full',np.ones(len(pts),bool)),('common',np.array(f['mask']))]:
            counts.append(dict(mapping=name,support=support,total=int(keep.sum()),old_only=int((a&~b&keep).sum()),candidate_only=int((~a&b&keep).sum()),overlap=int((a&b&keep).sum()),neither=int((~a&~b&keep).sum())))
    records.append(dict(old_id=r['old_id'],old_identity=oldid,candidate_identity=candidate,current_old_GT_box=oldbox,current_candidate_GT_box=newbox,counts=counts,warning='Overlapping instance boxes do not assign pixel ownership; being inside a box does not establish a physical part.'))
for n,h in inputs.items():assert sha(ROOT/n)==h,n
save('box_membership.json',dict(passed=True,records=records,scope='Descriptive exact geometry relative to annotations, not proof of foreground contamination.'))
print(json.dumps(cast(next(r for r in records if r['old_id']==574)),ensure_ascii=False,indent=2))
