"""Prepare metadata protocol. Does not import or execute a neural model."""
from common import *
import numpy as np,subprocess,platform,importlib.metadata as md
source=read(ROOT/'research_stage_localization_20261112/protocol.json')
native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json')
stage=read(ROOT/'research_failure_localization_20261112/stage1.json')
names=['research_stage_localization_20261112/protocol.json','runs/domain_matched_native_20261109/protocol.json','research_failure_localization_20261112/stage1.json','research_stage_localization_20261112/result.json','weights/masa/masa_r50.pth','tracking/masa_r50.py','tracking/gpu_lock.py','runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py']
names.extend(str(x.relative_to(ROOT)).replace('\\','/') for x in OUT.glob('*.py'))
cases=[];videos={}
for cc in source['cases']:
    c=cc['case'];v=c['video'];r=next(r for r in stage['records'] if r['old_id']==c['old_id'] and r['frame']==c['frame'] and r['raw_id']==c['raw_id'])
    ref_frame=cc['sequence'][cc['endpoints'][0]['length']-1][0]
    assert ref_frame==r['reference_frame'] and len(r['reference_hits'])==1
    entry=native['videos'][v];videos[v]=dict(video_path=entry['video_path'],detections=entry['detections'])
    names.extend([entry['video_path'],entry['detections'],f'data/bdd100k/labels/box_track_20/val/{v}.json',f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json'])
    with np.load(ROOT/entry['detections']) as z:
        rid=c['raw_id'];assert int(z['frame_idx'][rid])==c['frame'] and np.array_equal(z['xyxy'][rid].astype(float),c['current_box'])
        ids=np.flatnonzero((z['frame_idx']==c['frame'])&(z['cls']==2)&(z['conf']>=.1))
        objs=objects(v,c['frame']);candidates=[dict(raw_id=int(i),box=z['xyxy'][i].astype(float).tolist(),confidence=float(z['conf'][i]),cls=int(z['cls'][i]),hits=hits(z['xyxy'][i],objs),operational=True) for i in ids]
        desired=dict(raw_id=int(rid),box=z['xyxy'][rid].astype(float).tolist(),confidence=float(z['conf'][rid]),cls=int(z['cls'][rid]),hits=hits(z['xyxy'][rid],objs),operational=bool(rid in ids))
    identity=r['reference_hits'][0]['identity'];refobj=hits(c['reference_box'],objects(v,ref_frame));assert [x['identity'] for x in refobj]==[identity]
    oracle=[o for o in objs if o['identity']==identity];assert len(oracle)<=1
    cases.append(dict(old_id=c['old_id'],label=c['label'],video=v,reference_frame=ref_frame,current_frame=c['frame'],reference_box=c['reference_box'],reference_raw_id=r['history'][-1]['raw_id'],identity=identity,reference_hits=refobj,desired=desired,candidates=candidates,oracle_current_same_identity=oracle,reference_crop=c['reference_crop'],current_crop=c['current_crop']))
expected_frames=read(ROOT/'research_stage_localization_20261112/result.json')['raw_frame_hashes']
env={n:md.version(n) for n in ['numpy','torch','torchvision','opencv-python','psutil']}
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),inputs={n:sha(ROOT/n) for n in sorted(set(names))},environment=dict(python=platform.python_version(),packages=env),cases=cases,videos=videos,expected_raw_frame_hashes=expected_frames,deadline='2026-11-13T07:35:00+03:00',method='Frozen local MASA-R50; all5oldcases, exact stored reference_box, allcurrent raw car class2 confidence>=0.1 without identity preselection. Unique fullframes encoded once; RoIs batched16 in one GPU owner. Desired observed candidate separately retained if outside gate. Independent oracle exact annotation sameidentity box is separately marked and never mixed into operational ranks. Cosine rank/margin only; no threshold/fitting/tracker.',acceptance=['Every referenced source SHA256 unchanged before/after; original rawframes decode sequentially from0 and match prior raw pixel hashes.','Alloperational candidates retained, rawrow/frame/box equality checked; duplicate firstROI produces identical descriptor and all GPU/model tensors explicitly verified.','Independent CPU scalar cosine/rank/margin recomputation agrees1e-6 and alloperational/observed/oracle cases retained.','Availability defined from independent annotation IoU>=0.5, ambiguity retained. ReID output does not assign truth or visible pixel ownership.'],limits='Full-frame MASA receptive field differs from FlyVis small crops; reference box may be filtered/predicted rather than raw detection box. Five reused cases, exploratory diagnostic. No model gain, HOTA, physical correspondence, or core cause established.',state='prepared_waiting_parent_go',batch_size=16))
print('PROTOCOL_PREPARED',len(cases),sum(len(c['candidates']) for c in cases),sha(OUT/'protocol.json'))
