from common import *
import gzip,subprocess,shutil
import numpy as np
source=ROOT/'runs/domain_matched_native_20261109/rollouts/C34/b1ca2e5d-84cf9134.events.json.gz'
with gzip.open(source,'rt',encoding='utf8') as f:all_events=json.load(f)['events']
events=[next(e for e in all_events if e['frame']==frame) for frame in [1159,1160,1165]]
rawfile=ROOT/'runs/z13_val20_matched_20261024/cache/b1ca2e5d-84cf9134/detections/raw_video.npz'
with np.load(rawfile,allow_pickle=False) as z:raw={k:z[k] for k in ['xyxy','conf','cls','frame_idx']}
def native_box(raw_id):
    box=raw['xyxy'][raw_id].astype(np.float32);center=(box[:2]+box[2:])/np.float32(2);size=box[2:]-box[:2];xywh=np.r_[center,size].astype(float);xywh[:2]-=xywh[2:]/2;tlwh=xywh.astype(np.float32);tlwh[2:]+=tlwh[:2];return tlwh.tolist()
frames={}
def slot(frame,box):
    key=str(int(frame));item=frames.setdefault(key,dict(frame=int(frame),boxes=[]));box=list(map(float,box))
    if box not in item['boxes']:item['boxes'].append(box)
    return [key,item['boxes'].index(box)]
compact=[]
for e in events:
    rows=[]
    for r in e['rows']:
        h=r['history'][-1] if r['history'] else None
        assert h is None or h['frame']<e['frame']
        rows.append(dict(tid=r['tid'],cls=r['cls'],state=r['state'],mean=r['mean'],history=r['history'],reference=None if h is None else dict(frame=h['frame'],box=h['box'],slot=slot(h['frame'],h['box']))))
    candidates=[dict(raw_id=int(k),confidence=float(raw['conf'][k]),cls=int(raw['cls'][k]),box=native_box(k),slot=slot(e['frame'],native_box(k))) for k in e['high_raw_ids']]
    compact.append(dict(frame=e['frame'],rows=rows,candidates=candidates,base=e['C'],corrected=e['corrected'],actual_first=e['actual_first'],assigned=e['assigned']))
save('runtime.json',dict(video='b1ca2e5d-84cf9134',events=compact,scope='Savedpreassignmentstates; weightsprobeonly, no lifecycle replay.'))
save('descriptor_plan.json',dict(video='b1ca2e5d-84cf9134',video_path='data/bdd100k/videos/val/b1ca2e5d-84cf9134.mov',frames=frames,strictly_past_references=True,unique_frames=len(frames),unique_rois=sum(len(f['boxes']) for f in frames.values())))
# Evaluation labels never enter runtime or descriptor extraction. Exact prior
# anchor ownership is a persistent-ID expectation, not a label of last pixels.
clockfile=ROOT/'runs/identity_assignment_20261108/full_precision/runtime_inputs/b1ca2e5d-84cf9134.json';labelfile=ROOT/'data/bdd100k/labels/box_track_20/val/b1ca2e5d-84cf9134.json'
anchors=read(clockfile)['anchors'];past=[e for e in all_events if str(e['frame']) in anchors and 1147<=e['frame']<1165]
save('evaluation_plan.json',dict(clock=clockfile.relative_to(ROOT).as_posix(),labels=labelfile.relative_to(ROOT).as_posix(),past_anchored_assignments=[dict(frame=e['frame'],rows=[dict(tid=r['tid'],cls=r['cls']) for r in e['rows']],assigned=e['assigned']) for e in past],detector_file=rawfile.relative_to(ROOT).as_posix(),source_archive=source.relative_to(ROOT).as_posix(),method='ExpectedpersistentID from ALLexactavailable prioranchorassignments in12rawframes; each mustmatchunique sameGTobject, elseunknown. Newborn/absentpastunknown. Currentexactclock only. Never labelunannotated1160. LastdescriptorpixelsnotGT-assumed.'))
(OUT/'before').mkdir(exist_ok=False)
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copyfile(ROOT/n,OUT/'before'/Path(n).name)
save('initial_state.json',dict(started=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),branch=subprocess.check_output(['git','branch','--show-current']).decode().strip(),server_head_verified='502466123b3f80da72d0b2ad2f88682f690bb0e9',prior_science=read(ROOT/'research_identity_holdout_20261112/audit.json')['scientific_artifacts'],status=subprocess.check_output(['git','status','--short','-uno']).decode(),authorization='Userexplicitlyauthorizedsmallassociationweightprobeon2179copywithbaselinecomparisonandrollback; no core/fulltracker. No cleanupauthorized.',inputs={p.relative_to(ROOT).as_posix():sha(p) for p in [source,rawfile,clockfile,labelfile]},C_free_bytes=shutil.disk_usage('C:/').free,E_free_bytes=shutil.disk_usage('E:/').free))
print('INPUTS_PREPARED',len(frames),sum(len(f['boxes']) for f in frames.values()),'REFERENCES',sum(len(e['rows']) for e in compact))
