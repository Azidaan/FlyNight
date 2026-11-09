"""Recover real runtime-domain identity examples from existing training detections.

No score fitting, image/network processing or metric recomputation. The immutable
native output is a per-frame equality witness, never a newly evaluated baseline.
"""
import os,sys,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(3600,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
mode=sys.argv[1]
if mode=='register':
    origin=read(ROOT/'runs/historical_direct_query_inventory_20261109/protocol.json')
    videos=origin['videos'];assert len(videos)==84
    files=[Path(__file__),ROOT/'research_geometry_consistency_20261109/training_states_complete.json',
        ROOT/'runs/historical_direct_query_inventory_20261109/protocol.json',
        ROOT/'tracking/xywh_bytetrack.py',ROOT/'tracking/offline_association.py',ROOT/'tracking/foveated_eyes.py',
        ROOT/'tracking/spatial_identity_assignment.py',ROOT/'tracking/full_template_correspondence.py',
        ROOT/'tracking/geometric_template_correspondence.py',ROOT/'tracking/observed_spatial_reader.py',
        ROOT/'tracking/z13_readout.py',ROOT/'tracking/z13_region_assignment.py',
        ROOT/'runs/z13_local_peak_20261021/geometry.json',
        ROOT/'.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py',
        ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/stracks.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/kalman_filter.py',
        ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py']
    info={}
    for video,item in videos.items():
        info[video]={k:item[k] for k in ['raw','native','annotation','alignment']}
        for name in info[video].values():assert sha(ROOT/name)==origin['sources'][name],name;files.append(ROOT/name)
    save('protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        authorization='research_identity_synthesis_20261109/authorization.json',save_stop=STOP.isoformat(),videos=info,
        reason='The old91trainingedges contain6geometricallyadmitted own and0admitted foreign. The actual native C34 pilot used admitted proposals. Need directly labeled training examples in that SAME operating domain before any replacement head can be justified.',
        action='Prepare a domain-matched identity verifier using the same fixed reader/ordinary-input controls, only if existing native training observations supply both classes in every held-out-video training fold. Coordinate-reader deployment deferred until this gate.',
        new_information='Missing pre-first-assignment contexts and causal4observation history for the complete runtime-defined pool on84existingfit84trainingvideos. Previous32recovery saved only91olderquery states and does not contain these additional contexts. No new detector/video/eye/core pass, no metric evaluation, no baseline output replacement.',
        selection='Exactly the native pilot: lost car, >=2distinct warm times, gap5or6, original first global row unmatched, exactly2real regional high boxes, exactly1 geometric_match valid, old candidate-eroded support>=3. No GT, future owner, map value or other candidate label selects the eligible edge or pose. Save all otherwise-eligible proposals including old support failures as separate strata.',
        geometry='Record both existing geometric transform and old reader support. Keep minimum3unchanged. Record geometry-aligned visible support as readiness metadata only. No costs, assignment, birth or dedup behavior changed.',
        labels='Only exact real annotation frames for current AND actual last observed reference. Reference must be public on that frame and uniquely mapped to raw detector by exact float32 confidence/class/frame; unique noncrowd carGT IoU>=.5. If another public native ID maps to the same reference GT, mark ambiguous. Current chosen raw box uniqueGT; unknowns stay unknown. No requirement that the OTHER candidate has a known or opposite label.',
        parity='All84existingnativefullpublicfloat32arrays checked frame-by-frame in original order. A mismatch stops before accepting that video. Per-video atomic closed receipts authorize reuse; original output and metric files remain untouched.',
        readiness='Require both own and foreign among eligible supported directly labeled edges, retained after exclusion of each full video. Report class/video counts; this is algebraic trainability only, not adequacy or generalization proof. No sample/threshold relaxation after results.',
        baseline_controls='No-FlyVis control here is the identical geometry/detection/native history. Subsequent learned comparison must use identical domain examples, folds and fitting for geometry, ordinary34channelinputs and FlyVis. No method accepted by this inventory.',
        fixed_config=dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True),
        source_limit='Restored hidden states have exact public-output witnesses and fingerprinted source, but no saved old hidden-state equality witness. This is recorded reconstruction, not a retrospective original-state hash.',
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print('REGISTERED_DOMAIN_INVENTORY',len(info),flush=True);raise SystemExit(0)
assert mode=='run';p=read(OUT/'protocol.json');verify(p)
import numpy as np,torch
torch.set_num_threads(2)
from collections import deque,Counter
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers.utils import matching
from tracking.xywh_bytetrack import XYWHByteTrack
from tracking.offline_association import _DetSlice,make_tracker_args
from tracking.foveated_eyes import crop_box
from tracking.spatial_identity_assignment import geometric_match
from tracking.z13_readout import HexGeometry
from tracking.z13_region_assignment import regions
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.geometric_template_correspondence import geometric_template_plan
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
eye=HexGeometry(np.array(geom['points'],np.float32),geom['eye_size'])
reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720])
axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])@np.array([[13.,6.5],[0.,13.]])
def matches(box,labels):
    result=[];b=np.asarray(box,float)
    for item in labels:
        if item['category']!='car' or (item.get('attributes') or {}).get('crowd',False):continue
        q=np.array([item['box2d'][k] for k in ['x1','y1','x2','y2']])
        inter=np.maximum(np.minimum(b[2:],q[2:])-np.maximum(b[:2],q[:2]),0).prod()
        union=np.maximum(b[2:]-b[:2],0).prod()+np.maximum(q[2:]-q[:2],0).prod()-inter
        if union>0 and inter/union>=.5:result.append(str(item['id']))
    return result
all_records=[];receipts=[]
for video,info in p['videos'].items():
    dest=OUT/'videos'/(video+'.json')
    if dest.exists():
        r=read(dest);assert r['protocol_sha256']==sha(OUT/'protocol.json')
        receipts.append(r);all_records+=r['records'];continue
    with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs'];frames_total=int(z['frame_count']);assert native.dtype==np.float32
    truth=read(ROOT/info['annotation']);alignment=read(ROOT/info['alignment'])['matches'];assert len(truth)==len(alignment)
    annotation={int(a['video_frame_index']):t.get('labels') or [] for t,a in zip(truth,alignment)}
    histories={};searches={};records=[];frame=-1;current_ids=None
    class Export(XYWHByteTrack):
        def _pre_first_associate(self,pool,unconfirmed,img,high):
            super()._pre_first_associate(pool,unconfirmed,img,high)
            for t in pool:
                tid=int(t.track_id)
                if int(t.state)==2:searches.setdefault(tid,[]).append([frame,list(crop_box(tuple(t.mean[:4]),(1280,720)))])
                else:searches[tid]=[]
            if frame not in annotation:return
            candidates=[]
            for i,t in enumerate(pool):
                h=list(histories.get(int(t.track_id),[]))
                if int(t.state)!=2 or int(t.cls)!=2 or not h or len({v['frame'] for v in h})<2:continue
                if frame-h[-1]['frame'] not in (5,6):continue
                candidates.append((i,t,[h[0]]*max(0,4-len(h))+h[-4:]))
            if not candidates:return
            tracks=self.init_track(high,img);cost=self.get_dists(pool,tracks)
            first,unmatched,_=matching.linear_assignment(cost.copy(),thresh=.8);unmatched=set(map(int,unmatched))
            high_ids=current_ids[raw['conf'][current_ids]>=.25]
            first=np.asarray(first).reshape(-1,2)
            for i,t,h in candidates:
                if i not in unmatched:continue
                crop=list(crop_box(tuple(t.mean[:4]),(1280,720)))
                choices,_,_=regions(eye,crop,np.array(t.mean[:2]),high.xyxy,high.conf,high.cls,2);choices=choices[1:]
                if len(choices)!=2:continue
                posed={int(j):geometric_match(t.mean,t.covariance,h,crop,high.xyxy[j],geom) for j in choices}
                valid=[j for j,g in posed.items() if g['valid']]
                if len(valid)!=1:continue
                j=valid[0];g=posed[j];reference=h[-1];rc=np.array(reference['crop']);ct=np.array(crop)
                shifts=axes*np.maximum((rc[2:]-rc[:2])/geom['eye_size'],(ct[2:]-ct[:2])/geom['eye_size'])
                old=relative_template_plan(reader,reference['box'],rc,high.xyxy[j],ct,shifts,clip_to_candidate=True)
                aligned=geometric_template_plan(reader,reference['box'],rc,high.xyxy[j],ct,shifts,g['scale'],g['offset'])
                competitor=[int(pool[a].track_id) for a,b in first if int(b)==j]
                records.append(dict(video=video,frame=frame,old_id=int(t.track_id),raw_id=int(high_ids[j]),
                    reference_frame=reference['frame'],reference_box=reference['box'],reference_crop=reference['crop'],
                    current_box=high.xyxy[j].astype(float).tolist(),current_crop=crop,history=h,
                    sequence=[[x['frame'],x['crop']] for x in h]+searches[int(t.track_id)],
                    mean=t.mean.tolist(),covariance=t.covariance.tolist(),geometry=g,
                    choices=[int(high_ids[k]) for k in choices],native_first_candidate_owners=competitor,
                    original_cost=float(cost[i,j]),old_cells=len(old['cells']),aligned_cells=len(aligned['cells']),
                    runtime_supported=len(old['cells'])>=3,offsets=shifts.tolist(),kind='unknown'))
    BaseTrack.reset_id();tracker=Export(make_tracker_args(**p['fixed_config']))
    order=np.argsort(raw['frame_idx'],kind='stable');frame_indices=raw['frame_idx'][order];nativeframes=native[:,0]
    checked_rows=0
    for frame in range(frames_total):
        lo,hi=np.searchsorted(frame_indices,[frame,frame+1]);current_ids=order[lo:hi]
        output=tracker.update(_DetSlice(raw['xyxy'][current_ids],raw['conf'][current_ids],raw['cls'][current_ids]))
        actual=np.column_stack([np.full(len(output),frame),np.asarray(output)[:,:7]]).astype(np.float32) if len(output) else np.empty((0,8),np.float32)
        expected=native[nativeframes==frame];assert np.array_equal(actual,expected),(video,frame,'native output mismatch')
        checked_rows+=len(actual);public=set(map(int,actual[:,5]))
        for t in tracker.tracked_stracks:
            if t.frame_id!=tracker.frame_id:continue
            tid=int(t.track_id)
            ids=current_ids[(raw['conf'][current_ids].astype(np.float32)==np.float32(t.score))&(raw['cls'][current_ids]==int(t.cls))]
            histories.setdefault(tid,deque(maxlen=4)).append(dict(frame=frame,box=np.asarray(t.xyxy,float).tolist(),
                crop=list(crop_box(tuple(t.mean[:4]),(1280,720))),raw_id=int(ids[0]) if len(ids)==1 else None,public=tid in public))
            searches[tid]=[]
    # No GT has been used by the query, candidate, transformation or cell selection.
    rawframe_cache={};owner_cache={}
    for row in records:
        ref=row['history'][-1];rf=ref['frame'];rid=ref['raw_id'];reason=None
        if not ref['public']:reason='reference_not_public'
        elif rf not in annotation:reason='reference_not_annotation_frame'
        elif rid is None:reason='reference_raw_not_unique'
        else:
            rh=matches(raw['xyxy'][rid],annotation[rf]);ch=matches(row['current_box'],annotation[row['frame']])
            if len(rh)!=1:reason='reference_GT_not_unique'
            elif len(ch)!=1:reason='current_GT_not_unique'
            else:
                if rf not in owner_cache:
                    owners={};ids=rawframe_cache.setdefault(rf,np.flatnonzero(raw['frame_idx']==rf))
                    for track in native[nativeframes==rf]:
                        source=ids[(raw['conf'][ids].astype(np.float32)==np.float32(track[6]))&(raw['cls'][ids]==int(track[7]))]
                        if len(source)!=1:continue
                        hit=matches(raw['xyxy'][source[0]],annotation[rf])
                        if len(hit)==1:owners.setdefault(hit[0],[]).append(int(track[5]))
                    owner_cache[rf]=owners
                owners=owner_cache[rf].get(rh[0],[])
                if owners!=[row['old_id']]:reason='reference_has_multiple_public_owners'
                else:row.update(kind='own' if rh[0]==ch[0] else 'foreign',reference_identity=rh[0],current_identity=ch[0])
        row['unknown_reason']=reason
    receipt=dict(passed=True,video=video,frames=frames_total,exact_public_rows=checked_rows,
        all_public_outputs_exact=True,protocol_sha256=sha(OUT/'protocol.json'),records=records,
        sources={name:sha(ROOT/name) for name in info.values()})
    save(dest.relative_to(OUT),receipt);receipts.append(receipt);all_records+=records
    counts=Counter(r['kind'] for r in all_records if r['runtime_supported'])
    print('DOMAIN',len(receipts),'/84',video,'queries',len(all_records),'supported_known',dict(counts),flush=True)
known=[r for r in all_records if r['runtime_supported'] and r['kind']!='unknown']
videos=sorted({r['video'] for r in known})
folds={v:sorted({r['kind'] for r in known if r['video']!=v}) for v in videos}
ready=bool(folds) and all(c==['foreign','own'] for c in folds.values())
verify(p);assert not torch.cuda.is_initialized() and 'flyvis' not in sys.modules
save('results.json',dict(passed=True,records=all_records,counts=dict(Counter(r['kind'] for r in all_records if r['runtime_supported'])),
    videos=84,eligible_proposals=len(all_records),runtime_supported=sum(r['runtime_supported'] for r in all_records),
    known_videos=len(videos),own_videos=len({r['video'] for r in known if r['kind']=='own'}),foreign_videos=len({r['video'] for r in known if r['kind']=='foreign'}),
    held_out_training_classes=folds,readiness=ready,all_native_outputs_exact=True,
    exact_frames=sum(r['frames'] for r in receipts),exact_rows=sum(r['exact_public_rows'] for r in receipts),
    receipts={str((OUT/'videos'/(v+'.json')).relative_to(ROOT).as_posix()):sha(OUT/'videos'/(v+'.json')) for v in p['videos']},
    no_detector_eye_core_fit_score_or_metric_calls=True))
print('DOMAIN_COMPLETE',ready,'records',len(all_records),flush=True)
