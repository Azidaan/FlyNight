"""Recover missing pre-assignment means/covariances, proving archived output parity."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(1800,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
mode=sys.argv[1]
if mode=='register':
    old=read(ROOT/'runs/full_template_identity_20261109/protocol.json')
    edges=old['edges'];videos=sorted({e['video'] for e in edges});assert len(edges)==91 and len(videos)==32
    files=[Path(__file__),ROOT/'tracking/xywh_bytetrack.py',ROOT/'tracking/offline_association.py',ROOT/'tracking/spatial_identity_assignment.py',
        ROOT/'tracking/foveated_eyes.py',ROOT/'runs/full_template_identity_20261109/protocol.json',OUT/'coordinate_report.json',
        ROOT/'.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py',
        ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/stracks.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/kalman_filter.py',
        ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py']
    historical=read(ROOT/'runs/historical_direct_query_inventory_20261109/protocol.json')
    for video in videos:
        for name in [f'runs/train_cache/{video}/detections/raw_video.npz',f'runs/fly_signatures/{video}/signatures.npz']:
            assert sha(ROOT/name)==historical['sources'][name],name
            files.append(ROOT/name)
    save('training_state_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in files},videos=videos,edges=edges,
        reason='No mean/covariance in91trainingedge records; neither existing20valtrace nor26trainingtrace covers these32fit84videos. A new coordinate-aligned comparison cannot be trained consistently without the missing states.',
        operation='CPU association-state reconstruction only up to last needed query per video, over unchanged ready detection arrays. Export missing pre-first-assignment mean/covariance at exact archived oldID/frame. No detector, retinal, image, FlyVis, metric or alternative tracking trajectory computation.',
        reuse='Archived full outputs remain authoritative; do not save a new competing baseline or recompute any metric. Compare every returned frame array exactly through the needed prefix; mismatch stops immediately. Existing restored per-video states are hash-checked and reused.',
        checks='Exact public float32 outputs in original order; pre-assignment crop equals archived input crop; last reference box equals saved last actual observation; candidate cost equals archived original cost within2e-7; input files unchanged. Recovery is not proof of original hidden-state equality absent a stored hidden-state witness.',
        downstream='Compute geometric validity and common visible support for all91trainingedges without consulting labels for pose or cell selection. Attach labels only to readiness counts; no new score/head fitting here.',
        config=dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True),deadline=STOP.isoformat()))
    print('REGISTERED_STATE_RECOVERY',len(videos),len(edges),flush=True);raise SystemExit(0)
assert mode=='run';p=read(OUT/'training_state_protocol.json');verify(p)
import numpy as np
import torch
torch.set_num_threads(2)
from ultralytics.trackers.basetrack import BaseTrack
from tracking.xywh_bytetrack import XYWHByteTrack
from tracking.offline_association import _DetSlice,make_tracker_args
from tracking.foveated_eyes import crop_box
from tracking.spatial_identity_assignment import geometric_match
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
receipts=[];all_edges=[]
for video in p['videos']:
    dest=OUT/'training_states'/(video+'.json')
    if dest.exists():
        receipt=read(dest);assert receipt['protocol_sha256']==sha(OUT/'training_state_protocol.json')
        receipts.append(receipt);all_edges+=receipt['edges'];continue
    edges=[e for e in p['edges'] if e['video']==video];stop=max(e['frame'] for e in edges)
    queries={}
    for e in edges:queries.setdefault(e['frame'],set()).add(e['old'])
    rawpath=ROOT/f'runs/train_cache/{video}/detections/raw_video.npz'
    nativepath=ROOT/f'runs/fly_signatures/{video}/signatures.npz'
    with np.load(rawpath,allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    with np.load(nativepath,allow_pickle=False) as z:native=z['outputs'];assert native.dtype==np.float32
    frame=-1;states={};observations={};current_ids=None
    class Export(XYWHByteTrack):
        def _pre_first_associate(self,pool,unconfirmed,img,high):
            super()._pre_first_associate(pool,unconfirmed,img,high)
            if frame not in queries:return
            dists=self.get_dists(pool,high)
            highids=current_ids[raw['conf'][current_ids]>=.25]
            assert len(highids)==len(high)
            for i,t in enumerate(pool):
                tid=int(t.track_id)
                if tid in queries[frame]:states[(frame,tid)]=dict(mean=t.mean.tolist(),covariance=t.covariance.tolist(),
                    state=int(t.state),current_crop=list(crop_box(tuple(t.mean[:4]),(1280,720))),
                    high_raw_ids=highids.tolist(),cost=dists[i].astype(float).tolist())
    BaseTrack.reset_id();tracker=Export(make_tracker_args(**p['config']))
    order=np.argsort(raw['frame_idx'],kind='stable');frames=raw['frame_idx'][order]
    nativeframes=native[:,0];checked_rows=0
    for frame in range(stop+1):
        lo,hi=np.searchsorted(frames,[frame,frame+1]);current_ids=order[lo:hi]
        output=tracker.update(_DetSlice(raw['xyxy'][current_ids],raw['conf'][current_ids],raw['cls'][current_ids]))
        actual=np.column_stack([np.full(len(output),frame),np.asarray(output)[:,:7]]).astype(np.float32) if len(output) else np.empty((0,8),np.float32)
        expected=native[nativeframes==frame]
        assert np.array_equal(actual,expected),(video,frame,'archived public output mismatch')
        checked_rows+=len(actual)
        for t in tracker.tracked_stracks:
            if t.frame_id==tracker.frame_id:
                key=(frame,int(t.track_id))
                if any(e['old']==key[1] and e['sequence'][3][0]==frame for e in edges):observations[key]=np.asarray(t.xyxy,float).tolist()
    results=[]
    for e in edges:
        state=states[(e['frame'],e['old'])]
        assert state['state']==2 and state['current_crop']==e['sequence'][-1][1]
        np.testing.assert_array_equal(observations[(e['sequence'][3][0],e['old'])],e['reference_box'])
        column=state['high_raw_ids'].index(e['raw_id']);cost=state['cost'][column]
        assert abs(cost-e['original_cost'])<=2e-7,(video,e['uid'],cost,e['original_cost'])
        history=[dict(box=e['reference_box'],crop=e['sequence'][3][1])]
        g=geometric_match(state['mean'],state['covariance'],history,state['current_crop'],e['current_box'],geom)
        results.append(dict(video=video,uid=e['uid'],raw_id=e['raw_id'],frame=e['frame'],old_id=e['old'],
            mean=state['mean'],covariance=state['covariance'],current_crop=state['current_crop'],
            original_cost=cost,geometry=g,label=e['label']))
    receipt=dict(passed=True,video=video,prefix_frames=stop+1,exact_public_rows=checked_rows,
        protocol_sha256=sha(OUT/'training_state_protocol.json'),sources={q.relative_to(ROOT).as_posix():sha(q) for q in [rawpath,nativepath]},
        edges=results,hidden_state_limit='Reconstructed with present fingerprinted source and exact public/crop/reference/cost witnesses; original hidden means/covariances were not saved for direct equality.')
    save(dest.relative_to(OUT),receipt);receipts.append(receipt);all_edges+=results
    print('STATE_RECOVERY',video,len(receipts),len(all_edges),'exactframes',stop+1,flush=True)
verify(p);assert len(all_edges)==91 and len(receipts)==32
assert not torch.cuda.is_initialized() and not any(x=='flyvis' or x.startswith('flyvis.') for x in sys.modules)
save('training_states_complete.json',dict(passed=True,videos=32,edges=91,all_prefix_public_outputs_exact=True,
    exact_frames=sum(r['prefix_frames'] for r in receipts),exact_rows=sum(r['exact_public_rows'] for r in receipts),
    geometry_valid=sum(r['geometry']['valid'] for r in all_edges),geometry_valid_own=sum(r['geometry']['valid'] and r['label']=='own' for r in all_edges),
    geometry_valid_foreign=sum(r['geometry']['valid'] and r['label']=='foreign' for r in all_edges),
    reconstructed_states={str((OUT/'training_states'/(v+'.json')).relative_to(ROOT).as_posix()):sha(OUT/'training_states'/(v+'.json')) for v in p['videos']},
    no_detector_eye_network_metric_calls=True,hidden_state_limit='Exact stored hidden-state witnesses unavailable; same observable output/crop/reference/cost contracts were verified.'))
print('COMPLETE_STATES',len(all_edges),flush=True)
