"""Sequential full native rollouts; never changes an output ID afterwards."""
import sys
from common import *
mode=sys.argv[1];start('gpu' if mode=='C34' else 'cpu',timeout=14400)
import gzip,time
import numpy as np
import torch
torch.set_num_threads(2)
from native_runtime import Planner,LiveTracker,Detections,FeatureBank
from tracking.offline_association import make_tracker_args
from tracking.spatial_identity_assignment import corrected_cost
from datasets.bdd100k_loader import CLASS_NAMES
p=read(OUT/'protocol.json');phase=read(OUT/'execution_protocol.json');verify(phase)
assert read(OUT/'input_witness_recovery.json')['all_arrays_exact']
if mode!='zero':assert read(OUT/'zero_complete.json')['passed']


def scalabel(video,clock,output):
    labels=[dict(id=str(int(row[4])),category=CLASS_NAMES[int(row[6])],score=float(row[5]),
        box2d=dict(zip(['x1','y1','x2','y2'],map(float,row[:4])))) for row in output]
    return dict(videoName=video,name=clock['name'],frameIndex=clock['frameIndex'],labels=labels)


def rollout(video,arm,backbone=None):
    entry=p['videos'][video];zero=arm.startswith('zero_')
    dest=OUT/'rollouts'/arm;dest.mkdir(parents=True,exist_ok=True);marker=dest/(video+'.receipt.json')
    if marker.exists():
        receipt=read(marker);assert receipt['execution_sha256']==sha(OUT/'execution_protocol.json')
        for name,value in receipt['artifacts'].items():assert sha(ROOT/name)==value,name
        return receipt
    began=time.perf_counter();sources={}
    for name in [entry['detections'],entry['alignment'],entry['raw_array'],entry['baseline']]:
        sources[name]=sha(ROOT/name);assert sources[name]==entry['sources'][name]
    clockpath=LEGACY/'runtime_inputs'/(video+'.json');assert sha(clockpath)==phase['sources'][clockpath.relative_to(ROOT).as_posix()]
    clock=read(clockpath);anchors=clock['anchors']
    with np.load(ROOT/entry['detections'],allow_pickle=False) as z:raw={n:z[n] for n in z.files}
    order=np.argsort(raw['frame_idx'],kind='stable');frames=raw['frame_idx'][order]
    models=None if zero else read(OUT/'deployment.json')
    bank=FeatureBank(video,arm,backbone) if arm not in ['G7','zero_disabled','zero_empty'] else None
    planner=Planner(video,arm,models,bank);alpha=0 if arm=='zero_disabled' else p['alpha']
    tracker=LiveTracker(make_tracker_args(**p['native_config']),planner,alpha=alpha,beta=0,arm=arm)
    outputs=[];rows=[];events=[];assignments=[];zero_frames=0
    try:
        for f in range(entry['frames']):
            lo,hi=np.searchsorted(frames,[f,f+1]);ids=order[lo:hi]
            output=tracker.update(Detections(raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids],ids))
            if len(output):outputs.append(np.column_stack([np.full(len(output),f),np.asarray(output)[:,:7]]).astype(np.float32))
            if str(f) in anchors:rows.append(scalabel(video,anchors[str(f)],output))
            emitted={int(r[4]) for r in output}
            if not zero:
                assignments.extend(dict(frame=f,tid=int(t.track_id),raw_id=int(t.export_raw_id),
                    start_frame=int(t.start_frame),emitted=int(t.track_id) in emitted)
                    for t in tracker.tracked_stracks if t.frame_id==tracker.frame_id)
            snap=planner.last
            if snap is not None:
                assert snap['frame']==f
                actual=tracker.identity_events[-1];assert actual['frame']==f+1
                if zero:
                    assert not actual['votes'] and actual['changed_costs']==0
                    assert actual['matches']==snap['native_first'];zero_frames+=1
                elif snap['proposals']:
                    C=snap['cost'];plan=snap['plan']
                    events.append(dict(frame=f,rows=snap['rows'],high_raw_ids=planner.high.raw_ids.tolist(),
                        proposals=snap['proposals'],C=C.tolist(),
                        corrected=corrected_cost(C,plan['votes'],alpha,0,arm,{}).tolist(),
                        native_first=snap['native_first'],actual_first=actual['matches'],votes=actual['votes'],
                        emitted_ids=sorted(emitted),assigned={str(k):v for k,v in tracker.assigned.items()},duplicates=tracker.dupes))
            # Keep no long-lived per-frame native event history; proposals above retain evidence.
            tracker.identity_events.clear()
            if f%100==0:
                status('zero_parity' if zero else 'native_rollout',arm=arm,video=video,frame=f,total=entry['frames'],events=len(events))
                print('FRAME',arm,video,f,entry['frames'],len(events),flush=True)
        flat=np.concatenate(outputs) if outputs else np.zeros((0,8),np.float32)
        artifacts={}
        if zero:
            with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:
                assert int(z['frame_count'])==entry['frames'] and np.array_equal(flat,z['outputs']),(video,arm,'raw parity')
            assert rows==read(ROOT/entry['baseline']),(video,arm,'baseline JSON parity')
            assert not torch.cuda.is_initialized()
        else:
            rawpath=dest/(video+'.npz');jsonpath=OUT/'outputs'/arm/(video+'.json');eventpath=dest/(video+'.events.json.gz')
            assert not any(x.exists() for x in [rawpath,jsonpath,eventpath])
            with rawpath.open('xb') as fh:np.savez_compressed(fh,outputs=flat,frame_count=entry['frames'])
            write(jsonpath.relative_to(OUT),rows)
            with gzip.open(eventpath,'xt',encoding='utf8') as fh:json.dump(cast(dict(events=events,assignments=assignments)),fh,ensure_ascii=False,allow_nan=False)
            artifacts={x.relative_to(ROOT).as_posix():sha(x) for x in [rawpath,jsonpath,eventpath]}
        for name,value in sources.items():assert sha(ROOT/name)==value,name
        receipt=dict(passed=True,video=video,arm=arm,zero=zero,frames=entry['frames'],
            output_rows=len(flat),seconds=time.perf_counter()-began,active_empty_solver_frames=zero_frames,
            exact_full_raw_and_JSON=zero,execution_sha256=sha(OUT/'execution_protocol.json'),sources=sources,
            artifacts=artifacts,proposal_counts=dict(planner.counts),events=len(events),
            feature_files={} if bank is None else bank.files,new_maps=0 if bank is None else bank.new,
            cached_C8_parity=[] if bank is None else bank.parity,
            core_imported=any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules))
        if arm!='C34':assert not receipt['core_imported']
        write(marker.relative_to(OUT),receipt)
        print('COMPLETE',arm,video,round(receipt['seconds'],2),len(events),dict(planner.counts),flush=True)
        return receipt
    finally:
        if bank is not None:bank.close()


if mode=='zero':
    receipts=[]
    for arm in ['zero_disabled','zero_empty']:
        for video in p['videos']:receipts.append(rollout(video,arm))
    verify(phase)
    write('zero_complete.json',dict(passed=True,videos=20,runs=40,all_full_raw_arrays_exact=True,
        all_annotated_JSON_exact=True,active_empty_solver_frames=sum(r['active_empty_solver_frames'] for r in receipts),
        execution_sha256=sha(OUT/'execution_protocol.json'),receipts=[dict(video=r['video'],arm=r['arm'],seconds=r['seconds']) for r in receipts]))
    status('zero_parity_complete',videos=20,runs=40)
elif mode in p['arms']:
    assert read(OUT/'deployment.json')['execution_sha256']==sha(OUT/'execution_protocol.json')
    if mode=='C34':
        from tracking.gpu_lock import GpuLock
        from tracking.flyvis_adapter import activate_flyvis
        from scripts.z13_target_model import build_arm
        with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
            activate_flyvis(device='cuda');_,_,bb=build_arm('C',torch,None)
            assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12
            weights={k:v.detach().clone() for k,v in bb.net.named_parameters()}
            receipts=[rollout(v,mode,bb) for v in p['videos']]
            assert all(torch.equal(v,weights[k]) for k,v in bb.net.named_parameters())
    else:receipts=[rollout(v,mode) for v in p['videos']]
    verify(phase);write(mode+'_complete.json',dict(passed=True,arm=mode,videos=20,weights_unchanged=True,
        execution_sha256=sha(OUT/'execution_protocol.json'),events=sum(r['events'] for r in receipts)))
    status('arm_complete',arm=mode,videos=20)
else:raise ValueError(mode)
