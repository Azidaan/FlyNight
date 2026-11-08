"""Read-only diagnosis of direct residual recoveries, separate identity and box errors."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_singleton_error_audit_20261108.common import *


def main(obs):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from datasets.bdd100k_loader import CLASS_NAMES
    from research_singleton_free_20261108.full_audit import overlaps
    frozen=phase();p=read(OWN_FULL/'protocol.json');arms=read(RUN/'input_manifest.json')['arms'];totals={a:Counter() for a in arms};rows=[];per_video={a:{} for a in arms}
    def label(boxes,categories,objects):
        result={}
        for category in sorted(set(categories)):
            ii=[i for i,c in enumerate(categories) if c==category];gg=[o for o in objects if o[1]==category]
            if not ii or not gg:continue
            iou=overlaps(np.asarray(boxes)[ii],[o[2] for o in gg]);masked=np.where(iou>=.5,iou,0);rr,cc=linear_sum_assignment(-masked)
            for r,c in zip(rr,cc):
                if masked[r,c]>0:result[ii[r]]=gg[c][0]
        return result
    for number,(video,entry) in enumerate(p['videos'].items()):
        truth={};annotations={int(f['frameIndex']):f for f in read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')}
        for frame,anchor in read(OWN_FULL/'runtime_inputs'/(video+'.json'))['anchors'].items():
            f=annotations[anchor['frameIndex']];assert f['name']==anchor['name']
            truth[int(frame)]=[(str(o['id']),o['category'],[o['box2d'][k] for k in ['x1','y1','x2','y2']]) for o in f.get('labels') or [] if o['category'] in CLASS_NAMES.values() and not (o.get('attributes') or {}).get('crowd',False)]
        with np.load(ROOT/entry['detections'],allow_pickle=False) as z:detections={k:z[k] for k in z.files}
        raw_labels={}
        for arm,folder_name in arms.items():
            folder=ROOT/folder_name;local=Counter();before=len(rows);receipt=read(folder/'rollouts'/arm/(video+'.receipt.json'))
            for name,h in receipt['artifacts'].items():assert sha(ROOT/name)==h
            with np.load(folder/'rollouts'/arm/(video+'.npz'),allow_pickle=False) as z:trajectory=z['outputs'];assert int(z['frame_count'])==entry['frames']
            emitted={int(f):{int(r[5]):r for r in trajectory[trajectory[:,0]==f]} for f in np.unique(trajectory[:,0])}
            returns={(e['frame'],e['tid']):e for e in read(folder/'additional'/arm/(video+'.json'))['events']}
            with gzip.open(folder/'rollouts'/arm/(video+'.events.json.gz'),'rt',encoding='utf-8') as f:events=json.load(f)
            for event in events:
                frame=event['frame'];native=set(map(tuple,event['native_first']));added=set(map(tuple,event['actual_first']))-native
                assert native.issubset(set(map(tuple,event['actual_first'])))
                if not added:continue
                ids=np.flatnonzero((detections['frame_idx']==frame)&(detections['conf']>=.25));assert len(ids)==len(event['classes'])
                if frame in truth and frame not in raw_labels:
                    labeled=label(detections['xyxy'][ids],[CLASS_NAMES[int(c)] for c in detections['cls'][ids]],truth[frame]);raw_labels[frame]={int(ids[i]):v for i,v in labeled.items()}
                for i,j in sorted(added):
                    track=event['rows'][i];tid=track['tid'];raw_id=int(ids[j]);assert track['state']==2;local['added_first_pairs']+=1
                    output=emitted.get(frame,{}).get(tid);visible=output is not None;local['emitted_added_pairs' if visible else 'assigned_but_not_emitted']+=1
                    if visible:assert event['assigned'].get(str(tid))==raw_id
                    deleted=any(d['tracked_id']==tid and d['tracked_dropped'] for d in event['duplicates'])
                    if deleted:local['confirmed_duplicate_removal']+=1
                    recent=[]
                    for h in track['history']:
                        if h['frame'] in truth:
                            identity=label([h['box']],[CLASS_NAMES[track['cls']]],truth[h['frame']]).get(0)
                            if identity is not None:recent.append(identity)
                    unique=set(recent);recent_target=next(iter(unique)) if len(unique)==1 else None
                    if len(unique)>1:local['recent_history_identity_conflict']+=1
                    recovery=returns.get((frame,tid));target=recent_target;target_source='recent_history_consensus'
                    if recovery is not None and recovery['prior_identity'] is not None:
                        target=recovery['prior_identity'];target_source='last_known_identity_of_emitted_trajectory'
                        if recent_target is not None and recent_target!=target:local['recent_vs_last_known_conflict']+=1
                    identity=raw_labels.get(frame,{}).get(raw_id);kind='unknown' if target is None or identity is None else ('correct' if identity==target else 'foreign')
                    local['raw_identity_'+kind]+=1
                    item=dict(video=video,arm=arm,frame=frame,tid=tid,row=i,column=j,raw_id=raw_id,cost=event['C'][i][j],age_since_last_observation_frames=frame-track['history'][-1]['frame'],emitted=visible,
                        duplicate_removal=deleted,target_identity=target,target_source=target_source,recent_history_identity=recent_target,current_raw_identity=identity,raw_identity_kind=kind,
                        current_annotated=frame in truth,return_kind=None if recovery is None else recovery['kind'],return_sustained=None if recovery is None else recovery['sustained'])
                    if frame in truth:local['current_annotated_pairs']+=1
                    if identity is not None:
                        object_box=next(o[2] for o in truth[frame] if o[0]==identity);raw_iou=float(overlaps([detections['xyxy'][raw_id]],[object_box])[0,0]);assert raw_iou>=.5-1e-12
                        item['raw_box_iou_to_raw_identity']=raw_iou
                        if visible:
                            updated_iou=float(overlaps([output[1:5]],[object_box])[0,0]);item['updated_box_iou_to_raw_identity']=updated_iou
                            if updated_iou<.5:
                                local['raw_labeled_but_updated_box_below_half_iou']+=1
                                if kind=='correct':local['correct_raw_identity_but_updated_box_below_half_iou']+=1
                    if recovery is not None:local['direct_return_'+recovery['kind']]+=1
                    rows.append(item)
            assert len(rows)-before==local['added_first_pairs'];totals[arm].update(local);per_video[arm][video]=dict(local)
        status('direct_recovery_error_sources',complete=number+1,total=len(p['videos']),guard=obs.read())
    assert all(t['added_first_pairs']==t['raw_identity_correct']+t['raw_identity_foreign']+t['raw_identity_unknown'] for t in totals.values())
    assert not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules);phase()
    write(RUN/'report.json',dict(passed=True,totals={a:dict(v) for a,v in totals.items()},per_video=per_video,rows=rows,definitions=read(RUN/'protocol.json')['definitions'],
        limitations=read(RUN/'protocol.json')['limitations'],phase_sha256=sha(RUN/'phase.json'),new_detector_calls=0,new_core_calls=0,new_tracker_updates=0),new=True)


if __name__=='__main__':main(guard())
