"""Evaluator-only return, takeover, persistence and duplicate metrics, frozen before rollout."""
import os,sys,bisect
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_identity_assignment_run_20261108.common import *


def return_events(outputs,truth,match_by_frame,duplicate_extras):
    """Pure counts; identity matching is supplied by a separate evaluator."""
    last_frame={};last_known={};events=[];present={};target_frames=defaultdict(list);totals=Counter()
    for frame,objects in sorted(truth.items()):
        for identity,_,_ in objects:target_frames[identity].append(frame)
    for frame,rows in sorted(outputs.items()):
        ids=[int(r[4]) for r in rows];assert len(ids)==len(set(ids))
        present[frame]=set(ids)
        for tid in ids:
            identity=match_by_frame.get(frame,{}).get(tid)
            if tid in last_frame and frame>last_frame[tid]+1:
                target=last_known.get(tid);kind='unknown'
                if target is not None and identity is not None:kind='correct' if target==identity else 'wrong'
                events.append(dict(frame=frame,tid=tid,gap=frame-last_frame[tid]-1,prior_identity=target,identity=identity,kind=kind))
                totals['reactivations']+=1;totals[kind+'_returns']+=1
            last_frame[tid]=frame
            if identity is not None:last_known[tid]=identity
    for event in events:
        if event['kind']!='correct':event['sustained']='not_correct_return';continue
        frame=event['frame'];target=event['prior_identity'];future=target_frames[target];i=bisect.bisect_right(future,frame);future=future[i:i+2]
        if len(future)<2:kind='censored_insufficient_future'
        elif any(event['tid'] not in present.get(f,set()) for f in future):kind='failed'
        else:
            identities=[match_by_frame.get(f,{}).get(event['tid']) for f in future]
            if any(v is not None and v!=target for v in identities):kind='failed'
            elif any(v is None for v in identities):kind='censored_unknown'
            else:kind='success'
        event['sustained']=kind;event['future_frames']=future;totals['sustained_'+kind]+=1
    totals['duplicate_extra_IDs']=sum(duplicate_extras.values());totals['annotated_frames_with_duplicates']=sum(v>0 for v in duplicate_extras.values())
    return dict(totals=dict(totals),events=events,duplicate_extras={str(k):v for k,v in duplicate_extras.items() if v})


def measure(video,arm,p):
    import numpy as np
    from datasets.bdd100k_loader import CLASS_NAMES
    from scripts.association_log import label,iou
    from research_identity_assignment_run_20261108.full_precision_manager import FULL,raw_path
    entry=p['videos'][video];clock=read(FULL/'runtime_inputs'/(video+'.json'))['anchors'];gtpath=ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json'
    gt={int(f['frameIndex']):f for f in read(gtpath)};truth={}
    for frame,anchor in clock.items():
        source=gt[anchor['frameIndex']];assert source['name']==anchor['name'];objects=[]
        for obj in source.get('labels') or []:
            if obj['category'] not in CLASS_NAMES.values() or (obj.get('attributes') or {}).get('crowd',False):continue
            b=obj['box2d'];objects.append((str(obj['id']),obj['category'],np.array([b[k] for k in ['x1','y1','x2','y2']],float)))
        truth[int(frame)]=objects
    rawpath=raw_path(p,arm,video)
    with np.load(rawpath,allow_pickle=False) as z:raw=z['outputs'];assert int(z['frame_count'])==entry['frames']
    outputs={int(f):raw[raw[:,0]==f,1:] for f in np.unique(raw[:,0])};matches={};duplicate_extras={}
    for frame,objects in truth.items():
        rows=outputs.get(frame,np.empty((0,7)));classes=[CLASS_NAMES[int(r[6])] for r in rows]
        identities=label(rows[:,:4],classes,objects);matches[frame]={int(r[4]):identity for r,identity in zip(rows,identities)}
        assigned=[]
        for row,category in zip(rows,classes):
            candidates=[o for o in objects if o[1]==category]
            if not candidates:continue
            score=iou(row[:4],[o[2] for o in candidates])
            j=int(np.argmax(score))
            if score[j]>=.5:assigned.append((candidates[j][0],int(row[4])))
        groups=defaultdict(set)
        for identity,tid in assigned:groups[identity].add(tid)
        duplicate_extras[frame]=sum(max(0,len(ids)-1) for ids in groups.values())
    result=return_events(outputs,truth,matches,duplicate_extras)
    result.update(passed=True,video=video,arm=arm,raw_sha256=sha(rawpath),gt_sha256=sha(gtpath),
        clock_sha256=sha(FULL/'runtime_inputs'/(video+'.json')),definitions_sha256=sha(RUN/'full_execution_phase.json'))
    return result


def main(obs):
    verify_phase('full_additional_phase.json')
    from research_identity_assignment_run_20261108.full_precision_manager import FULL
    from research_identity_assignment_run_20261108.full_precision_manager import raw_path
    p=read(FULL/'protocol.json');assert read(FULL/'rollouts_complete.json')['passed'];summary={};reused={}
    for arm in p['parameters']:
        totals=Counter();files={}
        for video in p['videos']:
            path=FULL/'additional'/arm/(video+'.json')
            if path.exists():result=read(path)
            else:
                signature=(video,sha(raw_path(p,arm,video)))
                if signature in reused:result=dict(reused[signature],arm=arm,reused_from_identical_raw_output=True)
                else:result=measure(video,arm,p);reused[signature]=result
                write(path,result,new=True)
            totals.update(result['totals']);files[path.relative_to(ROOT).as_posix()]=sha(path)
        summary[arm]=dict(totals=dict(totals),sources=files)
    verify_phase('full_additional_phase.json');write(FULL/'additional_report.json',dict(passed=True,arms=summary,
        scope='Sparse annotated return identities; unknown/censored counts explicit. No labels used by tracker.'),new=True)


if __name__=='__main__':main(guard())
