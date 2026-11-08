"""Evaluator-only fit diagnosis; sparse actual anchors, no interpolation."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from research_native_confirm_policy_20261108.common import *


def native_labels(video,outputs):
    import numpy as np
    from scripts.association_log import label
    from datasets.bdd100k_loader import CLASS_NAMES
    annotations=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json');aligned=read(ROOT/f'runs/train_cache/{video}/alignment.json')['matches'];assert len(annotations)==len(aligned)
    tracklets=defaultdict(list);truth_frames=set()
    for frame,a in zip(annotations,aligned):
        f=int(a['video_frame_index']);truth_frames.add(f);objs=[]
        for l in frame.get('labels') or []:
            if l['category'] not in CLASS_NAMES.values() or (l.get('attributes') or {}).get('crowd',False):continue
            b=l['box2d'];objs.append((str(l['id']),l['category'],np.array([b[k] for k in ['x1','y1','x2','y2']])))
        out=outputs[outputs[:,0]==f];ids=label(out[:,1:5],[CLASS_NAMES[int(c)] for c in out[:,7]],objs)
        for row,identity in zip(out,ids):tracklets[int(row[5])].append((f,identity))
    return dict(tracklets),truth_frames


def edge_label(e,frame,labels):
    # A following annotation identifies only the confirmed new fragment, never
    # supplies a runtime input or an interpolated identity for its earlier box.
    last=e['row_data']['history'][-1]['frame'];old=[(f,g) for f,g in labels.get(e['old_tid'],[]) if last-12<=f<=last and g is not None]
    new=[(f,g) for f,g in labels.get(e['new_tid'],[]) if frame<=f<=frame+12 and g is not None]
    old_ids={g for f,g in old};new_ids={g for f,g in new};known=len(old_ids)==len(new_ids)==1
    target=next(iter(old_ids)) if len(old_ids)==1 else None;identity=next(iter(new_ids)) if len(new_ids)==1 else None
    return dict(known=known,positive=None if not known else target==identity,target=target,identity=identity,
        target_anchors=old,fragment_anchors=new,direct_current_known=known and any(f==frame for f,g in new),
        limitation='Evaluation-only identity of two consistent native fragments within12frames. No interpolation; delayed fragment label does not prove the raw box identity at a prior unannotated instant.')


def simulate(video,arm,outputs,events,cap=6):
    import numpy as np
    from tracking.native_confirm_identity import NativeConfirmedIdentity
    p=NativeConfirmedIdentity(enabled=arm!='B0');frames=int(outputs[:,0].max())+1 if len(outputs) else 0;byframe={r['frame']:r for r in events};flat=[];decisions=[]
    for f in range(frames):
        rows=outputs[outputs[:,0]==f,1:];event=byframe.get(f,dict(newly_confirmed=[],edges=[]));edges=[]
        for e in event['edges']:
            accepted=arm=='GEOMETRY' or arm=='AGE' and e['age']<=cap
            if arm=='RANDOM':
                value=int.from_bytes(hashlib.sha256(f"{video}|{f}|{e['old_tid']}|{e['column']}|42".encode()).digest()[:8],'little');accepted=2*(value/2**64)-1>=.25
            edges.append(dict(old_tid=e['old_tid'],new_tid=e['new_tid'],cost=e['cost'],accepted=accepted))
        out,result=p.update(f,rows,event['newly_confirmed'],edges)
        if len(out):
            assert np.array_equal(rows[:,[0,1,2,3,5,6,7]],out[:,[0,1,2,3,5,6,7]])
            flat.append(np.column_stack([np.full(len(out),f),out]).astype(np.float32))
        if result['links'] or result['revoked']:decisions.append(dict(frame=f,**result))
    return (np.concatenate(flat) if flat else np.empty((0,9),np.float32)),decisions


def main(obs):
    import numpy as np
    protocol=phase('fit_phase.json');study=read(RUN/'protocol.json');trace=read(TRACE/'native_trace_complete.json');results={};allrows=[];sources={}
    for video,r in trace['videos'].items():
        data=read(TRACE/'native_trace'/(video+'.json'))
        with np.load(TRACE/'native_trace'/(video+'.npz'),allow_pickle=False) as z:outputs=z['outputs']
        labels,truth=native_labels(video,outputs);lookup={}
        for event in data['events']:
            for e in event['edges']:
                row=dict(video=video,frame=event['frame'],old_tid=e['old_tid'],new_tid=e['new_tid'],**edge_label(e,event['frame'],labels));allrows.append(row);lookup[(event['frame'],e['old_tid'],e['new_tid'])]=row
        results[video]={}
        for arm in ['B0','GEOMETRY','AGE','RANDOM']:
            out,decisions=simulate(video,arm,outputs,data['events']);totals=Counter();selected=[]
            if arm=='B0':assert np.array_equal(out,outputs)
            for d in decisions:
                totals['revoked']+=len(d['revoked'])
                for e in d['links']:
                    row=lookup[(d['frame'],e['old_tid'],e['new_tid'])];totals['links']+=1;totals['correct' if row['known'] and row['positive'] else 'foreign' if row['known'] else 'unknown']+=1;selected.append(dict(frame=d['frame'],**e,**{k:v for k,v in row.items() if k not in ['frame','old_tid','new_tid']}))
            results[video][arm]=dict(totals=dict(totals),links=selected)
        for n in [f'data/bdd100k/labels/box_track_20/train/{video}.json',f'runs/train_cache/{video}/alignment.json']:sources[n]=sha(ROOT/n)
        status('fit_only_native_link_geometry',video=video,complete=len(results),total=len(trace['videos']),guard=obs.read())
    totals={a:dict(sum((Counter(v[a]['totals']) for v in results.values()),Counter())) for a in ['B0','GEOMETRY','AGE','RANDOM']}
    phase('fit_phase.json');write(RUN/'fit_geometry_diagnostic.json',dict(passed=True,totals=totals,per_video=results,edge_labels=allrows,label_sources=sources,
        physical_outputs_exact=True,no_interpolation=True,scope='Fit-only descriptive check. No26labels opened. These are new causal linking opportunities, not the old641query metric. Geometry correctness alone cannot accept the full tracker.'),new=True);print(totals)


if __name__=='__main__':main(guard())
