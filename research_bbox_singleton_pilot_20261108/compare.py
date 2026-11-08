"""Fit only on fit20, then inspect historical26; native boxes and states fixed."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from research_bbox_singleton_pilot_20261108.common import *


def edge_key(frame,e):return f"{frame}|{e['old_tid']}|{e['new_tid']}"


def events_for(data,config):
    from tracking.bbox_singleton_gate import proposals
    return [dict(frame=d['frame'],newly_confirmed=d['newly_confirmed'],edges=proposals(d,**config)) for d in data['events']]


def labels_for(video,native,events):
    import numpy as np
    from scripts.association_log import label
    from datasets.bdd100k_loader import CLASS_NAMES
    from research_native_confirm_policy_20261108.fit_geometry import edge_label
    ready=ROOT/'runs/native_confirm_policy_20261108/labels'/(video+'.json');old={edge_key(r['frame'],r):r for r in read(ready)['rows']} if ready.exists() else {}
    lookup={};new=[];needed=set()
    for d in events:
        for e in d['edges']:
            key=edge_key(d['frame'],e)
            if key in old:lookup[key]=old[key];continue
            new.append((d['frame'],e));last=e['row_data']['history'][-1]['frame'];needed.update(range(max(0,last-12),last+1));needed.update(range(d['frame'],d['frame']+13))
    annotations=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json');aligned=read(ROOT/f'runs/train_cache/{video}/alignment.json')['matches'];assert len(annotations)==len(aligned);tracklets=defaultdict(list);assignments=[]
    for frame,a in zip(annotations,aligned):
        f=int(a['video_frame_index'])
        if f not in needed:continue
        objs=[]
        for l in frame.get('labels') or []:
            if l['category'] not in CLASS_NAMES.values() or (l.get('attributes') or {}).get('crowd',False):continue
            b=l['box2d'];objs.append((str(l['id']),l['category'],np.array([b[k] for k in ['x1','y1','x2','y2']])))
        out=native[native[:,0]==f];identities=label(out[:,1:5],[CLASS_NAMES[int(c)] for c in out[:,7]],objs)
        assignments.append(dict(frame=f,matches=[[int(row[5]),identity] for row,identity in zip(out,identities)]))
        for row,identity in zip(out,identities):tracklets[int(row[5])].append((f,identity))
    for frame,e in new:lookup[edge_key(frame,e)]=dict(video=video,frame=frame,old_tid=e['old_tid'],new_tid=e['new_tid'],**edge_label(e,frame,tracklets))
    write(RUN/'labels'/(video+'.json'),dict(rows=lookup,ready_edges_reused=sum(k in old for k in lookup),new_edge_labels=len(new),new_needed_frame_assignments=assignments,no_interpolation=True),new=True)
    return lookup


def simulate(native,data,events,enabled=True):
    import numpy as np
    from tracking.native_confirm_identity import NativeConfirmedIdentity
    policy=NativeConfirmedIdentity(enabled=enabled);byframe={d['frame']:d for d in events};counts=np.bincount(native[:,0].astype(int),minlength=data['frames']);offset=np.r_[0,counts.cumsum()];decisions=[];flat=[]
    for frame in range(data['frames']):
        rows=native[offset[frame]:offset[frame+1],1:];event=byframe.get(frame,dict(newly_confirmed=[],edges=[]));out,d=policy.update(frame,rows,event['newly_confirmed'],[dict(e,accepted=True) for e in event['edges']])
        if d['links'] or d['revoked']:decisions.append(dict(frame=frame,**d))
        if len(out):flat.append(np.column_stack([np.full(len(out),frame),out]).astype(np.float32))
    outputs=np.concatenate(flat) if flat else np.empty((0,9),np.float32)
    assert np.array_equal(native[:,[0,1,2,3,4,6,7,8]],outputs[:,[0,1,2,3,4,6,7,8]])
    if not enabled:assert np.array_equal(outputs,native)
    return outputs,decisions


def total(rows):
    keys=set().union(*(r.keys() for r in rows));return {k:sum(r.get(k,0) for r in rows) for k in keys}


def run_video(video,configs,split):
    import numpy as np
    data=read(INVENTORY/'native'/(video+'.json'))
    with np.load(INVENTORY/'native'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
    events={name:events_for(data,cfg) for name,cfg in configs.items()};union={}
    for group in events.values():
        for d in group:
            union.setdefault(d['frame'],dict(frame=d['frame'],newly_confirmed=d['newly_confirmed'],edges={}))
            for e in d['edges']:union[d['frame']]['edges'][edge_key(d['frame'],e)]=e
    all_events=[dict(d,edges=list(d['edges'].values())) for d in union.values()];labels=labels_for(video,native,all_events);result={}
    for name in ['B0',*configs]:
        out,decisions=simulate(native,data,events.get(name,[]),enabled=name!='B0');counts=Counter(N=sum(r['known'] for r in labels.values()));links=[]
        for d in decisions:
            counts['revoked']+=len(d['revoked'])
            for e in d['links']:
                l=labels[edge_key(d['frame'],e)];kind='correct' if l['known'] and l['positive'] else 'foreign' if l['known'] else 'unknown';counts['links']+=1;counts[kind]+=1;links.append(dict(frame=d['frame'],**e,label=l,kind=kind))
        counts['J']=counts['correct']-counts['foreign'];r=dict(passed=True,video=video,split=split,arm=name,totals=dict(counts),links=links,decisions=decisions,physical_outputs_exact=True,outputs_sha256=hashlib.sha256(out.tobytes()).hexdigest())
        write(RUN/'outcomes'/name/(video+'.json'),r,new=True);result[name]=dict(counts)
    return result


def main(obs):
    import numpy as np
    p=phase();entries=read(INVENTORY/'protocol.json')['videos'];configs=p['configs'];results={}
    for split in ['fit20','evaluate26']:
        names=[v for v,e in entries.items() if e['split']==split];per={}
        if split=='evaluate26':assert sha(RUN/'selection.json')==sealed
        for video in names:
            per[video]=run_video(video,configs,split);status('bbox_geometry_pilot',split=split,video=video,complete=len(per),total=len(names),guard=obs.read())
        totals={a:total([v[a] for v in per.values()]) for a in ['B0',*configs]};results[split]=dict(per_video=per,totals=totals)
        if split=='fit20':
            eligible=[a for a in configs if totals[a].get('foreign',0)==0]
            chosen=max(eligible,key=lambda a:(totals[a].get('correct',0),-configs[a]['cap'],-int(configs[a]['bidirectional']))) if eligible else 'B0'
            selection=dict(registered=datetime.now(MSK).isoformat(),selected=chosen,configs=configs,totals=totals,fit_only=True,unknown_not_negative=True)
            write(RUN/'selection.json',selection,new=True);sealed=sha(RUN/'selection.json')
    names=list(results['evaluate26']['per_video']);idx=np.random.default_rng(42).integers(0,len(names),(20000,len(names)));comparisons={}
    for a in configs:
        delta=np.array([results['evaluate26']['per_video'][v][a]['J'] for v in names]);n=np.array([results['evaluate26']['per_video'][v][a]['N'] for v in names]);bs=100*delta[idx].sum(1)/np.maximum(1,n[idx].sum(1));comparisons[a+'-B0']=dict(delta_J=int(delta.sum()),delta_per100=100*delta.sum()/max(1,n.sum()),ci95=np.percentile(bs,[2.5,97.5]).tolist(),ci_corrected=np.percentile(bs,[2.5/12,100-2.5/12]).tolist())
    proceed=chosen!='B0' and results['fit20']['totals'][chosen].get('correct',0)>0 and results['evaluate26']['totals'][chosen].get('foreign',0)==0 and comparisons[chosen+'-B0']['ci95'][0]>0
    phase();write(RUN/'comparison_report.json',dict(passed=True,selected=chosen,results=results,comparisons=comparisons,selection_sha256=sealed,full_gate=dict(proceed=bool(proceed),criterion=p['gate']),historical=True,independent_final_test=False,new_core_calls=0,new_detector_calls=0,new_image_reads=0),new=True)


if __name__=='__main__':main(guard())
