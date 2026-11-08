"""A matched ordinary-cost control: vary only alpha and a bounded recency cap."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_bbox_cost_pilot_20261108.common import *
from research_bbox_singleton_pilot_20261108.compare import total,events_for,edge_key


def simulate(native,data,events,alpha):
    import numpy as np
    from tracking.native_confirm_identity import NativeConfirmedIdentity
    policy=NativeConfirmedIdentity(alpha=alpha);byframe={r['frame']:r for r in events};counts=np.bincount(native[:,0].astype(int),minlength=data['frames']);offsets=np.r_[0,counts.cumsum()];out=native.copy();decisions=[]
    for frame in range(data['frames']):
        lo,hi=offsets[frame:frame+2];rows=native[lo:hi,1:];event=byframe.get(frame,dict(newly_confirmed=[],edges=[]));changed,d=policy.update(frame,rows,event['newly_confirmed'],[dict(e,accepted=True) for e in event['edges']]);out[lo:hi,1:]=changed
        if d['links'] or d['revoked'] or d['allowed']:decisions.append(dict(frame=frame,**d))
    assert np.array_equal(out[:,[0,1,2,3,4,6,7,8]],native[:,[0,1,2,3,4,6,7,8]])
    return out,decisions


def run_video(video,configs,split):
    import numpy as np
    data=read(INVENTORY/'native'/(video+'.json'))
    with np.load(INVENTORY/'native'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
    labels=read(PRIOR_PILOT/'labels'/(video+'.json'))['rows'];result={};signatures={};events_by_cap={cap:events_for(data,dict(cap=cap,threshold=.2,bidirectional=False)) for cap in [2,4,6]}
    for arm,cfg in configs.items():
        ready=PRIOR_PILOT/'outcomes'/f"BBOX{cfg['cap']}"/(video+'.json')
        if cfg['alpha']==.2:
            old=read(ready);r=dict(old,arm=arm,reused_outcome_source=ready.relative_to(ROOT).as_posix(),reused_outcome_sha256=sha(ready))
        else:
            out,decisions=simulate(native,data,events_by_cap[cfg['cap']],cfg['alpha']);counts=Counter(N=sum(l['known'] for l in labels.values()));links=[]
            for d in decisions:
                counts['revoked']+=len(d['revoked'])
                for e in d['links']:
                    label=labels[edge_key(d['frame'],e)];kind='correct' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown';counts['links']+=1;counts[kind]+=1;links.append(dict(frame=d['frame'],**e,label=label,kind=kind))
            counts['J']=counts['correct']-counts['foreign'];r=dict(passed=True,video=video,split=split,arm=arm,totals=dict(counts),links=links,decisions=decisions,physical_outputs_exact=True,outputs_sha256=hashlib.sha256(out.tobytes()).hexdigest())
        write(RUN/'outcomes'/arm/(video+'.json'),r,new=True);result[arm]=r['totals']
    return result


def main(obs):
    import numpy as np
    p=phase();entries=read(INVENTORY/'protocol.json')['videos'];results={}
    for split in ['fit20','evaluate26']:
        if split=='evaluate26':assert sha(RUN/'selection.json')==sealed
        names=[v for v,e in entries.items() if e['split']==split];per={}
        for video in names:per[video]=run_video(video,p['configs'],split);status('bounded_cost_pilot',split=split,video=video,complete=len(per),total=len(names),guard=obs.read())
        totals={a:total([v[a] for v in per.values()]) for a in p['configs']};results[split]=dict(per_video=per,totals=totals)
        if split=='fit20':
            eligible=[a for a in p['configs'] if totals[a].get('foreign',0)==0];chosen=max(eligible,key=lambda a:(totals[a].get('correct',0),-p['configs'][a]['cap'],-p['configs'][a]['alpha'])) if eligible else 'B0';write(RUN/'selection.json',dict(selected=chosen,totals=totals,configs=p['configs'],fit_only=True),new=True);sealed=sha(RUN/'selection.json')
    names=list(results['evaluate26']['per_video']);idx=np.random.default_rng(42).integers(0,len(names),(20000,len(names)));comparisons={}
    for a in p['configs']:
        rows=[results['evaluate26']['per_video'][v][a] for v in names];d=np.array([r['J'] for r in rows]);n=np.array([r['N'] for r in rows]);sample=100*d[idx].sum(1)/np.maximum(1,n[idx].sum(1));comparisons[a+'-B0']=dict(delta_J=int(d.sum()),delta_per100=100*d.sum()/max(1,n.sum()),ci95=np.percentile(sample,[2.5,97.5]).tolist(),ci_corrected=np.percentile(sample,[2.5/20,100-2.5/20]).tolist())
    proceed=chosen!='B0' and results['fit20']['totals'][chosen].get('correct',0)>0 and results['evaluate26']['totals'][chosen].get('foreign',0)==0 and comparisons[chosen+'-B0']['ci95'][0]>0
    phase();write(RUN/'comparison_report.json',dict(passed=True,selected=chosen,selection_sha256=sealed,results=results,comparisons=comparisons,full_gate=dict(proceed=bool(proceed),criterion=p['gate']),historical=True,independent_final_test=False,new_core_calls=0,new_image_reads=0,new_detector_calls=0),new=True)


if __name__=='__main__':main(guard())
