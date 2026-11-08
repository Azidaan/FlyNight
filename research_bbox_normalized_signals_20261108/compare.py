"""Same causal small-cost policy: support, image/motion maps and FlyVis vetoes."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_bbox_normalized_signals_20261108.common import *
from research_bbox_singleton_pilot_20261108.compare import total,edge_key
ARMS=['B0','BBOX','SUPPORT','RANDOM_MATCHED','PERM_MATCHED','IMAGE','R1','DIS','FLYVIS']
ELIGIBLE=['SUPPORT','IMAGE','R1','DIS','FLYVIS']


def metadata(video):
    data=read(INVENTORY/'native'/(video+'.json'));items=[r for r in read(RUN/'inventory.json')['events'] if r['video']==video];grouped=defaultdict(list)
    for e in items:grouped[e['frame']].append(e)
    original={d['frame']:d for d in data['events']};return dict(data,events=[dict(frame=f,newly_confirmed=original[f]['newly_confirmed'],edges=rows) for f,rows in grouped.items()])


def passes(config,e,frame,scores):
    import numpy as np
    if not config['enabled']:return False
    if config['score_name'] is None:return True
    value=scores[edge_key(frame,e)]['scores'][config['score_name']];threshold=config['threshold']
    return threshold!='off' and value is not None and np.isfinite(value) and value>=threshold


def simulate(native,data,scores,config):
    import numpy as np
    from tracking.native_confirm_identity import NativeConfirmedIdentity
    policy=NativeConfirmedIdentity(alpha=.025,enabled=config['enabled']);byframe={d['frame']:d for d in data['events']};counts=np.bincount(native[:,0].astype(int),minlength=data['frames']);offset=np.r_[0,counts.cumsum()];result=native.copy();decisions=[]
    for frame in range(data['frames']):
        lo,hi=offset[frame:frame+2];rows=native[lo:hi,1:];event=byframe.get(frame,dict(newly_confirmed=[],edges=[]));out,d=policy.update(frame,rows,event['newly_confirmed'],[dict(e,accepted=bool(passes(config,e,frame,scores))) for e in event['edges']]);result[lo:hi,1:]=out
        if d['links'] or d['revoked'] or d['allowed']:decisions.append(dict(frame=frame,**d))
    assert np.array_equal(result[:,[0,1,2,3,4,6,7,8]],native[:,[0,1,2,3,4,6,7,8]])
    return result,decisions


def cutoff(fit,name):
    target=0;values=[]
    for video in fit:
        path=RUN/'scores'/(video+'.json');scores=read(path)['edges'] if path.exists() else {}
        for e in scores.values():
            f=e['scores']['FLYVIS'];target+=f is not None and f>=.25
            if e['scores'][name] is not None:values.append(float(e['scores'][name]))
    values.sort(reverse=True)
    threshold='off' if not target or not values else values[-1] if target>=len(values) else (values[target-1]+values[target])/2 if values[target-1]>values[target] else values[target-1]
    return threshold,dict(target=target,defined=len(values),actual=0 if threshold=='off' else sum(v>=threshold for v in values),labels_used=False)


def run_video(video,split,configs,technical):
    import numpy as np
    data=metadata(video)
    with np.load(INVENTORY/'native'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
    path=RUN/'scores'/(video+'.json');scores=read(path)['edges'] if path.exists() else {};labels=read(PRIOR_PILOT/'labels'/(video+'.json'))['rows'];results={}
    for arm,cfg in configs.items():
        ready=PRIOR_PILOT/'outcomes/B0'/(video+'.json') if arm=='B0' else COST/'outcomes/A025_C2'/(video+'.json') if arm=='BBOX' else None
        if ready:
            old=read(ready);r=dict(old,arm=arm,reused_outcome_source=ready.relative_to(ROOT).as_posix(),reused_outcome_sha256=sha(ready))
            if arm=='BBOX' and old['links'] and split not in technical:
                out,d=simulate(native,data,scores,cfg);assert hashlib.sha256(out.tobytes()).hexdigest()==old['outputs_sha256'];assert [(x['frame'],x['links'],x['revoked']) for x in d if x['links'] or x['revoked']]==[(x['frame'],x['links'],x['revoked']) for x in old['decisions'] if x['links'] or x['revoked']];technical[split]=video
        else:
            out,d=simulate(native,data,scores,cfg);totals=Counter(N=sum(r['known'] for r in labels.values()));links=[]
            for row in d:
                totals['revoked']+=len(row['revoked'])
                for e in row['links']:
                    label=labels[edge_key(row['frame'],e)];kind='correct' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown';totals['links']+=1;totals[kind]+=1;links.append(dict(frame=row['frame'],**e,label=label,kind=kind))
            totals['J']=totals['correct']-totals['foreign'];r=dict(passed=True,video=video,split=split,arm=arm,totals=dict(totals),links=links,decisions=d,physical_outputs_exact=True,outputs_sha256=hashlib.sha256(out.tobytes()).hexdigest())
        write(RUN/'outcomes'/arm/(video+'.json'),r,new=True);results[arm]=r['totals']
    return results


def main(obs):
    import numpy as np
    phase();phase('comparison_phase.json');p=read(RUN/'protocol.json');entries=read(INVENTORY/'protocol.json')['videos'];fit=[v for v,e in entries.items() if e['split']=='fit20'];random_t,random_count=cutoff(fit,'RANDOM');perm_t,perm_count=cutoff(fit,'PERM');configs={a:dict(enabled=a!='B0',score_name=None if a in ['B0','BBOX'] else 'RANDOM' if a=='RANDOM_MATCHED' else 'PERM' if a=='PERM_MATCHED' else a,threshold=random_t if a=='RANDOM_MATCHED' else perm_t if a=='PERM_MATCHED' else p['thresholds'].get(a)) for a in ARMS};results={};technical={}
    for split in ['fit20','evaluate26']:
        if split=='evaluate26':assert sha(RUN/'selection.json')==sealed
        names=[v for v,e in entries.items() if e['split']==split];per={}
        for video in names:per[video]=run_video(video,split,configs,technical);status('normalized_spatial_policy_comparison',split=split,video=video,complete=len(per),total=len(names),guard=obs.read())
        totals={a:total([r[a] for r in per.values()]) for a in ARMS};results[split]=dict(per_video=per,totals=totals)
        if split=='fit20':
            eligible=[a for a in ELIGIBLE if totals[a].get('correct',0)>0 and totals[a].get('foreign',0)==0];selected=max(eligible,key=lambda a:(totals[a].get('correct',0),-totals[a].get('links',0),-ELIGIBLE.index(a))) if eligible else 'B0';selection=dict(selected=selected,eligible_systems=eligible,configs=configs,fit_only=True,totals=totals,rate_controls=dict(RANDOM_MATCHED=random_count,PERM_MATCHED=perm_count));write(RUN/'selection.json',selection,new=True);sealed=sha(RUN/'selection.json')
    names=list(results['evaluate26']['per_video']);idx=np.random.default_rng(42).integers(0,len(names),(20000,len(names)));comparisons={}
    pairs=[(a,'B0') for a in ELIGIBLE]+[('FLYVIS',a) for a in ARMS if a!='FLYVIS']
    for a,b in pairs:
        rows=results['evaluate26']['per_video'];d=np.array([rows[v][a]['J']-rows[v][b]['J'] for v in names]);n=np.array([rows[v][a]['N'] for v in names]);sample=100*d[idx].sum(1)/np.maximum(1,n[idx].sum(1));comparisons[a+'-'+b]=dict(delta_J=int(d.sum()),delta_per100=100*d.sum()/max(1,n.sum()),ci95=np.percentile(sample,[2.5,97.5]).tolist(),ci_corrected=np.percentile(sample,[2.5/20,100-2.5/20]).tolist())
    qualified=[a for a in selection['eligible_systems'] if results['evaluate26']['totals'][a].get('foreign',0)==0 and comparisons[a+'-B0']['ci95'][0]>0];assert len(technical)==2;phase();phase('comparison_phase.json');write(RUN/'comparison_report.json',dict(passed=True,selected=selected,selection_sha256=sealed,results=results,comparisons=comparisons,full_gate=dict(proceed=bool(qualified),qualified_preregistered_systems=qualified),technical_cached_control_parity=technical,historical=True,independent_final_test=False,scope='Native-fragment links on sparse actual anchors; conditional correct links alone do not prove all public canonical identities or a full tracker gain. Only first problem; primary and predefined secondary remain separate.'),new=True)


if __name__=='__main__':main(guard())
