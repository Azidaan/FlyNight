"""Freeze fit-only system choice before opening any evaluate26 identity labels."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_native_confirm_policy_20261108.common import *

ARMS=['B0','GEOMETRY','AGE','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']


def main(obs):
    import numpy as np
    from research_native_confirm_policy_20261108.simulate import simulate
    from research_native_confirm_policy_20261108.fit_geometry import native_labels,edge_label
    phase('comparison_phase.json');p=read(RUN/'protocol.json');fit=read(TRACE/'native_trace_complete.json')['videos'];evaluate=read(RUN/'evaluate26_metadata_complete.json')['videos'];score_manifest=read(RUN/'scores_complete.json')['videos'];fitlabels=read(RUN/'fit_geometry_diagnostic.json')['edge_labels'];knownfit={(r['video'],r['frame'],r['old_tid'],r['new_tid']):r for r in fitlabels};results={};artifact={};label_sources={}
    for split,videos in [('fit20',fit),('evaluate26',evaluate)]:
        results[split]={a:dict(per_video={},totals={}) for a in ARMS}
        if split=='evaluate26':assert sha(RUN/'selection.json')==sealed
        for video in videos:
            data=read(TRACE/'native_trace'/(video+'.json'))
            with np.load(TRACE/'native_trace'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
            sp=RUN/'scores'/(video+'.json');scores=read(sp)['edges'] if sp.exists() else {}
            if sp.exists():assert sha(sp)==score_manifest[video]
            if split=='evaluate26':
                labels,truth=native_labels(video,native);lookup={}
                for event in data['events']:
                    for e in event['edges']:lookup[(event['frame'],e['old_tid'],e['new_tid'])]=dict(video=video,frame=event['frame'],old_tid=e['old_tid'],new_tid=e['new_tid'],**edge_label(e,event['frame'],labels))
            else:lookup={(f,o,n):r for (v,f,o,n),r in knownfit.items() if v==video}
            known=sum(r['known'] for r in lookup.values());write(RUN/'labels'/(video+'.json'),dict(rows=list(lookup.values())),new=True)
            for arm in ARMS:
                out,decisions=simulate(arm,native,data['events'],scores,p,data['frames']);c=Counter(N=known);links=[]
                for d in decisions:
                    c['revoked']+=len(d['revoked'])
                    for e in d['links']:
                        r=lookup[(d['frame'],e['old_tid'],e['new_tid'])];kind='correct' if r['known'] and r['positive'] else 'foreign' if r['known'] else 'unknown';c['links']+=1;c[kind]+=1;c['direct_current_known']+=bool(r.get('direct_current_known'));links.append(dict(frame=d['frame'],**e,label=r,kind=kind))
                c['J']=c['correct']-c['foreign'];results[split][arm]['per_video'][video]=dict(c)
                path=RUN/'outcomes'/arm/(video+'.json');write(path,dict(passed=True,split=split,video=video,arm=arm,totals=dict(c),links=links,decisions=decisions,physical_outputs_exact=True,outputs_sha256=hashlib.sha256(out.tobytes()).hexdigest()),new=True);artifact[path.relative_to(ROOT).as_posix()]=sha(path)
            for n in [f'data/bdd100k/labels/box_track_20/train/{video}.json',f'runs/train_cache/{video}/alignment.json']:label_sources[n]=sha(ROOT/n)
            status('native_confirm_identity_evaluation',split=split,video=video,complete=len(results[split]['B0']['per_video']),total=len(videos),guard=obs.read())
        for arm in ARMS:
            vv=list(results[split][arm]['per_video'].values());kk=set().union(*(v.keys() for v in vv));results[split][arm]['totals']={k:sum(v.get(k,0) for v in vv) for k in kk}
        if split=='fit20':
            order=['GEOMETRY','AGE','IMAGE','R1','DIS','FLYVIS'];selected=max(order,key=lambda a:(results[split][a]['totals'].get('J',0),-results[split][a]['totals'].get('foreign',0),-results[split][a]['totals'].get('links',0),-order.index(a)))
            write(RUN/'selection.json',dict(frozen=datetime.now(MSK).isoformat(),selected_system=selected,thresholds=p['thresholds'],recency_cap=p['recency_cap'],fit_totals={a:r['totals'] for a,r in results[split].items()},new_threshold_searches=0),new=True);sealed=sha(RUN/'selection.json')
        else:assert sha(RUN/'selection.json')==sealed
    comparisons={};pairs=[(selected,'B0'),('GEOMETRY','B0')]+[('FLYVIS',a) for a in ARMS if a!='FLYVIS']
    videos=sorted(evaluate);ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)))
    for a,b in pairs:
        key=a+'-'+b
        if key in comparisons:continue
        d=np.array([results['evaluate26'][a]['per_video'][v].get('J',0)-results['evaluate26'][b]['per_video'][v].get('J',0) for v in videos]);n=np.array([results['evaluate26'][a]['per_video'][v]['N'] for v in videos]);sample=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1));comparisons[key]=dict(delta_J=int(d.sum()),delta_per100=100*d.sum()/max(1,n.sum()),ci95=np.percentile(sample,[2.5,97.5]).tolist(),ci_corrected=np.percentile(sample,[2.5/12,100-2.5/12]).tolist())
    go=any(comparisons[a+'-B0']['ci95'][0]>0 for a in {selected,'GEOMETRY'})
    phase('comparison_phase.json');write(RUN/'comparison_report.json',dict(passed=True,selected_system=selected,selection_sha256=sealed,results=results,comparisons=comparisons,full_gate=dict(proceed=bool(go),criterion=p['conditional_gate']),artifacts=artifact,label_sources=label_sources,scope='Historical new-native-confirmation linking opportunities. Fixed physical boxes, not old641one-step requests and not final full tracker acceptance.'),new=True)


if __name__=='__main__':main(guard())
