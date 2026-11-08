"""Fit one recency bound under an empirical foreign-link constraint; reuse ready work."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_native_confirm_risk_20261108.common import *
from research_native_confirm_risk_20261108.simulate import simulate,gate_signature,accepted

ARMS=['B0','GEOMETRY','AGE_FIT','AGE6','RANDOM_MATCHED','PERM_MATCHED','IMAGE','R1','DIS','FLYVIS']


def combine(rows):
    keys=set().union(*(r.keys() for r in rows));return {key:sum(r.get(key,0) for r in rows) for key in keys}


def output_from_decisions(native,decisions,frames):
    import numpy as np
    out=native.copy();byframe={d['frame']:d for d in decisions};aliases={};counts=np.bincount(native[:,0].astype(int),minlength=frames);offsets=np.r_[0,counts.cumsum()]
    for frame in range(frames):
        d=byframe.get(frame,{})
        for r in d.get('revoked',[]):assert aliases[r['native_id']]==r['old_canonical_id'];del aliases[r['native_id']]
        for r in d.get('links',[]):aliases[r['new_tid']]=r['canonical_id']
        lo,hi=offsets[frame:frame+2];ids=[aliases.get(int(t),int(t)) for t in native[lo:hi,5]];assert len(ids)==len(set(ids));out[lo:hi,5]=ids
    assert np.array_equal(out[:,[0,1,2,3,4,6,7,8]],native[:,[0,1,2,3,4,6,7,8]])
    return out


class Video:
    def __init__(self,video,split):
        import numpy as np
        self.video=video;self.split=split;self.metadata=read(TRACE/'native_trace'/(video+'.json'))
        with np.load(TRACE/'native_trace'/(video+'.npz'),allow_pickle=False) as z:self.native=z['outputs']
        score=PRIOR/'scores'/(video+'.json');self.scores=read(score)['edges'] if score.exists() else {};labels=read(PRIOR/'labels'/(video+'.json'))['rows'];self.labels={(r['frame'],r['old_tid'],r['new_tid']):r for r in labels};self.N=sum(r['known'] for r in labels);self.cache={};self.reused=0;self.new=0
        thresholds=read(PRIOR/'protocol.json')['thresholds']
        for arm in ['B0','GEOMETRY','AGE','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:
            cfg=dict(enabled=arm!='B0',cap=6 if arm=='AGE' else 10**9,score_name=None if arm in ['B0','GEOMETRY','AGE'] else arm,threshold=thresholds.get(arm))
            signature=gate_signature(cfg,self.metadata['events'],self.scores);path=PRIOR/'outcomes'/arm/(video+'.json');self.cache.setdefault(signature,(read(path),path))

    def outcome(self,config):
        signature=gate_signature(config,self.metadata['events'],self.scores)
        if signature in self.cache:self.reused+=1;return self.cache[signature],signature
        dest=RUN/'cache'/self.video/(signature+'.json')
        if dest.exists():result=read(dest);self.cache[signature]=(result,dest);self.reused+=1;return (result,dest),signature
        _,decisions=simulate(self.native,self.metadata,self.scores,config,keep_outputs=False);totals=Counter(N=self.N);links=[]
        for d in decisions:
            totals['revoked']+=len(d['revoked'])
            for e in d['links']:
                label=self.labels[(d['frame'],e['old_tid'],e['new_tid'])];kind='correct' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown';totals['links']+=1;totals[kind]+=1;totals['direct_current_known']+=bool(label.get('direct_current_known'));links.append(dict(frame=d['frame'],**e,label=label,kind=kind))
        totals['J']=totals['correct']-totals['foreign'];result=dict(passed=True,video=self.video,split=self.split,totals=dict(totals),links=links,decisions=decisions,gate_signature=signature,config=config)
        write(dest,result,new=True);self.cache[signature]=(result,dest);self.new+=1;return (result,dest),signature


def matching_cutoff(videos,selected,score_name):
    import numpy as np
    target=0;population=0;values=[]
    for v in videos.values():
        for d in v.metadata['events']:
            for e in d['edges']:
                if e['age']>selected['cap']:continue
                population+=1;target+=bool(accepted(selected,e,d['frame'],v.scores));value=v.scores[f"{d['frame']}|{e['old_tid']}|{e['new_tid']}"]['scores'][score_name]
                if value is not None and np.isfinite(value):values.append(float(value))
    values.sort(reverse=True)
    if target==0 or not values:cutoff='off'
    elif target>=len(values):cutoff=values[-1]
    elif values[target-1]>values[target]:cutoff=(values[target-1]+values[target])/2
    else:cutoff=values[target-1]
    actual=sum(cutoff!='off' and x>=cutoff for x in values)
    return cutoff,dict(population=population,target_accepted=target,defined=len(values),actual_accepted=actual,fit_labels_used_for_cutoff=False,ties_or_undefined_limit=actual!=target)


def main(obs):
    import numpy as np
    phase('phase.json');p=read(RUN/'protocol.json');fit_names=read(TRACE/'native_trace_complete.json')['videos'];eval_names=read(PRIOR/'evaluate26_metadata_complete.json')['videos'];fit={v:Video(v,'fit20') for v in fit_names};candidates=[];details={};thresholds=p['calibration']['thresholds'];order=p['calibration']['source_candidates']
    # Only fit20 objects and their labels exist until selection is sealed.
    for source in order:
        for cap in p['calibration']['caps']:
            config=dict(enabled=True,cap=cap,score_name=None if source=='GEOMETRY' else source,threshold=thresholds.get(source));per={}
            for video,obj in fit.items():(out,path),signature=obj.outcome(config);per[video]=out['totals']
            totals=combine(list(per.values()));row=dict(source=source,cap=cap,config=config,totals=totals,eligible=totals.get('foreign',0)==0);candidates.append(row);details[source+'|'+str(cap)]=per
        status('fit_native_link_error_constraint',source=source,complete=order.index(source)+1,total=len(order),guard=obs.read())
    def rank(r):return (r['totals'].get('correct',0),-r['cap'],-order.index(r['source']))
    selected=max((r for r in candidates if r['eligible']),key=rank);age=max((r for r in candidates if r['source']=='GEOMETRY' and r['eligible']),key=rank);cap=selected['cap']
    random_threshold,random_matching=matching_cutoff(fit,selected['config'],'RANDOM');perm_threshold,perm_matching=matching_cutoff(fit,selected['config'],'PERM')
    variants={a:dict(enabled=a!='B0',cap=cap,score_name=None if a in ['B0','GEOMETRY','AGE_FIT','AGE6'] else 'RANDOM' if a=='RANDOM_MATCHED' else 'PERM' if a=='PERM_MATCHED' else a,threshold=random_threshold if a=='RANDOM_MATCHED' else perm_threshold if a=='PERM_MATCHED' else thresholds.get(a)) for a in ARMS};variants['AGE_FIT']['cap']=age['cap'];variants['AGE6']['cap']=6
    selection=dict(frozen=datetime.now(MSK).isoformat(),selected_system=selected['source'],shared_cap=cap,selected=selected,age_fit=age,candidates=candidates,variants=variants,frequency_controls=dict(RANDOM_MATCHED=random_matching,PERM_MATCHED=perm_matching),known_training_edges=sum(v.N for v in fit.values()),calibration_only_fit20=True)
    write(RUN/'selection.json',selection,new=True);sealed=sha(RUN/'selection.json');results={};artifacts={};reuse=Counter();technical=[]
    for split,group in [('fit20',fit),('evaluate26',None)]:
        if split=='evaluate26':assert sha(RUN/'selection.json')==sealed;group={v:Video(v,'evaluate26') for v in eval_names}
        results[split]={a:dict(per_video={},totals={}) for a in ARMS}
        for video,obj in group.items():
            for arm,config in variants.items():
                (value,source),signature=obj.outcome(config);out=output_from_decisions(obj.native,value['decisions'],obj.metadata['frames']);fingerprint=hashlib.sha256(out.tobytes()).hexdigest()
                if 'outputs_sha256' in value:assert value['outputs_sha256']==fingerprint
                result=dict(value,arm=arm,config=config,gate_signature=signature,reused_outcome_source=source.relative_to(ROOT).as_posix(),reused_outcome_sha256=sha(source),physical_outputs_exact=True,outputs_sha256=fingerprint);dest=RUN/'outcomes'/arm/(video+'.json');write(dest,result,new=True);artifacts[dest.relative_to(ROOT).as_posix()]=sha(dest);results[split][arm]['per_video'][video]=result['totals']
            # Check the optimized slicing implementation against a ready changed
            # output once per split, including its exact decision ledger.
            if not any(r['split']==split for r in technical):
                oldpath=PRIOR/'outcomes/GEOMETRY'/(video+'.json');old=read(oldpath)
                if old['links']:
                    oldcfg=dict(enabled=True,cap=10**9,score_name=None,threshold=None);actual,decisions=simulate(obj.native,obj.metadata,obj.scores,oldcfg);assert hashlib.sha256(actual.tobytes()).hexdigest()==old['outputs_sha256'] and decisions==old['decisions'];technical.append(dict(split=split,video=video,ready_changed_output_and_decisions_exact=True))
            reuse['ready_outcome_hits']+=obj.reused;reuse['new_unique_gate_outcomes']+=obj.new;status('empirical_error_gate_evaluation',split=split,video=video,complete=len(results[split]['B0']['per_video']),total=len(group),guard=obs.read())
        for arm in ARMS:results[split][arm]['totals']=combine(list(results[split][arm]['per_video'].values()))
    comparisons={};chosen=selection['selected_system'];pairs=[(chosen,'B0'),('AGE_FIT','B0'),('AGE6','B0'),('GEOMETRY','B0'),('IMAGE','B0')]+[('FLYVIS',a) for a in ARMS if a!='FLYVIS'];videos=sorted(eval_names);idx=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)))
    for a,b in pairs:
        key=a+'-'+b
        if key in comparisons:continue
        d=np.array([results['evaluate26'][a]['per_video'][v].get('J',0)-results['evaluate26'][b]['per_video'][v].get('J',0) for v in videos]);n=np.array([results['evaluate26'][a]['per_video'][v]['N'] for v in videos]);sample=100*d[idx].sum(1)/np.maximum(1,n[idx].sum(1));comparisons[key]=dict(delta_J=int(d.sum()),delta_per100=100*d.sum()/max(1,n.sum()),ci95=np.percentile(sample,[2.5,97.5]).tolist(),ci_corrected=np.percentile(sample,[2.5/16,100-2.5/16]).tolist())
    def qualifies(arm):return comparisons[arm+'-B0']['ci95'][0]>0 and results['evaluate26'][arm]['totals'].get('foreign',0)==0
    proceed=selected['totals'].get('correct',0)>0 and (qualifies(chosen) or qualifies('AGE_FIT'))
    assert len(technical)==2
    phase('phase.json');write(RUN/'comparison_report.json',dict(passed=True,selected_system=chosen,selection_sha256=sealed,results=results,comparisons=comparisons,full_gate=dict(proceed=bool(proceed),criterion=p['gate']),artifacts=artifacts,reuse=dict(reuse),technical_parity=technical,historical=True,independent_final_test=False),new=True)


if __name__=='__main__':main(guard())
