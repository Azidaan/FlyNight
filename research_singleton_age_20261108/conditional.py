"""Fit recency on the same calibration examples and reuse ready full updates."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict
from research_singleton_age_20261108.common import *


def main(obs):
    import numpy as np
    from research_singleton_verify_20261108.simulate import Simulator as Parent
    from research_singleton_free_20261108.simulate import FreeReplay
    from research_identity_assignment_run_20261108.simulate_recovery1 import match_key
    import research_identity_assignment_run_20261108.replay as replay
    from research_identity_assignment_run_20261108.compare_recovery1 import outcome,intervals
    from tracking.free_singleton_assignment import free_singleton_matches
    phase('comparison_phase.json');protocol=read(RUN/'protocol.json');exports=read(BASE/'export_complete.json')['videos'];calibration=read(OWN/'calibration_rows.json')
    export_cache={};rows=[]
    for r in calibration['rows']:
        if not r['primary_known']:continue
        video=r['video']
        if video not in export_cache:export_cache[video]=read(ROOT/exports[video]['path'])
        track=export_cache[video]['snapshots'][str(r['frame'])]['rows'][r['row']]
        rows.append(dict(uid=r['uid'],positive=r['positive'],gap=r['frame']-track['history'][-1]['frame'],support=r['support']))
    positive=sum(r['positive'] for r in rows);negative=len(rows)-positive;assert len(rows)==73 and positive==50 and negative==23
    candidates=[]
    for cap in protocol['calibration']['caps']:
        tp=sum(bool(r['positive']) and r['support']>=3 and r['gap']<=cap for r in rows)
        fp=sum(not r['positive'] and r['support']>=3 and r['gap']<=cap for r in rows)
        candidates.append(dict(cap=cap,true_positive=tp,false_positive=fp,youden=tp/positive-fp/negative))
    selected=max(candidates,key=lambda r:(r['youden'],-r['false_positive'],r['true_positive'],-r['cap']));cap=selected['cap']
    write(RUN/'selection.json',dict(cap=cap,cap_definition='current_frame minus last_actual_observation_frame; age2 means one intermediate missing frame, not two missing frames.',candidates=candidates,selected=selected,calibration_rows=rows,frozen=datetime.now(MSK).isoformat(),
        source_sha256=sha(OWN/'calibration_rows.json'),protocol_sha256=sha(RUN/'protocol.json')),new=True);frozen=sha(RUN/'selection.json')
    previous_sources=read(OWN/'comparison_report.json')['reused_source_manifests']['new_counterfactuals']
    class Simulator(Parent):
        def __init__(self,video):
            super().__init__(video,{});self.directory=RUN/'counterfactual'/video;self.directory.mkdir(parents=True,exist_ok=True)
            for path in sorted((OWN/'counterfactual'/video).glob('*.json')):
                name=path.relative_to(ROOT).as_posix();assert sha(path)==previous_sources[name];value=read(path);self.reuse_sources[name]=sha(path)
                for frame,result in value['frames'].items():
                    key=(frame,match_key(result['first']))
                    if key in self.memo:assert self.memo[key]['final']==result['final'] and self.memo[key]['all_final']==result['all_final'] and self.memo[key]['output']==result['output']
                    else:self.memo[key]=result
        def evaluate(self,arm,alpha,beta,frames):
            assert arm=='AGE' and alpha==.2 and beta==0;output={}
            for frame in map(str,frames):
                snap=self.export['snapshots'][frame];C=self.matrices['cost_'+frame];native=np.asarray(snap['native_matches'],int).reshape(-1,2)
                rr=sorted(set(range(C.shape[0]))-set(native[:,0]));cc=sorted(set(range(C.shape[1]))-set(native[:,1]));votes=[]
                for e in self.own.get(frame,{}).values():
                    track=snap['rows'][e['row']];gap=int(frame)-track['history'][-1]['frame']
                    if e['free'] and e['support']>=3 and gap<=cap:votes.append([e['row'],e['column']])
                plan=dict(votes=votes,columns={},_native_cost=C)
                matches,_,_,_,_=free_singleton_matches(C,native,rr,cc,plan,.2,0,'U_GEO',[r['state'] for r in snap['rows']],
                    [r['cls'] for r in snap['rows']],snap['classes'],.8);key=(frame,match_key(matches))
                if key in self.memo:result=self.memo[key];self.reused+=1
                else:
                    original=replay.ReplayTracker;replay.ReplayTracker=FreeReplay
                    try:result=replay.apply_state(self.blob(frame),self.cfg,self.inputs(frame),snap,plan,.2,0,'U_GEO')
                    finally:replay.ReplayTracker=original
                    assert match_key(result['first'])==key[1];self.memo[key]=result;self.replayed+=1
                output[frame]=result
            value=dict(video=self.video,arm=arm,cap=cap,alpha=.2,beta=0,frames=output);write(self.directory/'AGE.json',value,new=True);return value
    queries=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows'];strata={r['uid']:r for r in read(ROOT/'research_identity_assignment_protocol_20261108/evaluation_strata.json')['rows']}
    groups={s:defaultdict(list) for s in ['fit20','evaluate26']}
    for q in queries:groups[q['split']][q['video']].append(q)
    sims={};results={};outputs={}
    for split,group in groups.items():
        for video in group:sims[video]=Simulator(video)
        results[split]={}
        for arm in ['B0','GEO','SUPPORT','FLYVIS','IMAGE','AGE']:
            if arm=='AGE':value=outcome(split,arm,.2,0,group,sims,strata,detail=True)
            else:
                source=OWN/'outcomes'/(split+'_'+arm+'.json');assert sha(source)==read(OWN/'comparison_report.json')['outputs'][source.relative_to(ROOT).as_posix()]
                value=dict(read(source),reused_from=source.relative_to(ROOT).as_posix(),reused_sha256=sha(source))
            path=RUN/'outcomes'/(split+'_'+arm+'.json');write(path,value,new=True);outputs[path.relative_to(ROOT).as_posix()]=sha(path);results[split][arm]=value
        status('recency_frozen_conditional_evaluation',split=split,totals=results[split]['AGE']['totals'],cap=cap,guard=obs.read())
    comparisons={}
    for a,b in protocol['full_comparisons']:
        A=results['evaluate26'][a];B=results['evaluate26'][b];value=intervals(A,B);value.pop('ci99_375');videos=sorted(A['per_video'])
        d=np.array([A['per_video'][v].get('J',0)-B['per_video'][v].get('J',0) for v in videos]);n=np.array([A['per_video'][v].get('N',0) for v in videos])
        ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)));sample=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1))
        value['ci_joint14']=np.percentile(sample,[2.5/14,100-2.5/14]).tolist();comparisons[a+'-'+b]=value
    qualifies=comparisons['AGE-B0']['ci95'][0]>0 and results['evaluate26']['AGE']['totals'].get('wrong',0)<=results['evaluate26']['B0']['totals'].get('wrong',0)
    full=read(OWN_FULL/'comparison_report.json');go=qualifies or full['accepted_selected_system']
    phase('comparison_phase.json');assert sha(RUN/'selection.json')==frozen
    write(RUN/'comparison_report.json',dict(passed=True,selection_sha256=frozen,cap=cap,comparisons=comparisons,full_gate=dict(proceed=bool(go),criterion=protocol['full_gate']),
        summaries={s:{a:r['totals'] for a,r in values.items()} for s,values in results.items()},outputs=outputs,
        reused_complete_updates=sum(s.reused for s in sims.values()),new_complete_updates=sum(s.replayed for s in sims.values()),
        source_reuse_manifests={v:s.reuse_sources for v,s in sims.items()},scope=protocol['limitation']),new=True)
    for sim in sims.values():sim.close()


if __name__=='__main__':main(guard())
