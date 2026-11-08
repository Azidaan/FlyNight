"""Fit absolute identity gates, freeze once, then evaluate all historical queries."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_singleton_verify_20261108.common import *

ARMS=['B0','GEO','THRESHOLD','SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']


def main(obs):
    import numpy as np
    from research_singleton_verify_20261108.score import score_all
    from research_singleton_verify_20261108.simulate import Simulator
    from research_identity_assignment_run_20261108.evaluator import Labels
    from research_identity_assignment_run_20261108.compare_recovery1 import outcome,intervals,LABEL_CACHE
    from tracking.own_template_verification import passes
    phase('comparison_phase.json');assert read(RUN/'data_complete.json')['passed']
    import pytest
    assert pytest.main(['-q','tests/test_own_template_verification.py','--basetemp',str(RUN/'verification_pytest_temp')])==0
    zero=read(PREVIOUS/'technical.json');assert zero['passed'] and zero['zero_native_eight_columns_and_offsets_exact'] and zero['provider_calls']==0
    if not (RUN/'scores_complete.json').exists():score_all(obs)
    protocol=read(RUN/'protocol.json');inventory=read(RUN/'inventory_recovery1.json');exports=read(BASE/'export_complete.json')['videos']
    queries=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows']
    strata={r['uid']:r for r in read(ROOT/'research_identity_assignment_protocol_20261108/evaluation_strata.json')['rows']}
    groups={s:defaultdict(list) for s in ['fit20','evaluate26']}
    for q in queries:groups[q['split']][q['video']].append(q)
    labels={};calibration=[];label_sources={}
    # No evaluate26 Labels object is constructed until the selection is sealed.
    for video in groups['fit20']:
        export=read(ROOT/exports[video]['path']);lab=Labels(video,export);labels[video]=lab
        path=RUN/'scores'/(video+'.json');scores=read(path)['frames'] if path.exists() else {}
        for event in inventory['events']:
            if event['video']!=video or not event['calibration']:continue
            uid=event['query_uid'];row=scores[str(event['frame'])][f"{event['row']},{event['column']}"]
            raw_id=export['snapshots'][str(event['frame'])]['raw_ids'][event['column']]
            identity=lab.current_labels(event['frame']).get(raw_id);target=lab.supervision[uid]['target_identity']
            known=bool(strata[uid]['primary_known'] and identity is not None and target is not None)
            calibration.append(dict(uid=uid,video=video,frame=event['frame'],row=event['row'],column=event['column'],
                primary_known=known,positive=None if not known else identity==target,scores=row['scores'],support=row['support'],identity=identity,target=target))
        label_sources.update(lab.sources)
    known=[r for r in calibration if r['primary_known']];positive=sum(bool(r['positive']) for r in known);negative=len(known)-positive
    assert positive>0 and negative>0,('Both calibration identity classes are required',positive,negative)
    candidates={};selected={};thresholds={'SUPPORT':-1}
    def rank(r):return (r['youden'],-r['false_positive'],r['true_positive'],float('inf') if r['threshold']=='off' else r['threshold'])
    for arm in ARMS[4:]:
        rows=[]
        for threshold in protocol['calibration']['thresholds']:
            tp=sum(bool(r['positive']) and passes(r['scores'][arm],threshold) for r in known)
            fp=sum(not r['positive'] and passes(r['scores'][arm],threshold) for r in known)
            rows.append(dict(threshold=threshold,true_positive=tp,false_positive=fp,positive=positive,negative=negative,
                youden=tp/positive-fp/negative,undefined=sum(r['scores'][arm] is None for r in known)))
        best=max(rows,key=rank);candidates[arm]=rows;selected[arm]=best;thresholds[arm]=best['threshold']
    order=['IMAGE','R1','DIS','FLYVIS'];system=max(order,key=lambda arm:(selected[arm]['youden'],-selected[arm]['false_positive'],selected[arm]['true_positive'],-order.index(arm)))
    write(RUN/'calibration_rows.json',dict(rows=calibration,label_sources=label_sources),new=True)
    selection=dict(frozen=datetime.now(MSK).isoformat(),thresholds=thresholds,selected_system=system,candidates=candidates,selected=selected,
        calibration=dict(known=len(known),positive=positive,negative=negative,unknown=len(calibration)-len(known)),
        calibration_sha256=sha(RUN/'calibration_rows.json'),protocol_sha256=sha(RUN/'protocol.json'))
    write(RUN/'selection.json',selection,new=True);sealed=sha(RUN/'selection.json');sims={};results={};outputs={}
    LABEL_CACHE.update(labels)
    for split,group in groups.items():
        for video in group:sims[video]=Simulator(video,thresholds)
        results[split]={}
        for arm in ARMS:
            if arm in ['B0','GEO','THRESHOLD']:
                name={'B0':'B0','GEO':'U_GEO','THRESHOLD':'U_THRESHOLD'}[arm];source=PREVIOUS/'outcomes'/(split+'_'+name+'.json')
                value=read(source);assert sha(source)==read(PREVIOUS/'comparison_report.json')['outputs'][source.relative_to(ROOT).as_posix()]
                result=dict(value,arm=arm,reused_from=source.relative_to(ROOT).as_posix(),reused_sha256=sha(source))
            else:result=outcome(split,arm,.2,0,group,sims,strata,detail=True)
            assert result['totals']['N']==(323 if split=='fit20' else 641)
            results[split][arm]=result;path=RUN/'outcomes'/(split+'_'+arm+'.json');write(path,result,new=True);outputs[path.relative_to(ROOT).as_posix()]=sha(path)
            status('own_verification_frozen_evaluation',split=split,arm=arm,totals=result['totals'],guard=obs.read())
        assert sha(RUN/'selection.json')==sealed
    primary=[(system,'B0'),('SUPPORT','B0')]+[('FLYVIS',a) for a in ARMS if a!='FLYVIS'];comparisons={}
    for a,b in primary:
        key=a+'-'+b
        if key in comparisons:continue
        A=results['evaluate26'][a];B=results['evaluate26'][b];value=intervals(A,B);value.pop('ci99_375')
        videos=sorted(A['per_video']);d=np.array([A['per_video'][v].get('J',0)-B['per_video'][v].get('J',0) for v in videos]);n=np.array([A['per_video'][v]['N'] for v in videos])
        ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)));sample=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1))
        value['ci_corrected']=np.percentile(sample,[2.5/11,100-2.5/11]).tolist();comparisons[key]=value
    def qualifies(arm):
        return comparisons[arm+'-B0']['ci95'][0]>0 and results['evaluate26'][arm]['totals'].get('wrong',0)<=results['evaluate26']['B0']['totals'].get('wrong',0)
    go=qualifies(system) or qualifies('SUPPORT')
    phase('comparison_phase.json');write(RUN/'comparison_report.json',dict(passed=True,selected_system=system,selection_sha256=sealed,thresholds=thresholds,
        calibration=selection['calibration'],comparisons=comparisons,full_tracker_gate=dict(proceed=bool(go),criterion=protocol['full_gate']),
        summaries={s:{a:{k:r[k] for k in ['totals','other_tracks','first_stage_loss']} for a,r in value.items()} for s,value in results.items()},
        outputs=outputs,reused_complete_updates=sum(s.reused for s in sims.values()),new_complete_updates=sum(s.replayed for s in sims.values()),
        reused_source_manifests={v:s.reuse_sources for v,s in sims.items()},
        scope='First problem only; historical one-step interventions and separately fitted absolute identity gate. Full-tracker benefit and independent audit are still required.'),new=True)
    for sim in sims.values():sim.close()


if __name__=='__main__':main(guard())
