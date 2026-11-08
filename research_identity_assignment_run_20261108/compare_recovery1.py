"""Fit20 selection, frozen evaluate26 comparison and predeclared next-stage gate."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.simulate_recovery1 import VideoSimulator,native_result
from research_identity_assignment_run_20261108.evaluator import Labels

ARMS=['B0','G','G_SHIFT','G_IMAGE','G_R1','G_DIS','G_FLYVIS','G_PERM']
GRID=[0,.1,.2,.4,.8]
LABEL_CACHE={}


def binding_changes(native,changed):
    return sum(native['final'].get(t)!=changed['final'].get(t) for t in set(native['final'])|set(changed['final']))


def outcome(split,arm,alpha,beta,groups,simulators,strata,*,detail=False):
    total=Counter();per_video={};rows=[];harms=Counter();sources={};first_lost=Counter()
    for video,queries in groups.items():
        sim=simulators[video];frames=sorted({q['frame'] for q in queries})
        if arm=='B0':outputs={str(f):native_result(sim.export['snapshots'][str(f)]) for f in frames}
        else:outputs=sim.evaluate(arm,alpha,beta,frames)['frames']
        if video not in LABEL_CACHE:LABEL_CACHE[video]=Labels(video,sim.export)
        lab=LABEL_CACHE[video];vc=Counter();querytids=defaultdict(set)
        for q in queries:querytids[q['frame']].add(q['tid'])
        for frame in frames:
            base=native_result(sim.export['snapshots'][str(frame)]);result=outputs[str(frame)]
            total['changed_native_bindings']+=binding_changes(base,result)
            harms.update(lab.harm(frame,base,result,querytids[frame]))
        for q in queries:
            result=outputs[str(q['frame'])];snap=sim.export['snapshots'][str(q['frame'])]
            row_index=next(i for i,r in enumerate(snap['rows']) if r['tid']==q['tid']);row=snap['rows'][row_index]
            measured=lab.query(q,result);baseline=lab.query(q,native_result(snap));stratum=strata[q['uid']]
            eligible=sum(g['valid'] for g in row.get('geometry',[]))
            if stratum['primary_known']:
                vc[measured['kind']]+=1;vc['J']+=measured['J'];vc['N']+=1
                vc['transition_'+baseline['kind']+'_to_'+measured['kind']]+=1
                vc[stratum['reason']+'_N']+=1;vc[stratum['reason']+'_J']+=measured['J']
                if measured['J']>baseline['J']:vc['improved_queries']+=1
                if measured['J']<baseline['J']:vc['worsened_queries']+=1
                if baseline['kind']=='none' and measured['kind']=='correct':vc['none_to_correct']+=1
                if baseline['kind']=='none' and measured['kind']=='wrong':vc['none_to_wrong']+=1
                first=next((j for i,j in result['first'] if i==row_index),None)
                if first is not None and str(q['tid']) not in result['final']:
                    vc['first_stage_assigned_but_not_emitted']+=1
                    first_lost['singleton' if eligible==1 else 'other']+=1
                    if any(e['tracked_id']==q['tid'] and e['tracked_dropped'] for e in result.get('duplicate_events',[])):
                        vc['confirmed_duplicate_removal_after_first_assignment']+=1
            else:vc['shadow_'+measured['kind']]+=1;vc['shadow_N']+=1
            if detail:rows.append(dict(**q,reason=stratum['reason'],primary_known=stratum['primary_known'],
                geometric_candidates=eligible,state=row['state'],outcome=measured,baseline=baseline,
                first_high_index=next((j for i,j in result['first'] if i==row_index),None),
                duplicate_events=[e for e in result.get('duplicate_events',[]) if q['tid'] in [e['tracked_id'],e['lost_id']]]))
        total.update(vc);per_video[video]=dict(vc);sources.update(lab.sources)
    return dict(split=split,arm=arm,alpha=alpha,beta=beta,totals=dict(total),per_video=per_video,
        other_tracks=dict(harms),first_stage_loss=dict(first_lost),rows=rows,label_sources=sources)


def intervals(a,b):
    videos=sorted(set(a['per_video'])|set(b['per_video']));delta=[];den=[]
    for v in videos:
        delta.append(a['per_video'].get(v,{}).get('J',0)-b['per_video'].get(v,{}).get('J',0))
        den.append(a['per_video'].get(v,{}).get('N',0))
    delta=np.array(delta,float);den=np.array(den,float)
    ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)))
    samples=100*delta[ix].sum(1)/np.maximum(1,den[ix].sum(1))
    return dict(delta_J=int(delta.sum()),delta_J_per100=float(100*delta.sum()/den.sum()),
        ci95=np.percentile(samples,[2.5,97.5]).tolist(),ci99_375=np.percentile(samples,[.3125,99.6875]).tolist(),
        videos=len(videos),bootstrap_replicates=20000,seed=42,unit='paired_video')


def main(obs):
    verify_phase('comparison_recovery1_phase.json');assert read(RUN/'scores_complete.json')['passed']
    queries=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows']
    strata={r['uid']:r for r in read(ROOT/'research_identity_assignment_protocol_20261108/evaluation_strata.json')['rows']}
    groups={s:defaultdict(list) for s in ['fit20','evaluate26']}
    for q in queries:groups[q['split']][q['video']].append(q)
    assert len({q['uid'] for q in queries})==len(queries)==1991
    assert len(groups['fit20'])==20 and len(groups['evaluate26'])==26
    assert not set(groups['fit20'])&set(groups['evaluate26'])
    sims={};zero={};exports=read(RUN/'export_complete.json')['videos']
    # Check restored native outputs before any label-based selection or evaluation.
    for number,video in enumerate(exports):
        sim=VideoSimulator(video);sims[video]=sim
        for frame in sim.export['snapshots']:sim.verify_native(frame)
        zero[video]=sim.native_verified
        status('restored_zero_parity',complete=number+1,total=len(exports),frames=sum(zero.values()),guard=obs.read())
    write(RUN/'restored_zero_parity.json',dict(passed=True,frames=sum(zero.values()),videos=zero,
        scope='All exported pre-update states, exact native output arrays and emitted track-to-raw-detection bindings.'),new=True)
    if (RUN/'selection.json').exists():selection=read(RUN/'selection.json')
    else:
        candidates=[]
        for alpha in GRID:
            r=outcome('fit20','G',alpha,0,groups['fit20'],sims,strata)
            candidates.append(r);status('fit20_alpha_selection',alpha=alpha,totals=r['totals'],guard=obs.read())
        best=max(candidates,key=lambda r:(r['totals'].get('J',0),-r['totals']['changed_native_bindings'],-r['alpha']))
        alpha=best['alpha'];selection=dict(alpha=alpha,parameters={'B0':dict(alpha=0,beta=0),'G':dict(alpha=alpha,beta=0)},
            alpha_candidates=[{k:r[k] for k in ['alpha','totals','per_video']} for r in candidates],beta_candidates={})
        for arm in ARMS[2:]:
            candidates=[]
            for beta in GRID:
                r=outcome('fit20',arm,alpha,beta,groups['fit20'],sims,strata);candidates.append(r)
            best=max(candidates,key=lambda r:(r['totals'].get('J',0),-r['totals']['changed_native_bindings'],-r['beta']))
            selection['parameters'][arm]=dict(alpha=alpha,beta=best['beta'])
            selection['beta_candidates'][arm]=[{k:r[k] for k in ['beta','totals','per_video']} for r in candidates]
            status('fit20_beta_selection',arm=arm,beta=best['beta'],totals=best['totals'],guard=obs.read())
        selection.update(frozen=datetime.now(MSK).isoformat(),criterion='J=correct-wrong on fixed323 fit20 known queries; unknown0 explicitly counted.',
            tie='Fewer changed emitted track-to-raw-detection bindings across all selected fit20 frames, then smaller parameter.',
            comparison_phase_sha256=sha(RUN/'comparison_recovery1_phase.json'))
        write(RUN/'selection.json',selection,new=True)
    # Freeze is on disk before opening evaluate26 Labels or outcomes.
    selection_hash=sha(RUN/'selection.json');manifest={};results={}
    for split in ['fit20','evaluate26']:
        results[split]={}
        for arm in ARMS:
            r=outcome(split,arm,**selection['parameters'][arm],groups=groups[split],simulators=sims,strata=strata,detail=True)
            assert r['totals']['N']==(323 if split=='fit20' else 641)
            path=RUN/'outcomes'/f'{split}_{arm}.json';write(path,r,new=True);manifest[path.relative_to(ROOT).as_posix()]=sha(path)
            results[split][arm]=r
            status('frozen_evaluation',split=split,arm=arm,totals=r['totals'],guard=obs.read())
    evaluate=results['evaluate26'];comparisons={}
    for arm in ARMS[1:]:comparisons[arm+'-B0']=intervals(evaluate[arm],evaluate['B0'])
    for arm in ARMS[1:]:
        if arm!='G_FLYVIS':comparisons['G_FLYVIS-'+arm]=intervals(evaluate['G_FLYVIS'],evaluate[arm])
    go=any(comparisons[arm+'-B0']['ci95'][0]>0 for arm in ['G','G_FLYVIS'])
    report=dict(passed=True,selection_sha256=selection_hash,parameters=selection['parameters'],
        summaries={s:{a:{k:r[k] for k in ['totals','other_tracks','first_stage_loss']} for a,r in arms.items()} for s,arms in results.items()},
        evaluate26_comparisons=comparisons,full_tracker_gate=dict(proceed=go,criterion='G or G_FLYVIS paired-video lower95% deltaJ vsB0 >0',
            status='required_next_stage' if go else 'stopped_before_full_tracker'),
        limitations=['One-frame interventions from archived native states, not a closed-loop rollout.',
            'Reused historical train-source cohorts, not independent detector validation.',
            'Sparse unknown assignments neutral and counted; neither-search-candidate is not full-frame absence.',
            'Rival harm uses only consistent identities in four actual historical observations on annotated frames; remaining identities unknown.'],
        outputs=manifest,restored_parity_frames=sum(zero.values()),replayed_distinct_counterfactual_updates=sum(s.replayed for s in sims.values()))
    assert sha(RUN/'selection.json')==selection_hash;verify_phase('comparison_recovery1_phase.json')
    write(RUN/'comparison_report.json',report,new=True)
    for s in sims.values():s.close()
    status('conditional_comparison_complete',full_tracker_gate=report['full_tracker_gate'],guard=obs.read())


if __name__=='__main__':main(guard())
