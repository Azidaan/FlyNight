"""Calibrate on fit20, freeze once, then evaluate historical26 with all controls."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict
from research_singleton_free_20261108.common import *


def main(obs):
    phase();assert read(RUN/'technical.json')['passed']
    from research_singleton_free_20261108.simulate import Simulator
    from research_identity_assignment_run_20261108.compare_recovery1 import outcome,intervals
    queries=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows']
    strata={r['uid']:r for r in read(ROOT/'research_identity_assignment_protocol_20261108/evaluation_strata.json')['rows']}
    groups={s:defaultdict(list) for s in ['fit20','evaluate26']}
    for q in queries:groups[q['split']][q['video']].append(q)
    sims={};grid=[0,.1,.2,.4,.8];p=read(RUN/'protocol.json')
    # Only fit20 is instantiated and labeled until the selection is sealed.
    for video in groups['fit20']:sims[video]=Simulator(video)
    def calculate(split,arm,alpha,beta,detail=False):
        return outcome(split,arm,alpha,beta,groups[split],sims,strata,detail=detail)
    candidates={};parameters={'B0':dict(alpha=0,beta=0)}
    def best(items):return max(items,key=lambda r:(r['totals'].get('J',0),-r['totals']['changed_native_bindings'],-r['alpha'],-r['beta']))
    for arm in ['U_GEO','U_THRESHOLD']:
        values=[calculate('fit20',arm,a,0) for a in grid];selected=best(values)
        parameters[arm]=dict(alpha=selected['alpha'],beta=0)
        candidates[arm]=[{k:r[k] for k in ['alpha','beta','totals','per_video']} for r in values]
    alpha=parameters['U_GEO']['alpha']
    for arm in ARMS[3:]:
        values=[calculate('fit20',arm,alpha,b) for b in grid];selected=best(values)
        parameters[arm]=dict(alpha=alpha,beta=selected['beta'])
        candidates[arm]=[{k:r[k] for k in ['alpha','beta','totals','per_video']} for r in values]
        status('fit20_calibration',arm=arm,parameters=parameters[arm],guard=obs.read())
    selection=dict(parameters=parameters,candidates=candidates,frozen=datetime.now(MSK).isoformat(),criterion=p['calibration'],phase_sha256=sha(RUN/'phase.json'))
    write(RUN/'selection.json',selection,new=True);frozen=sha(RUN/'selection.json')
    for video in groups['evaluate26']:sims[video]=Simulator(video)
    results={};outputs={}
    for split in groups:
        results[split]={}
        for arm in ARMS:
            result=calculate(split,arm,**parameters[arm],detail=True);assert result['totals']['N']==(323 if split=='fit20' else 641)
            results[split][arm]=result;path=RUN/'outcomes'/(split+'_'+arm+'.json');write(path,result,new=True);outputs[path.relative_to(ROOT).as_posix()]=sha(path)
            status('frozen_conditional_evaluation',split=split,arm=arm,totals=result['totals'],guard=obs.read())
    comparisons={a+'-'+b:intervals(results['evaluate26'][a],results['evaluate26'][b]) for a,b in p['primary_comparisons']}
    go=any(comparisons[a+'-B0']['ci95'][0]>0 for a in ['U_GEO','U_THRESHOLD','U_FLYVIS'])
    assert sha(RUN/'selection.json')==frozen;phase()
    report=dict(passed=True,parameters=parameters,selection_sha256=frozen,comparisons=comparisons,
        summaries={s:{a:{k:r[k] for k in ['totals','other_tracks','first_stage_loss']} for a,r in arms.items()} for s,arms in results.items()},
        full_tracker_gate=dict(proceed=go,criterion=p['full_tracker_gate']),outputs=outputs,
        reused_complete_updates=sum(s.reused for s in sims.values()),new_complete_updates=sum(s.replayed for s in sims.values()),
        source_reuse_manifests={v:s.reuse_sources for v,s in sims.items()},
        scope='Historical one-step diagnostic; fixed native first matches; no new images, maps, core inference or GT in runtime. Full tracker acceptance is still required.')
    write(RUN/'comparison_report.json',report,new=True)
    for s in sims.values():s.close()


if __name__=='__main__':main(guard())
