"""Recover only the summary after a zero-known-query video, preserving all work."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_20261108.common import *


def main(obs):
    import numpy as np
    from research_identity_assignment_run_20261108.compare_recovery1 import intervals
    phase('comparison_phase.json');phase('summary_recovery1_phase.json');selection=read(RUN/'selection.json');protocol=read(RUN/'protocol.json')
    arms=['B0','GEO','THRESHOLD','SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM'];results={};outputs={}
    for split in ['fit20','evaluate26']:
        results[split]={}
        for arm in arms:
            path=RUN/'outcomes'/(split+'_'+arm+'.json');value=read(path)
            assert value['arm']==arm and value['totals']['N']==(323 if split=='fit20' else 641)
            results[split][arm]=value;outputs[path.relative_to(ROOT).as_posix()]=sha(path)
    system=selection['selected_system'];primary=[(system,'B0'),('SUPPORT','B0')]+[('FLYVIS',a) for a in arms if a!='FLYVIS'];comparisons={}
    for a,b in primary:
        key=a+'-'+b
        if key in comparisons:continue
        A=results['evaluate26'][a];B=results['evaluate26'][b];record=intervals(A,B);record.pop('ci99_375')
        videos=sorted(A['per_video']);d=np.array([A['per_video'][v].get('J',0)-B['per_video'][v].get('J',0) for v in videos])
        n=np.array([A['per_video'][v].get('N',0) for v in videos]);ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)))
        values=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1));record['ci_corrected']=np.percentile(values,[2.5/11,100-2.5/11]).tolist();comparisons[key]=record
    def qualifies(a):return comparisons[a+'-B0']['ci95'][0]>0 and results['evaluate26'][a]['totals'].get('wrong',0)<=results['evaluate26']['B0']['totals'].get('wrong',0)
    sources={};frame_signatures={};novel=set()
    old_sources={}
    for name in ['runs/identity_assignment_20261108/counterfactual','runs/singleton_free_20261108/counterfactual']:
        for path in (ROOT/name).glob('*/*.json'):
            value=read(path);video=path.parent.name
            for frame,result in value['frames'].items():
                key=(video,frame,tuple(sorted(map(tuple,result['first']))))
                frame_signatures.setdefault(key,result)
            old_sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in (RUN/'counterfactual').glob('*/*.json'):
        value=read(path);video=path.parent.name
        for frame,result in value['frames'].items():
            key=(video,frame,tuple(sorted(map(tuple,result['first']))))
            if key in frame_signatures:
                old=frame_signatures[key];assert old['final']==result['final'] and old['all_final']==result['all_final'] and old['output']==result['output']
            else:novel.add(key)
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    report=dict(passed=True,selected_system=system,selection_sha256=sha(RUN/'selection.json'),thresholds=selection['thresholds'],calibration=selection['calibration'],
        comparisons=comparisons,full_tracker_gate=dict(proceed=bool(qualifies(system) or qualifies('SUPPORT')),criterion=protocol['full_gate']),
        summaries={s:{a:{k:r[k] for k in ['totals','other_tracks','first_stage_loss']} for a,r in values.items()} for s,values in results.items()},outputs=outputs,
        recovered_summary_only=True,new_distinct_first_match_sets=len(novel),
        reused_source_manifests=dict(previous_complete_updates=old_sources,new_counterfactuals=sources),
        scope='First problem only; fixed fit20selection, historical one-step outcomes. MissingNfor a video with no known queries is interpreted as0. No maps, selections or tracker updates repeated.')
    phase('summary_recovery1_phase.json');write(RUN/'comparison_report.json',report,new=True)
    status('own_template_summary_recovered',full_tracker_gate=report['full_tracker_gate'])


if __name__=='__main__':main(guard())
