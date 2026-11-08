"""CPU-only causal recency variant; same frozen geometry and native lifecycle."""
import os,sys,gzip,shutil
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_singleton_age_20261108.common import *


def raw_path(p,arm,video):
    return ROOT/p['videos'][video]['raw_array'] if arm=='B0' else FULL/'rollouts'/arm/(video+'.npz')


def reused_metric(video,arm):
    import numpy as np
    with np.load(raw_path(read(FULL/'protocol.json'),arm,video),allow_pickle=False) as z:actual=z['outputs']
    p=read(OWN_FULL/'protocol.json')
    for previous in p['parameters']:
        source=p['aliases'].get(previous,previous);oldraw=ROOT/p['videos'][video]['raw_array'] if source=='B0' else OWN_FULL/'rollouts'/source/(video+'.npz')
        with np.load(oldraw,allow_pickle=False) as z:same=np.array_equal(actual,z['outputs'])
        if same and read(FULL/'outputs'/arm/(video+'.json'))==read(OWN_FULL/'outputs'/previous/(video+'.json')):
            old=OWN_FULL/'metrics'/previous/(video+'.json');m=read(old);details=old.with_suffix('.details.json.gz');assert m['passed'] and m['details_sha256']==sha(details)
            dest=FULL/'metrics'/arm;dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(details,dest/details.name)
            write(dest/old.name,dict(m,arm=arm,output_sha256=sha(FULL/'outputs'/arm/(video+'.json')),reused_metric_source=old.relative_to(ROOT).as_posix(),identical_output_arrays_verified=True),new=True)
            return previous
    return None


def main(obs):
    import numpy as np
    phase('full_phase.json');assert read(RUN/'independent_audit.json')['passed'];cap=read(RUN/'selection.json')['cap'];p=read(FULL/'protocol.json')
    import research_singleton_verify_full_20261108.engine as engine
    import research_singleton_free_20261108.full_paths as paths
    engine.FULL=FULL;engine.RUN=RUN;engine.status=status;paths.FULL=FULL;Parent=engine.Planner
    class Planner(Parent):
        def __init__(self,video,kind,threshold,representations=None):
            super().__init__(video,'SUPPORT','off' if threshold=='off' else -1,None)
        def provide(self,tracker,pool,detections,cost):
            plan=super().provide(tracker,pool,detections,cost);votes=[]
            for decision in self.last['decisions']:
                row=self.last['rows'][decision['row']];gap=self.frame-row['history'][-1]['frame'];decision.update(score=-gap,threshold=-cap,gap=gap)
                decision['accepted']=bool(decision['support']>=3 and gap<=cap)
                if decision['accepted']:votes.append([decision['row'],decision['column']])
            plan=dict(votes=votes,columns={});self.last['plan']=plan;return plan
    engine.Planner=Planner
    first=next(iter(p['videos']));r=engine.rollout(first,'ZERO_AGE',obs,zero=True);assert r['zero'] and not r['core_imported']
    write(FULL/'zero_parity.json',dict(passed=True,video=first,raw_and_annotation_outputs_exact=True,receipt_sha256=sha(FULL/'rollouts/ZERO_AGE'/(first+'.receipt.json'))),new=True)
    import research_identity_assignment_run_20261108.full_evaluate as evaluate
    evaluate.FULL=FULL;evaluate.status=status
    import research_identity_assignment_run_20261108.full_precision_manager as bridge
    bridge.FULL=FULL;bridge.raw_path=raw_path
    from research_identity_assignment_run_20261108.full_additional import measure
    totals=Counter();sources={}
    for index,video in enumerate(p['videos']):
        receipt=engine.rollout(video,'AGE',obs);assert receipt['passed'] and not receipt['core_imported'] and not receipt['feature_files']
        previous=reused_metric(video,'AGE')
        if previous is None:
            old=sys.argv
            try:sys.argv=['age_evaluate',video,'AGE','--parent-report',obs.read()['supervisor_report']];evaluate.main()
            finally:sys.argv=old
        if previous is not None:
            old=OWN_FULL/'additional'/previous/(video+'.json');result=read(old)
            assert result['clock_sha256']==sha(FULL/'runtime_inputs'/(video+'.json'))
            result=dict(result,arm='AGE',raw_sha256=sha(raw_path(p,'AGE',video)),same_output_array_verified=True,reused_from=old.relative_to(ROOT).as_posix())
        else:result=measure(video,'AGE',p)
        target=FULL/'additional/AGE'/(video+'.json');write(target,result,new=True);totals.update(result['totals']);sources[target.relative_to(ROOT).as_posix()]=sha(target)
        artifacts={n.relative_to(ROOT).as_posix():sha(n) for n in [FULL/'outputs/AGE'/(video+'.json'),FULL/'metrics/AGE'/(video+'.json'),FULL/'metrics/AGE'/(video+'.details.json.gz'),target]}
        write(FULL/'completed'/(video+'.json'),dict(passed=True,artifacts=artifacts),new=True);phase('full_phase.json');status('recency_full_video_complete',video=video,complete=index+1,total=len(p['videos']),cap=cap,guard=obs.read())
    arms={a:read(OWN_FULL/'additional_report.json')['arms'][a] for a in read(OWN_FULL/'protocol.json')['parameters']};arms['AGE']=dict(totals=dict(totals),sources=sources)
    write(FULL/'additional_report.json',dict(passed=True,arms=arms,original_report_sha256=sha(OWN_FULL/'additional_report.json')),new=True)
    write(FULL/'rollouts_complete.json',dict(passed=True,videos=len(p['videos']),arm='AGE',core_imported=False,cap=cap),new=True);phase('full_phase.json')


if __name__=='__main__':main(guard())
