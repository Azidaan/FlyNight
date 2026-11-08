"""Reuse complete updates, changing only the own-template gate on free edges."""
import numpy as np
from research_singleton_verify_20261108.common import *
from research_singleton_free_20261108.simulate import Simulator as Parent,FreeReplay
from research_identity_assignment_run_20261108.simulate_recovery1 import match_key,native_result
import research_identity_assignment_run_20261108.replay as replay
from tracking.free_singleton_assignment import free_singleton_matches
from tracking.own_template_verification import passes


class Simulator(Parent):
    def __init__(self,video,thresholds):
        super().__init__(video);self.thresholds=thresholds;self.directory=RUN/'counterfactual'/video;self.directory.mkdir(parents=True,exist_ok=True)
        self.own={}
        path=RUN/'scores'/(video+'.json')
        if path.exists():
            assert sha(path)==read(RUN/'scores_complete.json')['videos'][video];self.own=read(path)['frames']
        for source in sorted((PREVIOUS/'counterfactual'/video).glob('*.json')):
            previous=read(source);assert previous['sources']['export']==self.export['matrices_sha256'] and previous['sources']['scores']==sha(BASE/'scores'/(video+'.json'))
            self.reuse_sources[source.relative_to(ROOT).as_posix()]=sha(source)
            for frame,result in previous['frames'].items():
                key=(frame,match_key(result['first']))
                if key in self.memo:assert self.memo[key]['final']==result['final'] and self.memo[key]['all_final']==result['all_final'] and self.memo[key]['output']==result['output']
                else:self.memo[key]=result

    def evaluate(self,arm,alpha,beta,frames):
        assert alpha==.2 and beta==0 and arm in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']
        dest=self.directory/(arm+'.json')
        if dest.exists():
            result=read(dest);assert set(result['frames'])==set(map(str,frames)) and result['threshold']==self.thresholds.get(arm,-1);return result
        outputs={};first_changes=0
        for frame in map(str,frames):
            snap=self.export['snapshots'][frame];C=self.matrices['cost_'+frame];native=np.array(snap['native_matches'],int).reshape(-1,2)
            rr=sorted(set(range(C.shape[0]))-set(native[:,0]));cc=sorted(set(range(C.shape[1]))-set(native[:,1]))
            accepted=[]
            for e in self.own.get(frame,{}).values():
                if e['free'] and passes(e['scores'][arm],self.thresholds.get(arm,-1)):accepted.append([e['row'],e['column']])
            plan=dict(votes=accepted,columns={},_native_cost=C)
            matches,_,_,_,_=free_singleton_matches(C,native,rr,cc,plan,.2,0,'U_GEO',
                [r['state'] for r in snap['rows']],[r['cls'] for r in snap['rows']],snap['classes'],.8)
            key=(frame,match_key(matches));first_changes+=int(key[1]!=match_key(native))
            if key in self.memo:result=self.memo[key];self.reused+=1
            else:
                original=replay.ReplayTracker;replay.ReplayTracker=FreeReplay
                try:result=replay.apply_state(self.blob(frame),self.cfg,self.inputs(frame),snap,plan,.2,0,'U_GEO')
                finally:replay.ReplayTracker=original
                assert match_key(result['first'])==key[1];self.memo[key]=result;self.replayed+=1
            assert set(map(tuple,native)).issubset(set(map(tuple,result['first'])));outputs[frame]=result
        result=dict(video=self.video,arm=arm,alpha=.2,beta=0,threshold=self.thresholds.get(arm,-1),frames=outputs,changed_first_stage_frames=first_changes,
            sources=dict(export=self.export['matrices_sha256'],own_scores=sha(RUN/'scores'/(self.video+'.json')) if self.own else None))
        write(dest,result,new=True);return result
