"""Native matches stay locked; reuse every existing identical complete first assignment."""
import numpy as np
from research_singleton_free_20261108.common import *
from research_identity_assignment_run_20261108.simulate_recovery1 import VideoSimulator as Parent,match_key,native_result
import research_identity_assignment_run_20261108.replay as replay
from tracking.free_singleton_assignment import FreeSingletonAssociation,free_singleton_matches
from ultralytics.trackers.utils import matching


class FreeReplay(FreeSingletonAssociation,replay.ReplayTracker):pass


class Simulator(Parent):
    def __init__(self,video):
        super().__init__(video)
        state_path=ROOT/read(BASE/'export_complete.json')['videos'][video]['path']
        assert sha(state_path.with_name(video+'.states.zip'))==self.export['states_sha256']
        self.directory=RUN/'counterfactual'/video;self.directory.mkdir(parents=True,exist_ok=True)
        self.memo={};self.reuse_sources={};self.reused=0
        for path in sorted((BASE/'counterfactual'/video).glob('*.json')):
            value=read(path)
            assert value['sources']['export']==self.export['matrices_sha256'] and value['sources']['scores']==sha(BASE/'scores'/(video+'.json'))
            self.reuse_sources[path.relative_to(ROOT).as_posix()]=sha(path)
            for frame,result in value['frames'].items():
                key=(str(frame),match_key(result['first']))
                if key in self.memo:
                    old=self.memo[key];assert old['final']==result['final'] and old['all_final']==result['all_final'] and old['output']==result['output']
                else:self.memo[key]=result
        for frame,snap in self.export['snapshots'].items():
            key=(str(frame),match_key(snap['native_matches']))
            self.memo.setdefault(key,native_result(snap))

    def evaluate(self,arm,alpha,beta,frames):
        dest=self.directory/f'{arm}_a{alpha:g}_b{beta:g}.json'
        if dest.exists():
            result=read(dest);assert set(result['frames'])==set(map(str,frames));return result
        output={};changed=0
        for frame in map(str,frames):
            snap=self.export['snapshots'][frame];C=self.matrices['cost_'+frame];plan=dict(self.score['frames'][frame]);plan['_native_cost']=C
            native=np.asarray(snap['native_matches'],int).reshape(-1,2)
            rr=sorted(set(range(C.shape[0]))-set(native[:,0]));cc=sorted(set(range(C.shape[1]))-set(native[:,1]))
            if alpha==0 and beta==0:matches=native
            else:
                matches,_,_,_,_=free_singleton_matches(C,native,rr,cc,plan,alpha,beta,arm,
                    [r['state'] for r in snap['rows']],[r['cls'] for r in snap['rows']],snap['classes'],self.cfg['match_thresh'])
            key=(frame,match_key(matches));changed+=int(key[1]!=match_key(native))
            if key in self.memo:result=self.memo[key];self.reused+=1
            else:
                original=replay.ReplayTracker;replay.ReplayTracker=FreeReplay
                try:result=replay.apply_state(self.blob(frame),self.cfg,self.inputs(frame),snap,plan,alpha,beta,arm)
                finally:replay.ReplayTracker=original
                assert match_key(result['first'])==key[1]
                self.memo[key]=result;self.replayed+=1
            assert set(map(tuple,native)).issubset(set(map(tuple,result['first'])))
            output[frame]=result
        result=dict(video=self.video,arm=arm,alpha=alpha,beta=beta,frames=output,changed_first_stage_frames=changed,
            sources=dict(export=self.export['matrices_sha256'],scores=sha(BASE/'scores'/(self.video+'.json'))))
        write(dest,result,new=True);return result
