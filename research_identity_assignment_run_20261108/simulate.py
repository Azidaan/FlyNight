"""Label-free one-step interventions, with complete native lifecycle and memoization."""
import zipfile
import numpy as np
from ultralytics.trackers.utils import matching
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.replay import apply_state
from tracking.spatial_identity_assignment import corrected_cost


def match_key(pairs):return ';'.join(f'{i},{j}' for i,j in sorted(map(tuple,pairs)))


def native_result(snap):
    emitted={str(int(row[4])) for row in snap['native_output']}
    return dict(output=snap['native_output'],all_final=snap['native_final_assignments'],final={k:v for k,v in snap['native_final_assignments'].items() if k in emitted},
                first=snap['native_matches'],duplicate_events=snap.get('native_duplicate_events',[]))


class VideoSimulator:
    def __init__(self,video):
        self.video=video;entry=read(RUN/'export_complete.json')['videos'][video];p=ROOT/entry['path']
        assert sha(p)==entry['sha256'];self.export=read(p)
        assert sha(p.with_suffix('.npz'))==self.export['matrices_sha256']
        self.matrices=np.load(p.with_suffix('.npz'),allow_pickle=False)
        self.states=zipfile.ZipFile(p.with_name(video+'.states.zip'))
        self.cfg=read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json')['native_config']
        self.score=read(RUN/'scores'/(video+'.json'))
        assert sha(RUN/'scores'/(video+'.json'))==read(RUN/'scores_complete.json')['videos'][video]
        raw=ROOT/f'runs/train_cache/{video}/detections/raw_video.npz'
        assert sha(raw)==self.export['sources'][raw.relative_to(ROOT).as_posix()]
        with np.load(raw,allow_pickle=False) as z:self.raw={n:z[n] for n in z.files}
        self.directory=RUN/'counterfactual'/video;self.directory.mkdir(parents=True,exist_ok=True)
        self.memo={};self.blobs={};self.native_verified=0;self.replayed=0

    def close(self):self.states.close();self.matrices.close()

    def inputs(self,frame):
        ids=np.flatnonzero(self.raw['frame_idx']==int(frame))
        return tuple(self.raw[n][ids] for n in ['xyxy','conf','cls'])+(ids,)

    def blob(self,frame):
        if frame not in self.blobs:self.blobs[frame]=self.states.read(str(frame)+'.pkl')
        return self.blobs[frame]

    def verify_native(self,frame):
        snap=self.export['snapshots'][str(frame)];native=native_result(snap)
        result=apply_state(self.blob(frame),self.cfg,self.inputs(frame),snap,{},0,0,'B0')
        assert np.array_equal(np.asarray(result['output']),np.asarray(native['output'])),(self.video,frame,'output restore')
        assert result['final']==native['final'],(self.video,frame,'assigned detection restore',result['final'],native['final'])
        assert result['all_final']==native['all_final'],(self.video,frame,'unconfirmed births restore')
        snap['native_duplicate_events']=result['duplicate_events'];native['duplicate_events']=result['duplicate_events']
        self.memo[(str(frame),match_key(native['first']))]=native;self.native_verified+=1

    def evaluate(self,arm,alpha,beta,frames):
        name=f'{arm}_a{alpha:g}_b{beta:g}.json';dest=self.directory/name
        if dest.exists():
            result=read(dest);assert set(result['frames'])==set(map(str,frames));return result
        output={};changed=0
        for frame in frames:
            frame=str(frame);snap=self.export['snapshots'][frame];plan=dict(self.score['frames'][frame])
            C=self.matrices['cost_'+frame];plan['_native_cost']=C
            CZ=corrected_cost(C,plan['votes'],alpha,beta,arm,plan['columns'])
            if np.array_equal(CZ,C):matches=snap['native_matches']
            else:matches=np.asarray(matching.linear_assignment(CZ,thresh=self.cfg['match_thresh'])[0]).reshape(-1,2).tolist()
            key=match_key(matches);original=match_key(snap['native_matches']);memo_key=(frame,key)
            if key==original:result=native_result(snap)
            elif memo_key in self.memo:result=self.memo[memo_key]
            else:
                result=apply_state(self.blob(frame),self.cfg,self.inputs(frame),snap,plan,alpha,beta,arm)
                assert match_key(result['first'])==key,(self.video,frame,'runtime assignment')
                self.memo[memo_key]=result;self.replayed+=1
            changed+=int(key!=original);output[frame]=result
        result=dict(video=self.video,arm=arm,alpha=alpha,beta=beta,frames=output,changed_first_stage_frames=changed,
                    sources=dict(export=self.export['matrices_sha256'],scores=sha(RUN/'scores'/(self.video+'.json'))))
        write(dest,result,new=True);return result
