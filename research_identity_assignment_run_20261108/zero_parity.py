"""First gate: original native tracker vs every zero-intervention new arm."""
import os,sys,hashlib,time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    import numpy as np
    import torch
    from ultralytics.trackers.basetrack import BaseTrack
    from tracking.offline_association import _DetSlice,make_tracker_args
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.spatial_identity_assignment import SpatialIdentityByteTrack,corrected_cost
    torch.set_num_threads(2);phase=verify_phase('zero_phase.json');video=phase['video']
    raw=ROOT/f'runs/train_cache/{video}/detections/raw_video.npz'
    with np.load(raw,allow_pickle=False) as z:b=z['xyxy'];s=z['conf'];c=z['cls'];f=z['frame_idx']
    order=np.argsort(f,kind='stable');frames=f[order];total=int(f.max())+1
    # The source-native baseline is executed only once; all zero arms share identical
    # code path, so one complete modified rollout plus per-arm constructor checks suffice.
    baseline=[];baseline_states=[];outputs=[]
    def state(t):
        pools={name:sorted(getattr(t,name),key=lambda x:int(x.track_id)) for name in ['tracked_stracks','lost_stracks','removed_stracks']}
        value=[]
        for name,tracks in pools.items():
            for x in tracks:
                value.append((name,int(x.track_id),int(x.state),int(x.frame_id),int(x.start_frame),bool(x.is_activated),
                    np.asarray(x.mean).tobytes(),np.asarray(x.covariance).tobytes()))
        return hashlib.sha256(repr(value).encode()).hexdigest()
    began=time.perf_counter()
    for mode in ['native','zero']:
        BaseTrack.reset_id()
        args=make_tracker_args(**read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json')['native_config'])
        tracker=XYWHByteTrack(args) if mode=='native' else SpatialIdentityByteTrack(args,alpha=0,beta=0,
            provider=lambda *a:(_ for _ in ()).throw(AssertionError('Disabled provider called')))
        for frame in range(total):
            lo,hi=np.searchsorted(frames,[frame,frame+1]);ix=order[lo:hi]
            result=tracker.update(_DetSlice(b[ix],s[ix],c[ix]));fingerprint=state(tracker)
            if mode=='native':baseline.append(result.copy());baseline_states.append(fingerprint)
            else:assert np.array_equal(result,baseline[frame]) and fingerprint==baseline_states[frame],(video,frame)
            if frame%100==0:status('zero_parity',mode=mode,frame=frame,total=total,guard=obs.read())
        if mode=='zero':assert not tracker.identity_events
    base=np.array([[.64,.98],[.43,.7]],np.float32)
    for arm in ['G','G_SHIFT','G_IMAGE','G_R1','G_DIS','G_FLYVIS','G_PERM']:
        assert np.array_equal(corrected_cost(base,[(0,0)],0,0,arm,{}),base)
    # Preserve the complete new baseline output, to avoid its recalculation later.
    offsets=np.r_[0,np.cumsum([len(x) for x in baseline])]
    target=RUN/'zero_native_outputs.npz'
    with target.open('xb') as stream:np.savez_compressed(stream,outputs=np.concatenate(baseline),offsets=offsets)
    verify_phase('zero_phase.json');assert not torch.cuda.is_initialized()
    write(RUN/'zero_parity.json',dict(passed=True,video=video,frames=total,raw_outputs_bitwise_equal=True,
        all_native_states_bitwise_equal=True,zero_cost_arms=7,provider_calls=0,
        baseline_outputs_sha256=sha(target),seconds=time.perf_counter()-began),new=True)


if __name__=='__main__':main(guard())
