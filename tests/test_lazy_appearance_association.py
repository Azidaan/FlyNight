import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from types import SimpleNamespace

def test_disabled_full_sequence_and_state_equal_native():
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.lazy_appearance_association import LazyAppearanceXYWH
    from tracking.offline_association import _DetSlice
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    rng=np.random.default_rng(42);frames=[]
    for frame in range(100):
        b=np.array([[10+frame,20,50+frame,60],[200-frame,20,240-frame,60]],np.float32)
        keep=rng.random(2)>.15;frames.append(_DetSlice(b[keep],np.full(keep.sum(),.9,np.float32),np.zeros(keep.sum(),np.float32)))
    ref=XYWHByteTrack(args);expected=[]
    for d in frames:
        out=ref.update(d);expected.append((out.copy(),[(t.track_id,t.mean.copy(),t.covariance.copy()) for t in ref.tracked_stracks]))
    actual=LazyAppearanceXYWH(args,weight=0)
    for d,(out,state) in zip(frames,expected):
        assert np.array_equal(actual.update(d),out)
        assert len(state)==len(actual.tracked_stracks)
        for t,(ident,mean,cov) in zip(actual.tracked_stracks,state):assert t.track_id==ident and np.array_equal(t.mean,mean) and np.array_equal(t.covariance,cov)
    assert not actual.events

def test_appearance_changes_cost_only_on_existing_edges_and_uses_past():
    from tracking.lazy_appearance_association import LazyAppearanceXYWH,Observation
    class Server:
        def features(self,tokens,null_seed):return np.stack([t.vector for t in tokens])
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=False)
    tracker=LazyAppearanceXYWH(args,Server(),weight=.3);tracker.frame_id=2;tracker._inside_first=True
    def item(ident,frame,z):return SimpleNamespace(track_id=ident,xyxy=np.array([0,0,10,10],np.float32),angle=None,score=.9,observation=Observation((ident,),frame,None,np.array(z,np.float32)))
    old=[item(1,0,[1,0]),item(2,0,[0,1])];new=[item(3,1,[1,0]),item(4,1,[0,1])]
    cost=tracker.get_dists(old,new);assert cost[0,0]==0 and cost[1,1]==0 and cost[0,1]>0 and cost[1,0]>0
    assert tracker.events[0]['past_frames']==[0,0]
