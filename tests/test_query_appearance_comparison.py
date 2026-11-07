import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from types import SimpleNamespace
import numpy as np
from tracking.query_appearance_comparison import QueryAppearanceTracker,assert_states_equal
from tracking.frame_masa_appearance import FrameMasaServer
from tracking.offline_association import _DetSlice

class Server(FrameMasaServer):
    def features(self,tokens,null_seed=None):
        for token in tokens:
            if token.vector is None:
                assert token.frame==self.frame
                x=np.array([token.crop[0]+1,token.crop[1]+1,1],np.float32)
                token.vector=x/np.linalg.norm(x)
        return np.stack([t.vector for t in tokens])

def test_query_only_shadow_is_exact_native_and_history_strictly_past():
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    server=Server(None)
    zero=QueryAppearanceTracker(args,server,.1,True)
    native=QueryAppearanceTracker(args,None,0)
    for frame in range(50):
        boxes=np.array([[10+frame*.1,10,40+frame*.1,40],[15+frame*.1,10,45+frame*.1,40]],np.float32)
        detections=_DetSlice(boxes,np.array([.9,.8],np.float32),np.zeros(2,np.float32))
        active=frame%5==0
        if active:server.start_frame(frame,np.zeros((100,100,3),np.uint8))
        left=native.update_query_frame(detections,active)
        right=zero.update_query_frame(detections,active)
        if active:server.finish_frame()
        assert np.array_equal(left,right);assert_states_equal(native,zero)
    assert zero.shadow_pairs>0
    assert all(e['frame']%5==0 and all(f<e['frame'] for f in e['past_frames']) for e in zero.events)
