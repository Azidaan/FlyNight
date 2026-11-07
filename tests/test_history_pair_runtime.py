import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from types import SimpleNamespace
import numpy as np
import torch
from tracking.history_pair_runtime import HistoryAssociationTracker
from tracking.history_pair_model import HistoryPairHead
from tracking.xywh_bytetrack import XYWHByteTrack
from tracking.offline_association import _DetSlice
from ultralytics.trackers.basetrack import BaseTrack


def test_zero_shadow_matches_native_and_counters_are_isolated():
    torch.set_num_threads(2)
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,
        track_buffer=30,match_thresh=.8,fuse_score=True)
    torch.manual_seed(42);shadow=HistoryAssociationTracker(args,HistoryPairHead(),'134',0)
    native=XYWHByteTrack(args);native_count=0
    for frame in range(50):
        boxes=np.array([[10+frame*.2,10,30+frame*.2,30],[50-frame*.2,10,70-frame*.2,30]],np.float32)
        scores=np.array([.9,.8],np.float32);classes=np.zeros(2,np.float32)
        context=None
        if frame%5==0:
            rois={d:dict(embedding=np.eye(1,256,d,dtype=np.float32)[0],image=np.zeros((3,64,64),np.uint8)) for d in range(2)}
            context=dict(rois=rois,camera=np.zeros(2,np.float32),epoch=0,video_size=np.array([320,180]))
        saved=BaseTrack._count;BaseTrack._count=native_count
        expected=np.asarray(native.update(_DetSlice(boxes,scores,classes)),np.float32).reshape(-1,8)
        native_count=BaseTrack._count;BaseTrack._count=saved
        actual=shadow.update_frame(frame,boxes,scores,classes,context)
        assert np.array_equal(actual,expected)
        for a,b in zip(shadow.tracked_stracks,native.tracked_stracks):
            assert np.array_equal(a.mean,b.mean) and np.array_equal(a.covariance,b.covariance)
    assert shadow.shadow_pairs>0 and all(e['original']==e['corrected'] for e in shadow.events)
    assert all(all(t<e['frame'] for t in e['history_frames']) for e in shadow.events)
