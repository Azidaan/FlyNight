import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from types import SimpleNamespace
import numpy as np
from tracking.b2_pilot_runtime import IndependentByteTracker,assert_tracker_state_equal


def test_native_zero_isolated_ids_and_states_survive_interleaved_third_tracker():
    args=SimpleNamespace(track_high_thresh=.5,track_low_thresh=.1,new_track_thresh=.6,track_buffer=30,match_thresh=.8,fuse_score=True)
    native=IndependentByteTracker(args);zero=IndependentByteTracker(args);other=IndependentByteTracker(args)
    for f in range(50):
        boxes=np.array([[f,10,f+20,30]],np.float32) if f%11 else np.empty((0,4),np.float32)
        scores=np.full(len(boxes),.9,np.float32);classes=np.zeros(len(boxes),np.int16)
        a=native.step(boxes,scores,classes)
        other.step(np.array([[300,300,330,340]],np.float32),np.array([.95],np.float32),np.array([1],np.int16))
        b=zero.step(boxes.copy(),scores.copy(),classes.copy())
        assert np.array_equal(a,b)
        assert_tracker_state_equal(native,zero)
