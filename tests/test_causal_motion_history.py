import cv2,numpy as np,pytest
from tracking.causal_motion_history import CausalMotionHistory

def test_camera_translation_does_not_turn_whole_background_into_motion():
    rng=np.random.default_rng(1);first=rng.integers(0,256,(90,160),dtype=np.uint8)
    second=cv2.warpAffine(first,np.array([[1.,0,4],[0,1.,2]],np.float32),(160,90))
    history=CausalMotionHistory(160,90,threshold=.15)
    history.update(first,0);status=history.update(second,6)
    assert status['reliable'] and np.max(np.abs(status['shift']-[4,2]))<.4
    assert (history.recency[5:-5,6:-6]>.5).mean()<.1

def test_future_or_repeated_frames_cannot_update_memory():
    history=CausalMotionHistory(32,32);history.update(np.zeros((32,32),np.uint8),5)
    with pytest.raises(ValueError):history.update(np.ones((32,32),np.uint8),4)

def test_unknown_camera_resets_motion_and_marks_missing():
    history=CausalMotionHistory(32,32);history.update(np.zeros((32,32),np.uint8),0)
    status=history.update(np.zeros((32,32),np.uint8),6)
    assert not status['reliable'] and not history.recency.any() and not history.valid.any()
    assert history.crop([2,2,10,10],(32,32),8).shape==(3,8,8)
