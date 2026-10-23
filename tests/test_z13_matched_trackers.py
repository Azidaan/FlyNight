import numpy as np
from scripts.z13_matched_trackers_20261024 import configurations,make_tracker,ARMS
from tracking.offline_association import _DetSlice


def test_native_settings_and_common_inputs():
    cfg=configurations()
    for arm,c in cfg.items():
        for k,v in dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,fuse_score=True).items():assert c[k]==v
        if arm in ('botsort','tracktrack'):assert c['with_reid'] is False
        assert c['match_thresh']>0


def test_native_replay_resets_ids_and_does_not_mutate_inputs():
    cfg=configurations();image=np.zeros((120,160,3),np.uint8)
    boxes=np.array([[30,30,60,70]],np.float32);scores=np.array([.9],np.float32);classes=np.array([2],np.float32)
    for arm in ARMS:
        runs=[]
        for repeat in range(2):
            t=make_tracker(arm,cfg);answers=[]
            for frame in range(5):
                b=boxes+np.array([frame,0,frame,0],np.float32);before=b.copy();out=t.update(_DetSlice(b,scores,classes),img=image)
                np.testing.assert_array_equal(before,b);np.testing.assert_array_equal(scores,np.array([.9],np.float32));assert np.isfinite(out).all();answers.append(out.copy())
            runs.append(answers)
        for a,b in zip(*runs):np.testing.assert_array_equal(a,b)
