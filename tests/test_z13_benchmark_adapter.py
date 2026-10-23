import numpy as np
from scripts.z13_matched_trackers_20261024 import configurations,make_tracker,ARMS
from tracking.z13_benchmark_adapter import TrackerView,xywh_state
from tracking.offline_association import _DetSlice


def test_view_does_not_change_native_outputs_or_states():
    cfg=configurations();image=np.zeros((120,160,3),np.uint8)
    for arm in ARMS:
        outputs=[]
        for wrapped in (False,True):
            native=make_tracker(arm,cfg);tracker=TrackerView(native) if wrapped else native;rows=[]
            for f in range(6):
                boxes=np.array([[30+f,30,60+f,70]],np.float32);out=tracker.update(_DetSlice(boxes,np.array([.9]),np.array([2])),img=image);rows.append(out.copy())
                for t in native.tracked_stracks:
                    before=t.mean.copy();view=xywh_state(t);np.testing.assert_array_equal(t.mean,before)
                    # Current box matches the native box, independent of XYAH/XYWH.
                    np.testing.assert_allclose(view[:2]-view[2:4]/2,t.tlwh[:2],atol=1e-10)
                    np.testing.assert_allclose(view[2:4],t.tlwh[2:4],atol=1e-10)
                    nextm,_=t.kalman_filter.predict(t.mean.copy(),t.covariance.copy())
                    expected=nextm[:4].copy()
                    if type(t.kalman_filter).__name__=='KalmanFilterXYAH':expected[2]*=expected[3]
                    np.testing.assert_allclose(view[:4]+view[4:],expected,atol=1e-10)
            outputs.append(rows)
        for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)
