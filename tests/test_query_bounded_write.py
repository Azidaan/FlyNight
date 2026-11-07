from types import SimpleNamespace
import numpy as np
from tracking.query_bounded_write import BoundedWriteTracker,PeriodicWriteTracker
from tracking.query_appearance_comparison import QueryAppearanceTracker,assert_states_equal
from tracking.offline_association import _DetSlice
from tests.test_query_appearance_comparison import Server

def test_bounded_and_periodic_refresh_after_real_assignment_with_original_lifetime():
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    for cls in [BoundedWriteTracker,PeriodicWriteTracker]:
        server=Server(None);shadow=cls(args,server,.1,True);native=QueryAppearanceTracker(args)
        for frame in range(65):
            active=frame%5==0
            if active:server.start_frame(frame,None)
            boxes=np.array([[10+frame*.1,10,40+frame*.1,40],[15+frame*.1,10,45+frame*.1,40]],np.float32)
            d=_DetSlice(boxes,np.array([.9,.8],np.float32),np.zeros(2,np.float32))
            assert np.array_equal(shadow.update_query_frame(d,active),native.update_query_frame(d,active))
            assert_states_equal(shadow,native)
            if active:server.finish_frame()
        refreshed=[e for e in shadow.write_events if e['accepted'] and e['previous_frame'] is not None]
        assert refreshed and all(e['frame']-e['previous_frame']>=shadow.max_frames_lost for e in refreshed)
        assert {e['frame'] for e in refreshed}>={30,60}
        assert any(not e['accepted'] for e in shadow.write_events)
        assert all(e['previous_frame'] is None or e['previous_frame']<e['frame'] for e in shadow.write_events)
