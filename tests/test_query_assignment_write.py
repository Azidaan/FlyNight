import numpy as np
from types import SimpleNamespace
from tracking.query_assignment_write import AssignmentWriteTracker,safe_write_mask
from tracking.query_appearance_comparison import QueryAppearanceTracker,assert_states_equal
from tracking.offline_association import _DetSlice
from tests.test_query_appearance_comparison import Server

def test_unique_pair_requires_both_row_and_column():
    assert np.array_equal(safe_write_mask([[.1,.9],[.9,.2]],.8),np.eye(2,dtype=bool))
    assert not safe_write_mask([[.1,.2]],.8).any()
    assert not safe_write_mask([[.1],[.2]],.8).any()

def test_ambiguous_write_is_rejected_without_changing_shadow_state():
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    server=Server(None);zero=AssignmentWriteTracker(args,server,.1,True);native=QueryAppearanceTracker(args)
    for frame in range(40):
        active=frame%5==0
        if active:server.start_frame(frame,None)
        b=np.array([[10+frame*.1,10,40+frame*.1,40],[15+frame*.1,10,45+frame*.1,40]],np.float32)
        d=_DetSlice(b,np.array([.9,.8],np.float32),np.zeros(2,np.float32))
        before=zero.query_history.copy()
        assert np.array_equal(native.update_query_frame(d,active),zero.update_query_frame(d,active))
        assert_states_equal(native,zero)
        if active:
            server.finish_frame()
            for e in [e for e in zero.write_events if e['frame']==frame and not e['accepted']]:
                assert zero.query_history[e['track_id']] is before[e['track_id']]
    assert any(not e['accepted'] for e in zero.write_events)
