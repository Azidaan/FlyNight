import numpy as np
from types import SimpleNamespace
from tracking.query_memory_write import MemoryWriteTracker,combine
from tracking.query_appearance_comparison import QueryAppearanceTracker,assert_states_equal
from tracking.offline_association import _DetSlice
from tests.test_query_appearance_comparison import Server

def test_memory_write_limits_and_confidence():
    a=np.array([1,0,0],np.float32);b=np.array([0,1,0],np.float32)
    assert np.array_equal(combine(a,b,'last',.5,.25)[0],b)
    assert np.array_equal(combine(a,b,'cautious',.25,.25)[0],a)
    assert np.array_equal(combine(a,b,'mean',1,.25)[0],combine(a,b,'cautious',1,.25)[0])
    low=combine(a,b,'cautious',.3,.25)[0];mean=combine(a,b,'mean',.3,.25)[0]
    assert low[1]<mean[1] and np.isclose(np.linalg.norm(low),1)
    assert np.array_equal(a,[1,0,0]) and np.array_equal(b,[0,1,0])

def test_last_exact_and_cautious_shadow_state_and_no_shared_vector_mutation():
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    servers=[Server(None) for _ in range(3)]
    old=QueryAppearanceTracker(args,servers[0],.1)
    last=MemoryWriteTracker(args,servers[1],.1)
    native=QueryAppearanceTracker(args)
    zero=MemoryWriteTracker(args,servers[2],.1,True,'cautious')
    for frame in range(60):
        boxes=np.array([[10+frame*.1,10,40+frame*.1,40],[15+frame*.1,10,45+frame*.1,40]],np.float32)
        d=_DetSlice(boxes,np.array([.9,.3],np.float32),np.zeros(2,np.float32));active=frame%5==0
        if active:
            for s in servers:s.start_frame(frame,None)
        assert np.array_equal(old.update_query_frame(d,active),last.update_query_frame(d,active))
        assert_states_equal(old,last)
        assert np.array_equal(native.update_query_frame(d,active),zero.update_query_frame(d,active))
        assert_states_equal(native,zero)
        if active:
            snapshots={id(t):t.vector.copy() for t in servers[2].tokens.values() if t.vector is not None}
            for s in servers:s.finish_frame()
            for t in servers[2].tokens.values():
                if id(t) in snapshots:assert np.array_equal(t.vector,snapshots[id(t)])
    assert old.events==last.events and zero.write_events
    assert all(e['previous_frame'] is None or e['previous_frame']<e['frame'] for e in zero.write_events)
