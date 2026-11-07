import numpy as np
from types import SimpleNamespace
from ultralytics.trackers.utils import matching
from tracking.query_cost_use import patch_cost,CostUseTracker
from tracking.query_appearance_comparison import QueryAppearanceTracker,assert_states_equal
from tracking.offline_association import _DetSlice
from tests.test_query_appearance_comparison import Server

def test_negative_cost_solver_and_original_geometry_mask():
    c=np.array([[.01,.81],[.9,.78]],np.float32);d=np.array([[0,0],[0,.4]],np.float32);mask=c<=.8
    centered=patch_cost(c,d,mask,'centered');shift=patch_cost(c,d,mask,'shift')
    assert centered[0,0]<0 and centered[1,1]<.8
    assert np.array_equal(centered[~mask],c[~mask]) and np.array_equal(shift[~mask],c[~mask])
    pairs,_,_=matching.linear_assignment(centered,thresh=.8)
    assert {tuple(x) for x in pairs}=={(0,0),(1,1)}
    assert patch_cost(c,d,mask,'old')[1,1]>.8
    assert np.array_equal(patch_cost(c,np.ones_like(d),mask,'shift'),shift)

def test_old_mode_and_centered_shadow_reproduce_exact_tracks_and_states():
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    servers=[Server(None) for _ in range(3)]
    old=QueryAppearanceTracker(args,servers[0],.1);same=CostUseTracker(args,servers[1],.1,mode='old')
    zero=CostUseTracker(args,servers[2],.1,True,'centered');native=QueryAppearanceTracker(args)
    for frame in range(60):
        active=frame%5==0
        if active:
            for s in servers:s.start_frame(frame,None)
        d=_DetSlice(np.array([[10+frame*.1,10,40+frame*.1,40],[15+frame*.1,10,45+frame*.1,40]],np.float32),np.array([.9,.8],np.float32),np.zeros(2,np.float32))
        assert np.array_equal(old.update_query_frame(d,active),same.update_query_frame(d,active));assert_states_equal(old,same)
        assert np.array_equal(zero.update_query_frame(d,active),native.update_query_frame(d,active));assert_states_equal(zero,native)
        if active:
            for s in servers:s.finish_frame()
    assert old.events==same.events and zero.shadow_pairs>0
    assert all(e['previous_frame'] is None or e['previous_frame']<e['frame'] for e in zero.write_events)

def test_shift_control_does_not_use_appearance_content():
    class Constant(Server):
        def features(self,tokens,null_seed=None):
            for t in tokens:
                if t.vector is None:t.vector=np.array([1,0,0],np.float32)
            return np.stack([t.vector for t in tokens])
    args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
    servers=[Server(None),Constant(None)];trackers=[CostUseTracker(args,s,.1,mode='shift') for s in servers]
    for frame in range(40):
        active=frame%5==0
        if active:
            for s in servers:s.start_frame(frame,None)
        d=_DetSlice(np.array([[10+frame*.1,10,40+frame*.1,40],[15+frame*.1,10,45+frame*.1,40]],np.float32),np.array([.9,.8],np.float32),np.zeros(2,np.float32))
        assert np.array_equal(trackers[0].update_query_frame(d,active),trackers[1].update_query_frame(d,active))
        assert_states_equal(*trackers)
        if active:
            for s in servers:s.finish_frame()
    assert trackers[0].events and any(a['appearance']!=b['appearance'] for a,b in zip(trackers[0].events,trackers[1].events))
    assert all(a['updated']==b['updated'] for a,b in zip(trackers[0].events,trackers[1].events))
