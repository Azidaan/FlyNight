import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader


def reader():
    x,y=np.meshgrid(np.arange(10.,100,20),np.arange(10.,100,20))
    return ObservedSpatialReader(np.column_stack([x.ravel(),y.ravel()]),100,[100,100])


def test_readout_maps_the_visible_edge_and_matches_a_known_translation():
    r=reader();p=r.plan([50,20,90,80],[0,0,100,100],[70,20,100,80],[0,0,100,100],[15,0],[[0,0]],reference_observation=[50,20,90,80])
    assert p['anchor']['changed'] and np.array_equal(p['prediction'],[20,0])
    x,y=r.points.T
    a=np.column_stack([np.sin(x/13)+np.cos(y/17),np.sin(x*y/400)])
    b=np.column_stack([np.sin((x-20)/13)+np.cos(y/17),np.sin((x-20)*y/400)])
    result=r.compare(a,b,p)
    assert len(p['cells'])==6 and abs(result['score']-1)<1e-12 and result['winner']==0


def test_missing_raw_observation_preserves_prediction_and_flat_maps_abstain():
    r=reader();p=r.plan([50,20,90,80],[0,0,100,100],[70,20,100,80],[0,0,100,100],[15,0],[[0,0]])
    assert not p['anchor']['changed'] and np.array_equal(p['prediction'],[15,0])
    assert r.compare(np.ones((25,4)),np.ones((25,4)),p)['score'] is None
    p['cells']=p['cells'][:2];p['query']=p['query'][:2]
    assert r.compare(np.ones((25,4)),np.ones((25,4)),p)['reason']=='fewer_than_three_cells'
