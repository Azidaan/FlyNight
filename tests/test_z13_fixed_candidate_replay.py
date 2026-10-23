import numpy as np
from types import SimpleNamespace
from tracking.z13_candidates import candidates
from tracking.z13_fixed_candidate_replay import replay


def test_saved_anchors_remain_verifiable_when_extra_maximum_changes_selection():
    # 418/419 are the observed failure's nearly tied neighbor pair. The
    # historical anchors are450/195/245. The intervention here is a proof only,
    # never a modification of experiment response maps.
    points=np.asarray([[13*v+195,int(13*(u+v/2))+195] for u in range(-15,16) for v in range(max(-15,-15-u),min(15,15-u)+1)],np.float32)
    from tracking.z13_readout import HexGeometry
    g=HexGeometry(points,391)
    path='runs/z13_filter_descriptor_cache_20261023/parity_audit_cpu_20261024/observed_response_maps.npz'
    with np.load(path) as z:score=z['contiguous4_stored_2d']
    x=np.arange(721*2,dtype=float).reshape(721,2)/1500
    shifted=score.copy();shifted[418]-=2e-7
    before=candidates(shifted,x,g);after=candidates(score,x,g)
    assert before['indices'].tolist()==[450,195,245] and after['indices'].tolist()==[450,418,195]
    result=replay(score,x,g,before['centers'],before['scores'],[0,0,391,391],391)
    assert result['indices'].tolist()==before['indices'].tolist()
    assert result['center_max_px']<.5 and result['score_max_abs']<.001
    assert result['features'].shape==(3,2)


def test_edge_duplicate_neighbors_do_not_change_weighted_center():
    g=SimpleNamespace(neighbors=np.asarray([[0,1,0],[1,0,1]]),points=np.asarray([[0.,0.],[1.,0.]]))
    r=replay(np.array([2.,1.]),np.array([[2.],[3.]]),g,np.array([[.26894142137,0]]),np.array([1.]),[0,0,1,1],1)
    # sd=.5, hence exp(-2), not exp(-1). Use the exact expected value below.
    expected=1/(1+np.exp(2))
    assert abs(r['centers'][0,0]-expected)<1e-8
