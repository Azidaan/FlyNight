import numpy as np
from tracking.z13_spatial_derivatives import derivative_projector,spatial_descriptor,zero_extension

def geometry():
    from tracking.z13_readout import HexGeometry
    import json
    from pathlib import Path
    g=json.loads((Path(__file__).resolve().parents[1]/'runs/z13_local_peak_20261021/geometry.json').read_text(encoding='utf-8'))
    return HexGeometry(g['points'],g['eye_size'])

def raw_features(values,nb):return np.concatenate((values,values[:,nb[:,1:]].mean(-1)),axis=0)

def test_affine_fields_have_correct_derivatives_even_at_boundary():
    g=geometry();p=derivative_projector(g.points,g.neighbors,g.spacing);xy=g.points.astype(np.float64)/g.spacing
    base=np.stack((2*xy[:,0]-3*xy[:,1]+4,-xy[:,0]+.5*xy[:,1]))
    raw=raw_features(base,g.neighbors);out=spatial_descriptor(raw,np.zeros(4),np.ones(4),g.neighbors,p)
    np.testing.assert_allclose(out[:,4:6],np.tile([2,-1],(721,1)),atol=1e-12)
    np.testing.assert_allclose(out[:,6:],np.tile([-3,.5],(721,1)),atol=1e-12)
    np.testing.assert_array_equal(out[:,:4],raw.T)

def test_reversing_spatial_pattern_aliases_old_center_but_not_derivative():
    g=geometry();p=derivative_projector(g.points,g.neighbors,g.spacing);xy=g.points.astype(np.float64)/g.spacing
    center=g.nearest([g.eye_size/2,g.eye_size/2]);x=xy[:,0]-xy[center,0]
    a=raw_features(x[None],g.neighbors);b=raw_features(-x[None],g.neighbors)
    np.testing.assert_allclose(a[:,center],b[:,center],atol=1e-12)
    aa=spatial_descriptor(a,np.zeros(2),np.ones(2),g.neighbors,p)
    bb=spatial_descriptor(b,np.zeros(2),np.ones(2),g.neighbors,p)
    np.testing.assert_allclose(aa[center,2:],-bb[center,2:],atol=1e-12)
    assert abs(aa[center,2]-bb[center,2])>1

def test_neighbor_order_and_zero_extension_controls():
    g=geometry();rng=np.random.default_rng(16);values=rng.normal(size=(3,721));raw=raw_features(values,g.neighbors)
    nb=g.neighbors.copy();nb[:,1:]=nb[:,[5,2,6,1,4,3]]
    a=spatial_descriptor(raw,np.zeros(6),np.ones(6),g.neighbors,derivative_projector(g.points,g.neighbors,g.spacing))
    b=spatial_descriptor(raw,np.zeros(6),np.ones(6),nb,derivative_projector(g.points,nb,g.spacing))
    np.testing.assert_allclose(a,b,atol=1e-12)
    from tracking.z13_target_center import ridge_scores
    y=rng.normal(size=721);xs=rng.normal(size=(2,721,6))
    old=ridge_scores(raw.T,y,xs,[.01])[0]
    padded=ridge_scores(zero_extension(raw.T),y,zero_extension(xs),[.01])[0]
    np.testing.assert_allclose(old,padded,atol=1e-10,rtol=1e-10)
