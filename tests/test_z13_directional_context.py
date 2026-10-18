import numpy as np
from tracking.z13_directional_context import directional_neighbors,expand_features,helmert_contrasts,shuffled_neighbors


def test_angular_assignment_uses_geometry_not_neighbor_storage_order():
    angle=np.arange(6)*np.pi/3
    points=np.r_[[[0,0]],np.c_[np.cos(angle),np.sin(angle)]]
    neighbors=np.stack([np.r_[i,np.repeat(i,6)] for i in range(7)])
    neighbors[0]=[0,4,2,6,1,5,3]
    a=directional_neighbors(points,neighbors)
    neighbors[0]=[0,1,2,3,4,5,6]
    np.testing.assert_array_equal(a,directional_neighbors(points,neighbors))
    assert len(set(a[0]))==6


def test_contrasts_preserve_baseline_and_distinguish_equal_neighbor_means():
    h=helmert_contrasts()
    np.testing.assert_allclose(h.T@h,np.eye(5),atol=1e-15)
    np.testing.assert_allclose(h.sum(0),0,atol=1e-15)
    rng=np.random.default_rng(1);x=rng.normal(size=(7,68));slots=np.tile(np.arange(1,7),(7,1))
    a=expand_features(x,slots)
    np.testing.assert_array_equal(a[:,:68],x)
    rotated=np.roll(slots,1,axis=1);b=expand_features(x,rotated)
    assert not np.allclose(a[:,68:],b[:,68:])
    constant=np.ones((7,68));np.testing.assert_allclose(expand_features(constant,slots)[:,68:],0,atol=1e-15)


def test_shuffle_is_fixed_and_preserves_each_local_set():
    slots=np.tile(np.arange(6),(20,1));a=shuffled_neighbors(slots)
    np.testing.assert_array_equal(a,shuffled_neighbors(slots))
    assert len({tuple(x) for x in a})>1
    np.testing.assert_array_equal(np.sort(a,axis=1),slots)


def test_batched_feature_layout_matches_individual_frames():
    rng=np.random.default_rng(3);x=rng.normal(size=(3,7,68));slots=np.tile(np.arange(1,7),(7,1))
    expanded=expand_features(x,slots)
    assert expanded.shape==(3,7,238)
    np.testing.assert_allclose(expanded,np.stack([expand_features(frame,slots) for frame in x]),atol=1e-15)
