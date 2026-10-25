import numpy as np
import pytest
from tracking.z13_region_product import product_scale,product_features


def test_interaction_preserves_original_filter_response_and_target_channel_alignment():
    rng=np.random.default_rng(7);x=rng.normal(size=(4,3,68)).astype(np.float32)
    w=rng.normal(size=(4,68)).astype(np.float32);mask=np.ones((4,3),bool)
    scale=product_scale(x,w,mask);features=product_features(x,w,scale)
    np.testing.assert_allclose((features*scale).sum(-1),(x*w[:,None,:]).sum(-1),rtol=2e-6,atol=3e-6)
    assert not np.allclose(features,product_features(x,np.roll(w,1,axis=-1),scale))


def test_padding_does_not_change_fit_normalization_and_zero_control_is_finite():
    x=np.ones((2,2,68),np.float32);w=np.ones((2,68),np.float32)
    mask=np.array([[True,False],[True,False]]);a=product_scale(x,w,mask)
    x[:,1]=1e20
    np.testing.assert_array_equal(a,product_scale(x,w,mask))
    np.testing.assert_array_equal(product_features(np.zeros_like(x),w,a),np.zeros_like(x))


def test_invalid_interaction_input_rejected():
    with pytest.raises(ValueError):product_scale(np.ones((2,2,67)),np.ones((2,68)),np.ones((2,2),bool))
    with pytest.raises(ValueError):product_features(np.ones((2,2,68)),np.ones((2,68)),np.zeros(68))
