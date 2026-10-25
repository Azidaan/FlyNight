import numpy as np
from tracking.z13_selected_box_value import fit_value,predict_value


def test_fitted_training_statistics_are_fixed_at_inference():
    x=np.arange(60,dtype=float).reshape(20,3);y=np.tile([-1.,1.],10)
    m=fit_value(x,y);snapshot={k:v.copy() if isinstance(v,list) else v for k,v in m.items()}
    a=predict_value(m,x[:2]);b=predict_value(m,np.r_[x[:2],np.full((1,3),1e9)])
    np.testing.assert_array_equal(a,b[:2]);assert m==snapshot


def test_zero_feature_control_predicts_only_training_mean():
    m=fit_value(np.zeros((10,4)),np.array([1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.]))
    np.testing.assert_array_equal(predict_value(m,np.zeros((3,4))),np.zeros(3))
