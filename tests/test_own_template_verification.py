import numpy as np
from tracking.own_template_verification import correspondence,gather,absolute_score,passes


def fixture():
    points=np.array([(x,y) for y in range(3) for x in range(3)],float)
    g=dict(points=points,eye_size=3,spacing=1);history=[dict(box=[0,0,3,3],crop=[0,0,3,3])]
    transform=dict(valid=True,scale=[1,1],offset=[0,0]);return points,g,history,transform


def test_identical_spatial_patterns_pass_and_opposite_patterns_fail():
    points,g,h,t=fixture();pairs=correspondence(t,h,[0,0,3,3],[0,0,3,3],g,np.ones(9,bool),np.ones(9,bool))
    args=dict(video='v',tid=1,last_frame=0,frame=2,column=0)
    a=absolute_score('FLYVIS',points,points,pairs,[1,1],**args)
    b=absolute_score('FLYVIS',points,-points,pairs,[1,1],**args)
    assert a==1 and b==-1 and passes(a,.5) and not passes(b,.5)


def test_small_or_constant_support_abstains_instead_of_activating_geometry():
    points,g,h,t=fixture();valid=np.arange(9)<2
    assert correspondence(t,h,[0,0,3,3],[0,0,3,3],g,valid,np.ones(9,bool)) is None
    pairs=correspondence(t,h,[0,0,3,3],[0,0,3,3],g,np.ones(9,bool),np.ones(9,bool))
    value=absolute_score('IMAGE',np.ones((9,3)),np.ones((9,3)),pairs,[1,1],video='v',tid=1,last_frame=0,frame=2,column=0)
    assert value is None and not passes(value,-1)


def test_duplicate_current_samples_get_one_reference_cell_weight():
    pairs=dict(p=np.array([0,1,2,3]),unique=np.array([0,1,2]),inverse=np.array([0,0,1,2]))
    assert np.array_equal(gather(np.array([[0.],[2.],[3.],[4.]]),pairs),np.array([[1.],[3.],[4.]]))


def test_disabled_and_invalid_values_never_pass():
    assert not passes(1.,'off') and not passes(None,-1) and not passes(float('nan'),-1) and not passes(float('inf'),-1)
    assert passes(.5,.5) and not passes(.49,.5)
