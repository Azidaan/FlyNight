import numpy as np
import pytest
from tracking.history_representations import motion_history,ObservationHistory


def test_future_images_do_not_change_motion_history():
    images=np.zeros((4,3,3),np.float32);images[1:,1,1]=1
    original=motion_history(images,[0,1,2,3],np.ones_like(images,bool),query_time=1,max_age=2,threshold=.2)
    images[2:]=1-images[2:]
    changed=motion_history(images,[0,1,2,3],np.ones_like(images,bool),query_time=1,max_age=2,threshold=.2)
    assert np.array_equal(original['recency'],changed['recency'])
    assert original['recency'][1,1]==1 and original['used_times'].tolist()==[0,1]


def test_invalid_warp_pixels_cannot_create_a_motion_trail():
    images=np.zeros((2,3,3),np.float32);images[1]=1
    valid=np.ones_like(images,bool);valid[1,0]=False
    result=motion_history(images,[0,1],valid,query_time=1,max_age=2,threshold=.2)
    assert not result['recency'][0].any() and not result['comparison_valid'][0].any()
    assert result['recency'][1:].all()


def test_prediction_is_rejected_and_missing_frames_are_not_invented():
    bank=ObservationHistory(4,2)
    with pytest.raises(ValueError):bank.append(time=0,box_xywh=[0,0,2,2],appearance=[1,0],quality=1,observed=False)
    for t in [0,5]:bank.append(time=t,box_xywh=[t,0,2,2],appearance=[1,0],quality=1,observed=True)
    packet=bank.view(query_time=6,max_age=10,reference_xywh=[6,0,2,2])
    assert packet['observed_times'].tolist()==[0,5] and packet['valid'].sum()==2
    assert packet['age_seconds'][:2].tolist()==[6,1]


def test_future_observation_is_rejected_and_inputs_are_copied():
    bank=ObservationHistory(4,2);vector=np.array([1.,0.])
    bank.append(time=0,box_xywh=[0,0,2,2],appearance=vector,quality=1,observed=True);vector[:]=[0,1]
    packet=bank.view(query_time=1,max_age=10,reference_xywh=[0,0,2,2])
    assert packet['valid'].sum()==1 and np.array_equal(packet['appearance'][0],[1,0])
    last,average=bank.simple_controls(query_time=1,max_age=10,ema_alpha=.5)
    assert np.array_equal(last,[1,0]) and np.array_equal(average,[1,0])
    bank.append(time=2,box_xywh=[20,0,2,2],appearance=[0,1],quality=.1,observed=True)
    with pytest.raises(ValueError):bank.view(query_time=1,max_age=10,reference_xywh=[0,0,2,2])
    with pytest.raises(ValueError):bank.simple_controls(query_time=1,max_age=10,ema_alpha=.5)


def test_nonmonotonic_observations_and_empty_history_are_explicit():
    bank=ObservationHistory(4,2)
    assert bank.simple_controls(query_time=1,max_age=2,ema_alpha=.5)==(None,None)
    bank.append(time=2,box_xywh=[0,0,2,2],appearance=[1,0],quality=1,observed=True)
    with pytest.raises(ValueError):bank.append(time=1,box_xywh=[0,0,2,2],appearance=[1,0],quality=1,observed=True)
