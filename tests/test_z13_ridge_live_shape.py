from types import SimpleNamespace
import numpy as np
import torch
import pytest
from tracking.z13_ridge_localizer_v2 import RecoveredCenterRidgeLocalizer


def localizer():
    loc=object.__new__(RecoveredCenterRidgeLocalizer);loc.arm='R1';loc.torch=torch;loc.lam=.1;loc.mean=np.zeros(68);loc.sd=np.ones(68)
    loc.bank=SimpleNamespace(eye_size=391);points=np.column_stack([np.arange(721)%31,np.arange(721)//31]).astype(np.float32)
    loc.geom=SimpleNamespace(points=points,spacing=1.,neighbors=np.repeat(np.arange(721)[:,None],7,axis=1))
    return loc


def test_live_singleton_channel_is_exactly_the_flat_episode_calculation():
    torch.manual_seed(7);loc=localizer();hx=torch.rand(2,4,721);mask=np.zeros((2,721),bool);mask[:,100:120]=True
    flat=loc.start_sequence(hx,mask);live=loc.start_sequence(hx[:,:,None,:],mask)
    for _ in range(6):
        cur=torch.rand(2,721);a=loc.advance(flat,cur);b=loc.advance(live,cur[:,None,:])
        for i in range(2):
            assert np.array_equal(flat[i]['state'],live[i]['state'])
            assert np.array_equal(a[i][0],b[i][0]) and a[i][1]==b[i][1]


def test_multichannel_retina_is_not_silently_flattened():
    loc=localizer()
    with pytest.raises(ValueError):loc.start_sequence(torch.zeros(2,4,2,721),np.ones((2,721),bool))
    with pytest.raises(ValueError):loc.advance([],torch.zeros(2,2,721))
