import numpy as np
import pytest
from tracking.native_confirm_identity import NativeConfirmedIdentity


def output(tid):return np.array([[0,0,10,10,tid,.9,2,0]],np.float32)


@pytest.mark.parametrize('alpha,cost,expected',[(.01,.82,2),(.025,.82,1),(.025,.83,2),(.2,.83,1)])
def test_cost_allowance_is_applied_by_the_optional_solver(alpha,cost,expected):
    p=NativeConfirmedIdentity(alpha=alpha);p.update(0,output(1),[],[]);p.update(1,np.empty((0,8),np.float32),[],[])
    current=output(2);changed,d=p.update(2,current,[2],[dict(old_tid=1,new_tid=2,cost=cost,accepted=True)])
    assert int(changed[0,4])==expected
    assert len(d['allowed'])==1
    assert len(d['links'])==(expected==1)
    assert np.array_equal(current[:,[0,1,2,3,5,6,7]],changed[:,[0,1,2,3,5,6,7]])
