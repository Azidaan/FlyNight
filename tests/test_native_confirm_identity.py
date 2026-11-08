import numpy as np
from tracking.native_confirm_identity import NativeConfirmedIdentity


def boxes(*ids):return np.array([[1,2,30,40,t,.8,2,0] for t in ids],np.float32).reshape(-1,8)
def edge(old,new,cost=.95,accepted=True):return dict(old_tid=old,new_tid=new,cost=cost,accepted=accepted)


def test_confirmed_first_public_link_preserves_every_physical_value():
    p=NativeConfirmedIdentity();p.update(0,boxes(1),set(),[]);source=boxes(2)
    out,event=p.update(1,source,{2},[edge(1,2)])
    assert out[0,4]==1 and len(event['links'])==1
    assert np.array_equal(out[:,[0,1,2,3,5,6,7]],source[:,[0,1,2,3,5,6,7]]) and source[0,4]==2
    later,_=p.update(2,boxes(2),set(),[]);assert later[0,4]==1


def test_never_steals_a_working_public_identity_or_links_unconfirmed():
    p=NativeConfirmedIdentity();p.update(0,boxes(1,3),set(),[])
    out,e=p.update(1,boxes(1,2,3),{2},[edge(1,2),edge(3,2)])
    assert np.array_equal(out,boxes(1,2,3)) and not e['links']
    out,e=p.update(2,boxes(4),set(),[edge(1,4)])
    assert out[0,4]==4 and not e['links']
    out,e=p.update(3,boxes(4),{4},[edge(1,4)])
    assert out[0,4]==4 and not e['links']


def test_only_current_native_owner_may_offer_an_old_canonical_id():
    p=NativeConfirmedIdentity();p.update(0,boxes(1),set(),[]);p.update(1,boxes(2),{2},[edge(1,2)])
    out,e=p.update(2,boxes(3),{3},[edge(1,3)])
    assert out[0,4]==3 and not e['links']
    out,e=p.update(3,boxes(4),{4},[edge(2,4)])
    assert out[0,4]==1 and len(e['links'])==1


def test_native_return_revokes_alias_without_dropping_or_retroactively_editing():
    p=NativeConfirmedIdentity();old,_=p.update(0,boxes(1),set(),[]);linked,_=p.update(1,boxes(2),{2},[edge(1,2)])
    out,e=p.update(2,boxes(1,2),set(),[])
    assert np.array_equal(out,boxes(1,2)) and len(e['revoked'])==1
    assert old[0,4]==1 and linked[0,4]==1


def test_shared_solver_assigns_one_old_identity_only_once():
    p=NativeConfirmedIdentity();p.update(0,boxes(1,2),set(),[])
    out,e=p.update(1,boxes(3,4),{3,4},[edge(1,3,.9),edge(1,4,.94),edge(2,3,.91)])
    assert len(e['links'])==2 and set(out[:,4])=={1,2}


def test_gate_and_zero_mode():
    for kwargs,accepted in [({},False),({'enabled':False},True),({'alpha':0},True)]:
        p=NativeConfirmedIdentity(**kwargs);p.update(0,boxes(1),set(),[])
        out,e=p.update(1,boxes(2),{2},[edge(1,2,accepted=accepted)])
        assert np.array_equal(out,boxes(2)) and not e['links']
