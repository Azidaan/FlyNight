import pytest
from scripts.z13_readout_overlap import audit


def row(i, useful, kind=1):
    return dict(label=i, video='v', kind=kind, scale=10, err_pred=5,
                err=1 if useful else 8, center=[i,0], useful=useful, pos=0)


def test_partition_and_unknown_exclusion():
    a=[row(0,True),row(1,True),row(2,False),row(3,False),row(4,False,-1)]
    b=[row(0,True),row(1,False),row(2,True),row(3,False),row(4,True,-1)]
    rep,_=audit(a,b[::-1])
    assert rep['positives']==4
    assert all(c['n']==1 for c in rep['cells'].values())
    assert rep['oracle_gain_over_old_mean_video']==.25


def test_reject_mismatched_reference_and_duplicates():
    with pytest.raises(ValueError,match='Duplicate'):
        audit([row(0,True)]*2,[row(0,True)])
    changed=row(0,False); changed['err_pred']=6
    with pytest.raises(ValueError,match='reference'):
        audit([row(0,True)],[changed])
