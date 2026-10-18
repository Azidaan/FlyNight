import pytest
from scripts.z13_score_swap import mix,selected_keys


def test_score_swap_preserves_predictions_and_outcomes():
    centers=[dict(video='a',label=1,kind=1,center=[3,4],useful=True,p_present=.1)]
    scores=[dict(video='a',label=1,kind=1,center=[99,99],useful=False,p_present=.9)]
    assert mix(centers,scores)==[dict(centers[0],p_present=.9)]
    assert centers[0]['p_present']==.1


def test_ranking_ignores_target_outcomes():
    rows=[dict(video='a',label=i,kind=1,p_present=i/10,useful=bool(i%2)) for i in range(10)]
    changed=[dict(r,useful=not r['useful'],center=[-999,-999]) for r in rows]
    assert selected_keys(rows,.2)==selected_keys(changed,.2)


def test_swapping_rejects_mismatched_or_duplicated_rows():
    a=dict(video='a',label=1,kind=1,p_present=.1)
    with pytest.raises(ValueError):mix([a],[dict(a,video='b')])
    with pytest.raises(ValueError):mix([a,a],[a,a])
