import numpy as np
from scripts.hota_error_budget_20261027 import interventions
from tracking.recovery_admission_features import CausalAdmissionMemory

def label(ident,x=0):return dict(id=str(ident),category='car',score=.8,box2d=dict(x1=x,y1=0,x2=x+10,y2=10))

def test_identity_keeps_detection_geometry_and_false_detections():
    base=[label(10,1),label(20,30)];gt=[label('true')]
    changed=interventions(base,gt)
    assert len(changed['identity'])==2
    assert [x['box2d'] for x in changed['identity']]==[x['box2d'] for x in base]
    assert changed['identity'][1]['id']=='P_20'
    assert changed['geometry'][0]['id']=='10' and changed['geometry'][1]==base[1]
    assert changed['geometry'][0]['box2d']==gt[0]['box2d']
    assert base[0]['box2d']['x1']==1

def test_unmatched_truth_addition_is_conditional_on_same_identity_assignment():
    changed=interventions([label(10)],[label('first'),label('missing',30)])
    assert changed['identity_missing'][:1]==changed['identity']
    assert len(changed['identity_missing'])==2
    assert len({x['id'] for x in changed['identity_missing']})==2

def test_current_observation_cannot_be_used_as_past_crop():
    from types import SimpleNamespace
    m=CausalAdmissionMemory();image=np.zeros((20,40,3),np.uint8)
    row=np.array([[0,0,10,10,1,.8,2,0]],np.float32)
    m.remember(3,row,image,{1})
    track=SimpleNamespace(cls=2,score=.8,track_id=1,end_frame=3)
    import pytest
    with pytest.raises(AssertionError):m.describe(3,label(1),track,[],[],[],[],image)
    x,current,past,previous=m.describe(4,label(1),track,[],[],[],[],image)
    assert previous==3 and np.isfinite(x).all()
