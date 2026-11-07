import numpy as np
from tracking.history_training_data import label_detection_identities,identity_target

def test_unknown_identity_never_becomes_negative():
    assert identity_target(None,'7')==(-1,False)
    assert identity_target('7',None)==(-1,False)
    assert identity_target('7','8')==(0,True)
    assert identity_target('7','7')==(1,True)

def test_gt_matching_is_unique_and_excludes_crowd():
    labels=[dict(id='7',category='car',box2d=dict(x1=0,y1=0,x2=10,y2=10),attributes={}),
            dict(id='8',category='car',box2d=dict(x1=20,y1=0,x2=30,y2=10),attributes=dict(crowd=True))]
    boxes=np.array([[0,0,10,10],[0,0,10,10],[20,0,30,10]],np.float32)
    found=label_detection_identities(boxes,[0,0,0],labels,['car'])
    assert found.count('7')==1 and found.count(None)==2
    assert label_detection_identities(boxes[:1],[2],labels,{0:'pedestrian',2:'car'})==['7']
