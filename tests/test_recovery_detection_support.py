from copy import deepcopy
from tracking.recovery_detection_support import supported_candidates


def label(ident,x=0,category='car'):
    return dict(id=str(ident),category=category,score=.2,box2d=dict(x1=x,y1=0,x2=x+10,y2=10))


def test_support_is_one_to_one_and_preserves_candidate_boxes():
    candidates=[label(1),label(2,1)];before=deepcopy(candidates)
    accepted,counts=supported_candidates([],candidates,[label('det')])
    assert len(accepted)==1 and accepted[0]==candidates[0] and candidates==before
    assert counts['accepted']==1


def test_occupied_or_wrong_class_detection_cannot_support_lost_candidate():
    assert supported_candidates([label(5)],[label(1)],[label('det')])[0]==[]
    assert supported_candidates([],[label(1)],[label('det',category='person')])[0]==[]


def test_empty_candidates_or_detections_produce_no_additions():
    assert supported_candidates([],[],[label('det')])[0]==[]
    assert supported_candidates([],[label(1)],[])[0]==[]
