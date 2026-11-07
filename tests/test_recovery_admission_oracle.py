from tracking.recovery_admission_oracle import PastIdentityAdmissionOracle


def label(ident,x=0):return dict(id=str(ident),category='car',box2d=dict(x1=x,y1=0,x2=x+10,y2=10))


def test_current_truth_cannot_create_a_past_identity():
    oracle=PastIdentityAdmissionOracle()
    assert oracle.step([],[label(7)],[label('truth')])==[]
    oracle.step([label(7)],[],[label('truth')])
    assert oracle.step([],[label(7)],[label('truth')])==[label(7)]


def test_existing_base_coverage_and_wrong_identity_refuse_addition():
    oracle=PastIdentityAdmissionOracle();oracle.step([label(7)],[],[label('old')])
    assert oracle.step([label(9)],[label(7)],[label('old')])==[]
    assert oracle.step([],[label(7)],[label('other')])==[]
