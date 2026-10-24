import numpy as np
from types import SimpleNamespace
from scripts.z13_refusal_trace_20261025 import instrumented_source, mechanism
from tracking.z13_ridge_localizer import map_center


def test_trace_instrumentation_compiles_without_replacing_the_solver():
    source=instrumented_source()
    compile(source,'instrumented_replay','exec')
    assert source.count('assignment_vector(altered,self.args.match_thresh)')==1
    assert 'assert np.array_equal(actual,expected)' in source
    assert 'finally:t.mean[:]=original' in source


def test_relative_peak_confidence_cannot_see_positive_affine_score_changes():
    geom=SimpleNamespace(points=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]]),
        neighbors=np.array([[0,1],[1,2],[2,3],[3,2]]))
    scores=np.array([1.,2.,4.,3.])
    center,confidence=map_center(scores,geom)
    center2,confidence2=map_center(7*scores-200,geom)
    np.testing.assert_allclose(center,center2,atol=1e-8)
    np.testing.assert_allclose(confidence,confidence2,atol=1e-8)


def test_competing_track_and_missing_high_detection_are_separate():
    trace=dict(query_row=0,causal_identities=['A','B'],baseline_assignment=[0,1],
        high_identities=['A','B'],raw_detections=[],track_ids=[11,12],
        candidate_assignments={'D':[[1,0]]},candidate_cost_rows={'D':[[.5,.1]]},match_threshold=.8)
    row=dict(trace=trace,arms={'D':[dict(beneficial=False,harmful=True)]})
    assert mechanism(row,'D',0)['reason']=='correct_detection_assigned_to_competing_track'
    trace['high_identities']=['C','B'];trace['raw_detections']=[dict(identity='A')]
    assert mechanism(row,'D',0)['reason']=='correct_detection_only_below_high_threshold'
