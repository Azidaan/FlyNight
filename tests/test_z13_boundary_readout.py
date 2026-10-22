import json
from pathlib import Path
import numpy as np
import pytest
from tracking.z13_readout import HexGeometry
from tracking.z13_boundary_readout import outer_ring,read_response
from tracking.z13_ridge_localizer import map_center


def geometry():
    g=json.loads((Path(__file__).resolve().parents[1]/'runs/z13_local_peak_20261021/geometry.json').read_text(encoding='utf-8'))
    return HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])


def test_geometry_and_refusal_do_not_move_the_answer():
    g=geometry();edge=outer_ring(g)
    assert len(edge)==721 and edge.sum()==90
    for index,accepted in [(int(np.flatnonzero(edge)[0]),False),(int(np.flatnonzero(~edge)[0]),True)]:
        score=np.zeros(721);score[index]=10
        center,confidence,actual=read_response(score,g)
        original,oldconf=map_center(score,g)
        assert np.array_equal(center,original) and confidence==oldconf and actual is accepted
        assert read_response(score,g,False)[2] is True


def test_affine_map_changes_do_not_change_boundary_refusal():
    g=geometry();x=np.sin(np.arange(721));x[0]=8
    a=read_response(x,g);b=read_response(3*x+7,g)
    assert a[2]==b[2]
    assert np.allclose(a[0],b[0],atol=1e-8)


def test_rejects_nonfinite_map():
    g=geometry();x=np.zeros(721);x[0]=np.nan
    with pytest.raises(ValueError):read_response(x,g)
