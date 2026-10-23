import numpy as np
from scripts.z13_effect_path_audit_20261023 import compare


def test_global_id_renaming_is_distinct_from_fragmentation():
    a=np.array([[0,1,2,3,4,8,.9,2],[1,2,2,4,4,8,.9,2]],float)
    b=a.copy();b[:,5]=99
    r=compare(a,b);assert r['different_frames']==[0,1] and not r['geometry_different_frames'] and r['equal_up_to_global_id_renaming']
    b[1,5]=100
    r=compare(a,b);assert not r['geometry_different_frames'] and not r['equal_up_to_global_id_renaming']


def test_missing_detection_changes_geometry_even_without_changed_ids():
    a=np.array([[0,1,2,3,4,8,.9,2],[1,2,2,4,4,8,.9,2]],float)
    r=compare(a,a[:1]);assert r['different_frames']==[1] and r['geometry_different_frames']==[1] and not r['equal_up_to_global_id_renaming']
