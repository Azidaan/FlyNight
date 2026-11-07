"""Не запрашивать нейронную выдачу между запросами, но сохранять прошлое."""
from types import SimpleNamespace
import numpy as np
from tracking.sam2_query_recovery import QueryMaskOutputRecovery

def test_unrequested_frames_keep_real_history_without_neural_propagation():
    class Predictor:
        image_size=16
        def add_new_points_or_box(self,*a,**k):raise AssertionError('Not requested')
        def propagate_in_video(self,*a,**k):raise AssertionError('Not requested')
    class Branch(QueryMaskOutputRecovery):
        def _init(self):self.state={'obj_ids':[]}
    branch=Branch(Predictor(),10);image=np.zeros((20,20,3),np.uint8)
    base=np.array([[1,2,10,12,3,.9,0,0]],np.float32)
    assert not len(branch.step(0,image,base,[],request_output=False))
    lost=SimpleNamespace(track_id=3)
    assert not len(branch.step(1,image,np.zeros((0,8),np.float32),[lost],request_output=False))
    assert branch.observed[3][0]==0 and branch.frames.latest==1
    assert branch.initialized==0 and branch.propagated==0
