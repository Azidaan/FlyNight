"""Инварианты адаптации: история только из прошлого, без изменения трекера."""
from types import SimpleNamespace
import numpy as np
import pytest

def test_only_past_complete_memory_can_add_an_existing_lost_id():
    from tracking.busca_runtime import imports
    imports()
    from tracking.busca_output_recovery import BuscaOutputRecovery
    class Decision:
        def associate_embeddings(self,mem,dets,dists,**kw):
            assert len(mem)==1 and len(mem[0].images_mem)==11
            assert np.array_equal(mem[0].tlwh_mem[-1],np.array([2,3,8,9]))
            probs=np.zeros((1,len(dets)+1));probs[0,-1]=1
            return probs,np.array([True])
    branch=BuscaOutputRecovery(Decision())
    image=np.zeros((20,20,3),dtype=np.uint8)
    output=np.array([[2,3,10,12,7,.9,2,0]],dtype=np.float32)
    for frame in range(11):branch.remember(frame,output,image)
    track=SimpleNamespace(track_id=7,xyxy=np.array([3,3,11,12]),state='lost')
    accepted=branch.select(11,[track],image,np.array([[2,3,10,12]]),np.array([.9]))
    assert accepted==[track] and track.state=='lost'
    assert len(branch.history[7])==11
    with pytest.raises(AssertionError,match='leaked'):
        branch.select(10,[track],image,np.empty((0,4)),np.empty(0))

def test_incomplete_history_never_runs_the_model():
    from tracking.busca_output_recovery import BuscaOutputRecovery
    class Never:
        def associate_embeddings(self,*a,**kw):raise AssertionError('Must not run')
    branch=BuscaOutputRecovery(Never());image=np.zeros((20,20,3),dtype=np.uint8)
    track=SimpleNamespace(track_id=3,xyxy=np.array([1,1,10,10]))
    assert branch.select(5,[track],image,np.empty((0,4)),np.empty(0))==[]
