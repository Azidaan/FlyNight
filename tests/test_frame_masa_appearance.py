import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import pytest

def test_masa_server_requires_completed_past_and_rejects_future():
    from tracking.frame_masa_appearance import FrameMasaServer
    s=FrameMasaServer(None);s.start_frame(0,np.zeros((20,20,3),np.uint8));t=s.observation([0,0,10,10])
    with pytest.raises(AssertionError,match='Сначала'):s.start_frame(1,np.zeros((20,20,3),np.uint8))
    t.vector=np.eye(1,256,dtype=np.float32)[0];s.finish_frame();s.start_frame(1,np.ones((20,20,3),np.uint8))
    assert np.array_equal(s.features([t])[0],t.vector)
    t.vector=None
    with pytest.raises(AssertionError,match='Прошлые'):s.features([t])

def test_masa_null_vectors_are_reproducible_without_identity_signal():
    from tracking.frame_masa_appearance import FrameMasaServer
    a=FrameMasaServer(None);a.start_frame(0,np.zeros((20,20,3),np.uint8));x=a.observation([0,0,10,10]);y=a.observation([1,0,11,10])
    z=a.features([x,y],42);assert z.shape==(2,256) and np.allclose(np.linalg.norm(z,axis=1),1)
    assert not np.array_equal(z[0],z[1]) and not np.array_equal(z,a.features([x,y],43))
    b=FrameMasaServer(None);b.start_frame(0,np.ones((20,20,3),np.uint8));t=b.observation([0,0,10,10])
    assert np.array_equal(z[0],b.features([t],42)[0])
