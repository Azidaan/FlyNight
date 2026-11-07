import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from tracking.learned_recovery_admission import raw_input

def test_feature_of_an_observation_is_independent_of_other_rows_and_future_images():
    rng=np.random.default_rng(17);x=rng.normal(size=(4,24)).astype(np.float32);e=rng.normal(size=(4,2,512)).astype(np.float32)
    projection=rng.normal(size=(24,2048)).astype(np.float32);mean=np.zeros(24,np.float32);std=np.ones(24,np.float32)
    visual=raw_input(x,e,'appearance',mean,std,projection);control=raw_input(x,e,'numeric',mean,std,projection)
    changed=e.copy();changed[1:]*=100
    assert np.array_equal(visual[:1],raw_input(x,changed,'appearance',mean,std,projection)[:1])
    assert np.array_equal(control,raw_input(x,changed,'numeric',mean,std,projection))
    assert visual.shape==control.shape==(4,2072)
    assert np.allclose(visual[:1],raw_input(x[:1],e[:1],'appearance',mean,std,projection))
