"""Причинный ввод кадров и геометрия отказа, без запуска GPU."""
import numpy as np
import pytest
from tracking.sam2_mask_recovery import CausalFrames,MaskOutputRecovery

def test_frame_store_cannot_read_future_or_forgotten_images():
    frames=CausalFrames(100,image_size=16,keep=2)
    image=np.zeros((12,20,3),np.uint8)
    frames.append(0,image);assert tuple(frames[0].shape)==(3,16,16)
    with pytest.raises(AssertionError,match='Future'):frames[1]
    frames.append(1,image);frames.append(2,image)
    with pytest.raises(AssertionError,match='outside'):frames[0]
    assert all(request<=seen for seen,request in frames.requests)

def test_duplicate_refusal_uses_high_overlap_not_a_nearby_object():
    base=np.array([[0,0,10,10,1,.9,0,0]],np.float32)
    assert MaskOutputRecovery.duplicate(np.array([0,0,10,10]),base)
    assert not MaskOutputRecovery.duplicate(np.array([5,0,15,10]),base)
    assert not MaskOutputRecovery.duplicate(np.array([20,0,30,10]),base)
