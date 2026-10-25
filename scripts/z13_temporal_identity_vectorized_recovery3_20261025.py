"""Structurally stack factory-created equivalent module classes; preserve frozen recovery2."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_temporal_identity_vectorized_recovery2_20261025 as V


def structural_stack(models,torch):
    assert models and all(m.training==models[0].training for m in models)
    parameters=[dict(m.named_parameters()) for m in models]
    buffers=[dict(m.named_buffers()) for m in models]
    assert all(list(x)==list(parameters[0]) for x in parameters)
    assert all(list(x)==list(buffers[0]) for x in buffers)
    for key,value in parameters[0].items():
        assert all(x[key].shape==value.shape and x[key].requires_grad==value.requires_grad for x in parameters)
    params={key:torch.stack([x[key] for x in parameters]).detach().requires_grad_(value.requires_grad) for key,value in parameters[0].items()}
    state={key:torch.stack([x[key] for x in buffers]) for key in buffers[0]}
    return params,state


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    torch.func.stack_module_state=lambda models:structural_stack(models,torch)
    V.retry_write(V.OUT/'structural_stack_recovery3.json',dict(at=V.S.now().isoformat(),wrapper_sha256=V.S.sha(Path(__file__)),
        frozen_recovery2_sha256=V.S.sha(Path(V.__file__)),
        reason='CPU test caught exact-class-identity restriction:make_model defines a fresh nested class per call. Stack identical named parameter/buffer tensors after verifying shapes and requires_grad;no scientific settings or numerical tolerances change.',
        original_internal_deadline_retained=V.S.read(V.OUT/'protocol.json')['save_at']))
    try:V.worker()
    except Exception as e:V.retry_write(V.OUT/'failure.json',dict(at=V.S.now().isoformat(),error=repr(e)));raise
