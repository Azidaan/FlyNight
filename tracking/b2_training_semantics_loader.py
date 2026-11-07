"""Preserve FlyVis D constraint projection during no-grad evaluation.

FlyVis Network.clamp gates projection and symmetry on requires_grad. The
training dev pass retains D free-parameter flags, although autograd is off.
This loader preserves that function; no optimizer or backward is performed.
"""
def load_branches_training_semantics(checkpoint, device):
    from tracking.b2_pilot_runtime import load_branches
    from tracking.z13_readout import FREE
    models=load_branches(checkpoint,device)
    _,backbone=models['D']
    for name,param in backbone.net.named_parameters():
        param.requires_grad_(name in FREE)
    return models
