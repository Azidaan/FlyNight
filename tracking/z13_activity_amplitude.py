"""Retain query activity mean/scale with the same trainable affine parameters."""
def make_amplitude_ranker(torch,activity=True,preserve=False):
    from tracking.z13_region_assignment_context import make_context_ranker
    model=make_context_ranker(torch,activity=activity)
    if preserve:
        class Affine(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.ones(68))
                self.bias=torch.nn.Parameter(torch.zeros(68))
            def forward(self,x):return x*self.weight+self.bias
        model.embed[0]=Affine()
    return model
