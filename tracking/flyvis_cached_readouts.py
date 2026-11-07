"""Small residual readouts of eight observable T4/T5 score contributions.

No FlyVis import, state, detector, identity annotation or geometry API belongs here.
The frozen existing score is sum(g); every new model starts with zero residual.
"""
import numpy as np

KINDS = ('linear', 'mlp', 'products', 'sigma_pi')
PAIRS = tuple((i,j) for i in range(8) for j in range(i+1,8))

def make_model(torch, kind, seed):
    if kind not in KINDS:
        raise ValueError(kind)
    torch.manual_seed(seed)
    class Readout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if kind=='linear':
                self.w=torch.nn.Parameter(torch.zeros(8,dtype=torch.float64))
                self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
            elif kind=='products':
                self.w=torch.nn.Parameter(torch.zeros(36,dtype=torch.float64))
                self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
            elif kind=='mlp':
                self.w1=torch.nn.Parameter(torch.randn(2,8,dtype=torch.float64)*.1)
                self.b1=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
                self.w2=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
                self.b2=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
            else:
                self.w=torch.nn.Parameter(torch.zeros(8,dtype=torch.float64))
                self.b=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
                self.alpha=torch.nn.Parameter(torch.randn(8,dtype=torch.float64)*.05)
                self.gamma=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
                self.m=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))

        def forward(self, g, mu, sd):
            z=(g-mu)/sd
            if kind=='linear':
                residual=z@self.w+self.b
            elif kind=='products':
                terms=torch.cat([z,torch.stack([z[...,i]*z[...,j] for i,j in PAIRS],-1)],-1)
                residual=terms@self.w+self.b
            elif kind=='mlp':
                residual=torch.tanh(z@self.w1.T+self.b1)@self.w2+self.b2
            else:
                gates=torch.sigmoid(self.alpha)
                product=self.m*torch.prod(1-gates+gates*z,dim=-1)
                mix=torch.sigmoid(self.gamma)
                residual=(1-mix)*(z@self.w+self.b)+mix*product
            return g.sum(-1)+residual
    return Readout().to(device='cpu',dtype=torch.float64)

def export_parameters(model):
    return {k:v.detach().cpu().numpy().copy() for k,v in model.named_parameters()}
