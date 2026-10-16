import numpy as np
import torch
from types import SimpleNamespace
from tracking.z13_candidate_modules import transport_indices, make_candidate, box_iou, run_candidate
from tracking.z13_spatial import make_spatial_readout

def grid():
    # Tests use an explicit 721-column regular grid, not a FlyVis GPU runtime.
    points=np.stack([np.arange(721)%31,np.arange(721)//31],-1).astype(float)
    neighbors=np.arange(721)[:,None].repeat(7,1)
    return points,neighbors

def test_transport_identity_translation_and_missing():
    pts=np.array([[0.,0.],[1.,0.],[2.,0.]])
    index,valid=transport_indices(pts,1.,3.,[0,0,3,3],[0,0,3,3])
    assert np.array_equal(index,[0,1,2]) and valid.all()
    index,valid=transport_indices(pts,1.,3.,[0,0,3,3],[1,0,4,3])
    assert np.array_equal(index[:2],[1,2]) and valid.tolist()==[True,True,False]

def test_size_initializes_as_identity_and_iou():
    p,n=grid(); ro=make_candidate(torch,make_spatial_readout(torch))(34,p,31,n,'size')
    f=torch.randn(2,34,721); mask=torch.zeros(2,721,dtype=torch.bool); mask[:,:20]=True
    ex=ro.exemplar(f,mask); logits=ro.logits(ex,f)
    assert torch.equal(ro.size_delta(ex,f,logits),torch.zeros(2,2))
    assert box_iou([5,5],[10,10],[0,0,10,10])==1
    assert box_iou([5,5],[20,20],[0,0,10,10])==.25

def test_background_zero_gain_no_context_and_gradient():
    p,n=grid(); cls=make_spatial_readout(torch); base=cls(34,p,31,n)
    ro=make_candidate(torch,cls)(34,p,31,n,'foreground'); ro.load_state_dict(base.state_dict(),strict=False)
    f=torch.randn(2,34,721); mask=torch.zeros(2,721,dtype=torch.bool); mask[0,:20]=True; mask[1,:]=True
    got=ro.logits(ro.exemplar(f,mask),f); want=base.logits(base.exemplar(f,mask),f)
    assert torch.allclose(got,want)
    with torch.no_grad(): ro.background_gain.fill_(1)
    got=ro.logits(ro.exemplar(f,mask),f)
    assert torch.allclose(got[1],want[1])
    got[0,0].backward(); assert ro.background_gain.grad is not None and torch.isfinite(ro.background_gain.grad)

def test_adaptive_starts_equal_fixed_multiple_frames():
    p,n=grid(); cls=make_candidate(torch,make_spatial_readout(torch))
    fixed=cls(34,p,31,n,'fixed'); adaptive=cls(34,p,31,n,'adaptive')
    adaptive.load_state_dict(fixed.state_dict(),strict=False)
    class Backbone:
        def init(self,x): return None
        def step(self,state,last,cur): return None,cur[:,None].expand(-1,34,-1)
    geom=SimpleNamespace(points=p,spacing=1)
    hx=torch.randn(1,7,721); mask=torch.zeros(1,721,dtype=torch.bool); mask[:,:20]=True
    crops=np.tile([0,0,31,31],(1,7,1)); crops[:,5:]+=np.array([1,0,1,0])
    a,_=run_candidate(torch,Backbone(),fixed,hx,mask,crops,geom,31)
    b,_=run_candidate(torch,Backbone(),adaptive,hx,mask,crops,geom,31)
    assert torch.allclose(a,b,atol=1e-6)
    b.sum().backward(); assert adaptive.time_bias.grad is not None


def test_checkpoint_restores_real_next_optimizer_step(tmp_path):
    from tracking.z13_checkpoint import save_checkpoint, restore_checkpoint
    p,n=grid(); ro=make_candidate(torch,make_spatial_readout(torch))(34,p,31,n,'adaptive')
    core=torch.nn.Linear(2,2)
    opt=torch.optim.Adam(ro.parameters(),lr=.003)
    rng=np.random.default_rng(7)
    def advance():
        opt.zero_grad()
        sum(q.square().sum() for q in ro.parameters()).backward()
        opt.step()
    advance()
    path=tmp_path/'state.pt'; ident={'test':'roundtrip'}
    save_checkpoint(torch,path,'B',core,ro,opt,rng,1,[],ident)
    a=rng.integers(100,size=8); b=torch.rand(8)
    advance(); expected={k:v.clone() for k,v in ro.state_dict().items()}
    step,_=restore_checkpoint(torch,path,'B',core,ro,opt,rng,ident)
    assert step==1 and np.array_equal(a,rng.integers(100,size=8)) and torch.equal(b,torch.rand(8))
    advance()
    assert all(torch.equal(v,expected[k]) for k,v in ro.state_dict().items())
