import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from tracking.z13_learned_association import make_association,pack_pairs,LearnedIdentityReadout,target_index


def example(torch):
    rng=np.random.default_rng(1)
    pairs=[(rng.normal(size=(2,68)),rng.normal(size=(2,3)),rng.normal(size=(3,68)),rng.normal(size=(3,3)))]
    return [torch.as_tensor(a) for a in pack_pairs(pairs)]


def test_assignment_permutation_padding_and_gradient():
    import torch
    torch.manual_seed(1);model=make_association(torch);x,g,m,y,h,n=example(torch)
    z=model(x,g,m,y,h,n);perm=[1,0,2];extended=[1,0,2,3]
    torch.testing.assert_close(model(x[:,perm],g[:,perm],m[:,perm],y,h,n),z[:,extended],atol=2e-5,rtol=1e-5)
    torch.testing.assert_close(model(x,g,m,y[:,perm],h[:,perm],n[:,perm]),z[:,:,extended],atol=2e-5,rtol=1e-5)
    assert torch.isfinite(z).all();assert z[0,2,:3].max()<-1000
    (-z[0,0,0]).backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.embed[1].weight.grad.abs().sum()>0


def test_geometry_mask_and_assignment_marginals():
    import torch
    model=make_association(torch,False);x,g,m,y,h,n=example(torch);z=model(x,g,m,y,h,n)
    torch.testing.assert_close(model(x+100,g,m,y-200,h,n),z,rtol=0,atol=0)
    a=z.exp();torch.testing.assert_close(a[:,:,:3].sum(1),n.float(),atol=.02,rtol=.02)
    torch.testing.assert_close(a[:,:3].sum(2),m.float(),atol=.02,rtol=.02)


def test_explicit_missing_and_causal_identity_recovery():
    calls=[]
    def predict(pair):
        calls.append(pair);ns,nt=len(pair[0]),len(pair[2]);z=np.full((ns+1,nt+1),-5.);z[:,-1]=0.
        if len(calls)>1:z[ns-1,0]=2.
        return z
    f=np.ones((1,68));r=LearnedIdentityReadout([0,0],f[0],predict)
    assert r.step([[0,0]],f,[1.],64)[0]==0
    assert r.step([[1,1]],f,[1.],64)[0]==-1
    assert r.step([[2,2]],f,[1.],64)[0]==0
    assert len(calls)==2


def test_unknown_annotation_is_never_negative():
    f={'centers':np.array([[0,0],[2,2]])}
    assert target_index({'kind':-1},f) is None
    assert target_index({'kind':0},f)==-1
    assert target_index({'kind':1,'gt':[0,0],'scale':2},f)==0


def test_batched_episodes_equal_sequential_and_future_cannot_change_prefix():
    import torch
    from tracking.z13_learned_association import select_sequences,predict_pairs
    torch.manual_seed(3);model=make_association(torch);model.eval();rng=np.random.default_rng(5)
    seq={e:[dict(t=t,centers=rng.normal(size=(2+(t%2),2)),features=rng.normal(size=(2+(t%2),68)),scores=np.ones(2+(t%2)),width=64,seed_center=np.zeros(2),seed_features=np.zeros(68)) for t in range(3,7)] for e in range(3)}
    batch=select_sequences(torch,[model],seq)
    for e,frames in seq.items():
        r=LearnedIdentityReadout(frames[0]['seed_center'],frames[0]['seed_features'],lambda p:predict_pairs(torch,[model],[p])[0])
        for f in frames:
            single=r.step(f['centers'],f['features'],f['scores'],f['width']);assert single[0]==batch[(e,f['t'])][0]
            assert abs(single[1]-batch[(e,f['t'])][1])<1e-4
    short=select_sequences(torch,[model],{e:frames[:-1] for e,frames in seq.items()})
    assert all(short[k]==batch[k] for k in short)


def test_synthetic_known_links_never_point_to_padding():
    from scripts.z13_learned_association_20261023 import synthetic_batch
    pool=[(np.ones((3,68),np.float32),np.zeros((3,3),np.float32),None,None)]
    arrays,y=synthetic_batch(np,np.random.default_rng(4),pool,64)
    np.testing.assert_allclose(y.sum((1,2)),1.)
    for i in range(64):
        for a,b in zip(*np.nonzero(y[i])):
            assert a==3 or arrays[2][i,a]
            assert b==3 or arrays[5][i,b]
            assert (a,b)!=(3,3)
    assert y[:,:3,3].sum()>0 and y[:,3,:3].sum()>0


def test_real_pairs_use_known_endpoints_only():
    from scripts.z13_learned_association_20261023 import build_training
    seq={0:[dict(t=t,centers=np.array([[0.,0.],[10.,10.],[20.,20.]]),features=np.ones((3,68),np.float32),scores=np.ones(3),width=64,seed_center=np.zeros(2),seed_features=np.zeros(68)) for t in range(3,11)]}
    labels=[dict(video='v',episode=0,pos=pos,kind=kind,gt=[0,0],scale=4) for pos,kind in [(0,1),(1,-1),(6,1)]]
    _,y,_,counts=build_training(labels,seq)
    assert len(y)==1 and counts=={'gap6_matched':1}
