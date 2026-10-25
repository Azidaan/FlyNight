import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from tracking.z13_region_assignment import pool_activity,apply_policy,regions


def test_region_pool_and_null_have_explicit_different_support():
    x=np.arange(12).reshape(4,3)
    out=pool_activity(x,[[True]*4,[True,False,False,False]])
    np.testing.assert_allclose(out[0],x.mean(0));np.testing.assert_allclose(out[1],x[0])
    assert apply_policy([-3,-2],0)[0]==-1
    assert apply_policy([4,1],1.01)[0]==-1
    assert apply_policy([4,1],0)[0]==0


def test_real_region_support_rejects_other_class_and_unseen_center():
    class G:
        points=np.array([[4,4],[5,5],[6,6]],float);eye_size=10;spacing=2
        def nearest(self,p):return int(np.linalg.norm(self.points-p,axis=1).argmin())
    g=G();args=(g,[0,0,100,100],[50,50],[[40,40,60,60],[40,40,60,60],[200,200,220,220]],[.8,.8,.9],[1,2,1],1)
    choices,masks,geo=regions(*args)
    assert choices==[-1,0] and masks.shape==(2,3) and geo.shape==(2,8)
    shifted=list(args);shifted[1]=[100,50,200,150];shifted[2]=[150,100];shifted[3]=(np.asarray(args[3])+[100,50,100,50]).tolist()
    c2,m2,g2=regions(*shifted);assert c2==choices;assert np.array_equal(m2,masks);np.testing.assert_allclose(g2,geo)


def test_joint_region_loss_has_finite_cpu_gradient_for_null_only_rows():
    import torch
    from tracking.z13_filter_descriptor import make_ranker
    from tracking.z13_competing_candidates import marginal_choice_loss
    torch.set_num_threads(2);torch.set_default_device('cpu');torch.manual_seed(0)
    model=make_ranker(torch)
    x=torch.randn(2,3,68);seed=torch.randn(2,68);geo=torch.randn(2,3,8);mask=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
    score=model(x,seed,geo,mask);target=torch.tensor([[0,1,0],[0,0,0]],dtype=torch.float)
    loss=marginal_choice_loss(torch,score,target,mask,1.);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert next(model.parameters()).device.type=='cpu' and not torch.cuda.is_initialized()
