import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import pytest
import torch
from tracking.history_pair_features import VARIANTS,make_features
from tracking.history_pair_model import HistoryPairHead,weighted_pair_loss,corrected_cost


def sample():
    rng=np.random.default_rng(42); b,k=2,6
    h=rng.normal(size=(b,k,256)).astype(np.float32);h/=np.linalg.norm(h,axis=-1,keepdims=True)
    cur=rng.normal(size=(b,256)).astype(np.float32);cur/=np.linalg.norm(cur,axis=-1,keepdims=True)
    return dict(frame=np.array([100,100]),box=np.tile([50.,50.,70.,70.],(b,1)),
        geometry=np.zeros((b,14),np.float32),image=rng.integers(0,256,(b,3,64,64),dtype=np.uint8),
        current_embedding=cur,history_frame=np.tile(np.arange(70,100,5),(b,1)),
        history_box=np.tile([40.,50.,60.,70.],(b,k,1)),history_quality=np.ones((b,k),np.float32),
        history_valid=np.ones((b,k),bool),history_embedding=h,
        current_camera=np.ones((b,2),np.float32),history_camera=np.zeros((b,k,2),np.float32),
        current_epoch=np.zeros(b,np.int32),history_epoch=np.zeros((b,k),np.int32),video_size=np.array([320,180]))


def test_features_reject_future_and_stale_camera_is_masked():
    data=sample();data['history_frame'][0,-1]=100
    with pytest.raises(ValueError):make_features(**data,variant='134')
    data=sample();data['history_epoch'][:]=2
    f=make_features(**data,variant='134')
    assert not f['observation_valid'].any() and not f['observations'].any()
    assert f['appearance_valid'].all()


def test_last_control_ignores_earlier_content_and_mhi():
    data=sample();first=make_features(**data,variant='0')
    data['history_embedding'][:,:-1]*=-1;data['history_box'][:,:-1]+=100
    data['image'][:,1]=0
    second=make_features(**data,variant='0')
    assert all(np.array_equal(first[k],second[k]) for k in first)
    assert not np.array_equal(make_features(**sample(),variant='134')['appearance'],
        make_features(**data,variant='134')['appearance'])


def test_variants_share_architecture_but_information_is_separate():
    torch.set_num_threads(2);counts=[]
    for variant in VARIANTS:
        torch.manual_seed(42);model=HistoryPairHead()
        counts.append(sum(p.numel() for p in model.parameters()))
        f=make_features(**sample(),variant=variant)
        batch={k:torch.from_numpy(v) for k,v in f.items()}
        output=model(batch)
        loss=weighted_pair_loss(output,torch.tensor([1,0]),torch.ones(2));loss.backward()
        assert output.shape==(2,) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert not torch.cuda.is_initialized()
    assert len(set(counts))==1


def test_unknown_has_no_gradient_and_zero_correction_is_exact():
    logits=torch.tensor([.3,.5,-.4],requires_grad=True)
    loss=weighted_pair_loss(logits,torch.tensor([1,-1,0]),torch.ones(2));loss.backward()
    assert logits.grad[1]==0 and logits.grad[0]<0 and logits.grad[2]>0
    original=torch.tensor([.4,.85,.2]);eligible=torch.tensor([True,False,True])
    assert torch.equal(corrected_cost(original,logits,eligible,0),original)
    changed=corrected_cost(original,logits,eligible)
    assert changed[1]==original[1] and torch.all(torch.abs(changed-original)<=.1)
