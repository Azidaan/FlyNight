import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

def test_null_offsets_preserve_even_outside_image_boxes():
    from tracking.pixel_box_refinement import apply_offsets
    b=np.array([-4,3,1284,724],np.float32)
    assert np.array_equal(apply_offsets(b,np.zeros(4)),b)
    assert np.array_equal(apply_offsets(b,np.full(4,.1),alpha=0),b)

def test_current_only_ignores_past_and_batch_has_no_cross_row_effect():
    import torch
    from tracking.pixel_box_refinement import make_model
    torch.set_num_threads(2);torch.manual_seed(1);model=make_model().eval()
    # Nonzero head to avoid a vacuous test through zero initialization.
    torch.nn.init.normal_(model.head[-1].weight,std=.03)
    x=torch.randn(2,2,3,96,96);m=torch.randn(2,12)
    with torch.no_grad():
        before=model(x,m,False);changed=x.clone();changed[:,0]+=100
        assert torch.equal(before,model(changed,m,False))
        one=model(x[:1],m[:1],True);batch=model(x,m,True)[:1]
        assert torch.allclose(one,batch,atol=1e-6,rtol=1e-5)
        assert (model(x,m,True).abs()<=.25).all()
