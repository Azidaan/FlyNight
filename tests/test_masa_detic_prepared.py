import torch
from tracking.masa_detic_prepared import official_swin_classes, EvalFFN


def test_shifted_attention_matches_independent_torchvision_on_padding_and_shift():
    from torchvision.models.swin_transformer import ShiftedWindowAttention
    torch.set_num_threads(1)
    classes=official_swin_classes()
    for shift in (0,3):
        source=classes['ShiftWindowMSA'](embed_dims=32,num_heads=4,window_size=7,shift_size=shift)
        reference=ShiftedWindowAttention(32,[7,7],[shift,shift],4)
        state=source.w_msa.state_dict()
        state['relative_position_index']=state['relative_position_index'].reshape(-1)
        reference.load_state_dict(state,strict=True)
        source.eval();reference.eval()
        for height,width in ((9,13),(14,21),(8,8)):
            torch.manual_seed(height*100+width)
            x=torch.randn(1,height,width,32)
            with torch.inference_mode():
                expected=reference(x).reshape(1,height*width,32)
                actual=source(x.reshape(1,height*width,32),(height,width))
            torch.testing.assert_close(actual,expected,rtol=0,atol=1e-6)


def test_patch_merge_matches_manual_channel_first_unfold_for_odd_shapes():
    classes=official_swin_classes()
    merge=classes['PatchMerging'](3,6)
    x=torch.arange(5*7*3,dtype=torch.float32).reshape(1,35,3)
    padded=torch.nn.functional.pad(x.reshape(1,5,7,3).permute(0,3,1,2),(0,1,0,1))
    # Independent explicit lattice, channel-major then (row,column), as in nn.Unfold.
    parts=torch.stack([padded[:,:,dy::2,dx::2] for dy in (0,1) for dx in (0,1)],dim=2)
    flat=parts.reshape(1,12,-1).transpose(1,2)
    expected=merge.reduction(merge.norm(flat))
    actual,shape=merge(x,(5,7))
    assert shape==(3,4)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_ffn_shim_residual_and_eval_only():
    import pytest
    f=EvalFFN(8,32,2,0,{'type':'DropPath','drop_prob':.3},{'type':'GELU'},True,None)
    x=torch.randn(1,5,8);identity=torch.randn_like(x)
    with pytest.raises(AssertionError):f(x,identity)
    f.eval()
    expected=torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x,f.layers[0][0].weight,f.layers[0][0].bias)),f.layers[1].weight,f.layers[1].bias)+identity
    torch.testing.assert_close(f(x,identity),expected,rtol=0,atol=0)
