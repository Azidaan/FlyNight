"""Independent torchvision Swin-B reference for loaded MASA feature backbone."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    import torch
    from torchvision.models import swin_b
    from tracking.masa_detic_prepared import build
    torch.set_default_device('cpu');torch.set_num_threads(1)
    folder=RUN/'masa_detic_backbone_reference'
    p=service.read(folder/'protocol.json')
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    model=build('cpu')
    reference=swin_b(weights=None).features.eval()
    src=model.backbone.state_dict();mapped={}
    for key,value in src.items():
        if key.startswith('patch_embed.projection.'):dest=key.replace('patch_embed.projection.','0.0.')
        elif key.startswith('patch_embed.norm.'):dest=key.replace('patch_embed.norm.','0.2.')
        elif key.startswith('stages.'):
            parts=key.split('.');stage=int(parts[1])
            if parts[2]=='blocks':
                dest=str(2*stage+1)+'.'+'.'.join(parts[3:])
                dest=dest.replace('.attn.w_msa.','.attn.').replace('.ffn.layers.0.0.','.mlp.0.').replace('.ffn.layers.1.','.mlp.3.')
            elif parts[2]=='downsample':
                dest=str(2*stage+2)+'.'+'.'.join(parts[3:])
                channels=model.backbone.stages[stage].downsample.in_channels
                # Unfold channel-major [00,01,10,11] -> torchvision offset-major [00,10,01,11].
                order=torch.cat([torch.arange(channels)*4+j for j in [0,2,1,3]])
                value=value[:,order] if parts[3]=='reduction' else value[order]
            else:raise AssertionError(key)
        elif key.startswith(('norm0.','norm1.','norm2.','norm3.')):continue
        else:raise AssertionError(key)
        mapped[dest]=value
    reference.load_state_dict(mapped,strict=True)
    reference.requires_grad_(False)
    checked=[]
    for height,width in [(64,96),(95,127)]:
        image=torch.linspace(-1,1,3*height*width).reshape(1,3,height,width)
        with torch.inference_mode():
            expected=model.backbone(image);got=[];x=image
            for index,layer in enumerate(reference):
                x=layer(x)
                if index in [3,5,7]:
                    norm=getattr(model.backbone,'norm'+str((index-1)//2))
                    got.append(norm(x).permute(0,3,1,2).contiguous())
        differences=[float((a-b).abs().max()) for a,b in zip(expected,got)]
        assert len(differences)==3
        checked.append(dict(input_shape=list(image.shape),shapes=[list(x.shape) for x in got],max_difference=differences))
        service.write(folder/'status.json',dict(checked=checked,updated=service.now()))
        assert max(differences)<=p['absolute_tolerance'],differences
    assert not torch.cuda.is_initialized()
    service.write(folder/'report.json',dict(passed=True,complete=True,checked=checked,strict_feature_parameters=True,
        independent_implementation='installed torchvision Swin-B; all feature parameters mapped, patch-merge permutation explicit',
        full_embedding_parity=False,scientific_verdict=None,hota_measured=False,updated=service.now()))

if __name__=='__main__':main()
