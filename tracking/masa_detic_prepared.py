"""Inference-only MASA-Detic preparation, separate from every frozen experiment.

Swin class bodies are the SHA-pinned reviewed OpenMMLab implementation (Apache
2.0), with only dependency construction shims. No dependency installation,
training, detector replacement or silent random parameters. Full numerical
acceptance remains a separate gate before scientific tracker experiments.
"""
from pathlib import Path
import ast, hashlib, math, warnings
from copy import deepcopy
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Optional, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import roi_align
from tracking.masa_r50 import ConvBN, DyHeadBlock, EmbedHead, preprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/continuation_20261028/reid_candidates_preparation/masa'
WEIGHTS=ROOT/'weights/masa_detic_author_20261029/detic_masa.pth'
EXPECTED='10c19938af1b70c8bea1ca4a49139198abf2c2cc77c42e8372bfeb0a4e461879'

class BaseModule(nn.Module):
    def __init__(self,init_cfg=None):
        super().__init__();self.init_cfg=init_cfg

class EvalFFN(nn.Module):
    def __init__(self,embed_dims,feedforward_channels,num_fcs,ffn_drop,dropout_layer,act_cfg,add_identity,init_cfg):
        super().__init__()
        assert num_fcs==2 and ffn_drop==0 and act_cfg=={'type':'GELU'} and add_identity
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(embed_dims,feedforward_channels),nn.GELU(),nn.Dropout(0)),nn.Linear(feedforward_channels,embed_dims)])
    def forward(self,x,identity=None):
        assert not self.training,'Inference only'
        return self.layers[1](self.layers[0](x))+(x if identity is None else identity)

def to_2tuple(value):return value if isinstance(value,Iterable) else (value,value)

def build_norm_layer(config,channels):
    assert config['type']=='LN'
    assert set(config).issubset({'type','eps'})
    return 'ln',nn.LayerNorm(channels,eps=config.get('eps',1e-5))

def build_conv_layer(config,**kwargs):
    assert config=={'type':'Conv2d'}
    return nn.Conv2d(**kwargs)

def official_swin_classes():
    namespace=dict(globals(),ModuleList=nn.ModuleList,FFN=EvalFFN,OptConfigType=Optional[dict],
                   build_dropout=lambda config:nn.Identity())
    specs=[('reference_layers_mmdet330.py','5eb55105ba19beef52b01103713396c7af90f4fe3f430360679937c3b5d39511',
            {'AdaptivePadding','PatchEmbed','PatchMerging'}),
           ('reference_swin_mmdet330.py','a239779a44a94c03cfb4c8605f589989e707715e2d57ecf00a945fb47a96c370',
            {'WindowMSA','ShiftWindowMSA','SwinBlock','SwinBlockSequence','SwinTransformer'})]
    for filename,digest,names in specs:
        raw=(SOURCE/filename).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==digest,'Reviewed upstream source changed'
        tree=ast.parse(raw)
        classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
        assert {x.name for x in classes}==names
        for node in classes:
            node.decorator_list=[]
            node.body=[x for x in node.body if not (isinstance(x,ast.FunctionDef) and x.name=='init_weights')]
        code=ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[]))
        exec(compile(code,filename,'exec'),namespace)
    return namespace

class DeticFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs=nn.ModuleList([ConvBN(c,256,1) for c in (256,512,1024)])
        self.fpn_convs=nn.ModuleList([ConvBN(256,256,3) for _ in range(3)])
    def forward(self,features):
        assert len(features)==3
        lateral=[layer(x) for layer,x in zip(self.lateral_convs,features)]
        for i in (2,1):lateral[i-1]=lateral[i-1]+F.interpolate(lateral[i],size=lateral[i-1].shape[-2:],mode='nearest')
        outputs=[layer(x) for layer,x in zip(self.fpn_convs,lateral)]
        while len(outputs)<5:outputs.append(F.max_pool2d(outputs[-1],1,stride=2))
        return outputs

class MasaDeticPrepared(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=official_swin_classes()['SwinTransformer'](embed_dims=128,depths=(2,2,18,2),num_heads=(4,8,16,32),
            window_size=7,mlp_ratio=4,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0,drop_path_rate=.3,
            patch_norm=True,out_indices=(1,2,3),with_cp=False)
        self.fpn=DeticFPN();self.fusion=nn.ModuleList([DyHeadBlock() for _ in range(3)]);self.embed=EmbedHead()
    def features(self,image):
        assert not self.training,'No training or mutation of frozen weights'
        x=self.fpn(self.backbone(image))
        for layer in self.fusion:x=layer(x)
        return x
    def embed_boxes(self,features,boxes,scale):
        if len(boxes)==0:return features[0].new_zeros((0,256))
        boxes=boxes*scale
        side=((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])).clamp(min=1e-6).sqrt()
        levels=torch.floor(torch.log2(side/56+1e-6)).clamp(0,2).long()
        output=features[0].new_zeros((len(boxes),256,7,7))
        for i,stride in enumerate((8,16,32)):
            selected=levels==i
            if selected.any():
                region=torch.cat([boxes.new_zeros((int(selected.sum()),1)),boxes[selected]],1)
                output[selected]=roi_align(features[i],region.to(features[i]),7,spatial_scale=1/stride,sampling_ratio=0,aligned=True)
        return self.embed(output)

def build(device='cpu'):
    h=hashlib.sha256()
    with WEIGHTS.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    assert h.hexdigest()==EXPECTED,'Upstream checkpoint hash mismatch'
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)['state_dict']
    model=MasaDeticPrepared();mapped={}
    prefixes={'detector.backbone.':'backbone.','masa_adapter.0.':'fpn.',
              'masa_adapter.1.dyhead_blocks.':'fusion.','track_head.embed_head.':'embed.'}
    for key,value in state.items():
        for source,dest in prefixes.items():
            if key.startswith(source):mapped[dest+key[len(source):]]=value;break
    # Every required tensor is loaded: no permissive shape filtering or random fill.
    model.load_state_dict(mapped,strict=True)
    model.eval();model.to(device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model
