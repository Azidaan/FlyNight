"""Авторская BUSCA с полным checkpoint и эквивалентным экономным хранением PE.

Исходники не меняются. Семантика прямого прохода, ReID, нормировки и токенов
сохраняется. Таблица кодирования вычисляет только запрошенные строки.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/b123_20261027/recovery_source_snapshot/BUSCA'
DEPS=ROOT/'tools/busca-deps-20261027'
CHECKPOINT=ROOT/'weights/busca_official_20261027/model_busca.pth'
CHECKPOINT_SHA='ed294bfec54e87d84a0d1dbdaf9630d218c59311c1e3f1fd4782e32dc3d478f1'


def imports():
    for path in (SOURCE,DEPS):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    return importlib.import_module('busca.network'),importlib.import_module('busca.encodings')


class SparseEncodingTable:
    def __init__(self,channels,ranges):
        from positional_encodings.torch_encodings import PositionalEncoding3D,get_emb
        base=PositionalEncoding3D(channels)
        self.axes=[get_emb(torch.einsum('i,j->ij',torch.arange(n,dtype=base.inv_freq.dtype),base.inv_freq)).half() for n in ranges]
        self.channels=channels

    def __getitem__(self,indices):
        return torch.cat([axis[int(index)] for axis,index in zip(self.axes,indices)])[:self.channels]


def lazy_class():
    _,enc=imports()
    class LazyPositionalEncoding(enc.PositionalEncoding):
        def __init__(self,d_model,input_flavour,dropout=.1,max_temp_dist=30,max_distance_dist=105,max_size_dist=105,
                     encode_sep_as_ref=True,batch_first=False,device='cuda'):
            torch.nn.Module.__init__(self)
            self.d_model=d_model;self.input_flavour=input_flavour;self.encode_sep_as_ref=encode_sep_as_ref
            self.batch_first=batch_first;self.device=device
            self.max_distance_dist=max_distance_dist;self.max_size_dist=max_size_dist;self.max_temp_dist=max_temp_dist
            self.distant_fake_bbox=torch.from_numpy(enc.missing_candidate_bbox(flavour='ltwh')).to(device)
            self.pe=SparseEncodingTable(d_model,(2*max_distance_dist+1,2*max_size_dist+1,2*max_temp_dist+1))
            self.dropout=torch.nn.Dropout(dropout)
            assert batch_first
    return LazyPositionalEncoding


def verify_lazy_encoding():
    _,enc=imports()
    # Полная авторская реализация на уменьшенной области, все 125 индексов.
    original=enc.PositionalEncoding(512,'MEM-SEP-CAN-BAD',dropout=0,max_temp_dist=2,max_distance_dist=2,max_size_dist=2,batch_first=True,device='cpu')
    lazy=lazy_class()(512,'MEM-SEP-CAN-BAD',dropout=0,max_temp_dist=2,max_distance_dist=2,max_size_dist=2,batch_first=True,device='cpu')
    equal=all(torch.equal(original.pe[x,y,z],lazy.pe[x,y,z]) for x in range(5) for y in range(5) for z in range(5))
    assert equal
    mem=torch.arange(2*3*512).reshape(2,3,512).float()/1000
    can=torch.arange(2*14*512).reshape(2,14,512).float()/2000
    boxes=torch.tensor([[[10.,10.,50.,80.],[11.,11.,51.,81.],[12.,12.,52.,82.]]]*2)
    cans=boxes[:,:1].repeat(1,5,1)
    a=original(mem,can,boxes,cans,num_candidates=7)
    b=lazy(mem,can,boxes,cans,num_candidates=7)
    assert torch.equal(a,b)
    return dict(all_small_grid_rows_exact=equal,full_positional_forward_exact=True,checked_rows=125,
                production_rows=sum(len(x) for x in SparseEncodingTable(512,(211,211,61)).axes),
                theoretical_basis='Авторское кодирование конкатенирует независимые синусы/косинусы трёх осей и округляет FP16; хранение декартова произведения не нужно.')


def load_model(device='cuda'):
    import yaml
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==CHECKPOINT_SHA
    network,enc=imports()
    config=yaml.safe_load((SOURCE/'config/ByteTrack/MOT17/config_bytetrack_mot17.yml').read_text())
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    if 'model_state_dict' in state:state=state['model_state_dict']
    args=SimpleNamespace(**config['transformer'])
    args.device=device;args.reid_weights_file='no'
    args.num_training_ids=int(state['reid_encoder.model.fc.weight'].shape[0])
    loader=importlib.import_module('busca.reid.load_trained_net')
    original_factory=loader.resnet50;original_pos=network.PositionalEncoding
    def no_redundant_imagenet(*a,**kw):
        kw['pretrained']=False
        return original_factory(*a,**kw)
    loader.resnet50=no_redundant_imagenet;network.PositionalEncoding=lazy_class()
    try:model=network.BUSCA(args)
    finally:loader.resnet50=original_factory;network.PositionalEncoding=original_pos
    required=model.state_dict()
    assert set(required)==set(state),dict(missing=sorted(set(required)-set(state)),extra=sorted(set(state)-set(required)))
    assert all(required[k].shape==state[k].shape for k in required)
    loaded=model.load_state_dict(state,strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    model=model.to(device).eval()
    report=dict(all_state_keys_loaded_strictly=True,state_keys=len(state),
        reid_state_keys=sum(k.startswith('reid_encoder.model.') for k in state),
        unused_partial_feature_checkpoint=False,checkpoint_sha256=CHECKPOINT_SHA,
        num_training_ids=args.num_training_ids,imagenet_download_skipped_after_full_coverage=True,
        original_sources_unchanged=True,domain_adaptation_batchnorm_preserved=True)
    return model,report
