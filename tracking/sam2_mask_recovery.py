"""Собственная причинная масочная ветвь на SAM2.1tiny; не LTR и не SGAR-MOT."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tools/sam2-runtime-20261027'
DEPS=ROOT/'tools/sam2-deps-20261027'
WEIGHTS=ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt'


class CausalFrames:
    """Только уже декодированные кадры. Любое чтение будущего — ошибка."""
    def __init__(self,total,image_size=1024,keep=32):
        self.total=total;self.image_size=image_size;self.keep=keep;self.raw=OrderedDict();self.latest=-1;self.requests=[]
    def __len__(self):return self.total
    def append(self,index,image):
        assert index==self.latest+1
        self.raw[index]=image.copy();self.latest=index
        while len(self.raw)>self.keep:self.raw.popitem(last=False)
    def __getitem__(self,index):
        assert 0<=index<=self.latest,'Future image requested'
        assert index in self.raw,'Past frame outside retained prompt window'
        self.requests.append((self.latest,index))
        from PIL import Image
        rgb=np.array(Image.fromarray(self.raw[index][...,::-1]).resize((self.image_size,self.image_size)))
        tensor=torch.from_numpy(rgb/255.).permute(2,0,1).float()
        mean=torch.tensor([.485,.456,.406])[:,None,None]
        std=torch.tensor([.229,.224,.225])[:,None,None]
        return (tensor-mean)/std


def load_predictor(device='cuda'):
    for path in (SOURCE,DEPS):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    provenance=json.loads((ROOT/'runs/b123_20261027/sam2_weights_download/report.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==provenance['sha256']
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(WEIGHTS),device=device,apply_postprocessing=False)
    # Официальный загрузчик проверяет все отсутствующие и лишние ключи.
    assert predictor.fill_hole_area==0 and not predictor.training
    return predictor


class MaskOutputRecovery:
    def __init__(self,predictor,total_frames):
        self.predictor=predictor;self.frames=CausalFrames(total_frames,predictor.image_size)
        self.state=None;self.observed={};self.blocked=set();self.initialized=0;self.propagated=0;self.rejected=0

    def _init(self):
        # Сохраняется официальный init_state; заменяется только ввод изображений.
        import sam2.sam2_video_predictor as module
        original=module.load_video_frames
        def stream(video_path,**kw):
            assert video_path is self.frames
            image=next(iter(self.frames.raw.values()))
            return self.frames,image.shape[0],image.shape[1]
        module.load_video_frames=stream
        try:
            # init_state прогревает frame0. Он должен быть вызван на первом кадре.
            assert self.frames.latest==0
            self.state=self.predictor.init_state(self.frames,offload_video_to_cpu=True,offload_state_to_cpu=True)
        finally:module.load_video_frames=original

    @staticmethod
    def duplicate(box,base):
        if not len(base):return False
        x1=np.maximum(box[:2],base[:,:2]);x2=np.minimum(box[2:],base[:,2:4])
        inter=np.maximum(x2-x1,0).prod(axis=1)
        union=(box[2]-box[0])*(box[3]-box[1])+(base[:,2]-base[:,0])*(base[:,3]-base[:,1])-inter
        return bool(np.any(inter/np.maximum(union,1e-9)>.9))

    @torch.inference_mode()
    def step(self,frame,image,base,lost):
        self.frames.append(frame,image)
        if self.state is None:self._init()
        active={int(row[4]) for row in base};eligible={track.track_id:track for track in lost}
        # Возврат к реальному назначению завершает масочный эпизод.
        for ident in list(self.state['obj_ids']):
            if ident not in eligible:self.predictor.remove_object(self.state,ident,need_output=False)
        self.blocked.difference_update(active)
        for ident,track in eligible.items():
            if ident in self.blocked or ident in self.state['obj_ids'] or ident not in self.observed:continue
            previous,box=self.observed[ident]
            assert previous<frame
            if previous not in self.frames.raw:continue
            self.predictor.add_new_points_or_box(self.state,frame_idx=previous,obj_id=ident,box=box)
            self.initialized+=1
        extra=[]
        if self.state['obj_ids']:
            # В API конец включителен: ноль означает ровно текущий кадр.
            for index,identifiers,masks in self.predictor.propagate_in_video(self.state,start_frame_idx=frame,max_frame_num_to_track=0):
                assert index==frame
                reject=[]
                for k,ident in enumerate(identifiers):
                    objidx=self.state['obj_id_to_idx'][ident]
                    out=self.state['output_dict_per_obj'][objidx]['non_cond_frame_outputs'].get(frame)
                    if out is None:out=self.state['output_dict_per_obj'][objidx]['cond_frame_outputs'][frame]
                    visible=float(out['object_score_logits'].flatten()[0])>0
                    mask=masks[k,0]>0
                    yy,xx=torch.where(mask)
                    self.propagated+=1
                    if not visible or not len(xx):reject.append(ident);continue
                    box=np.array([int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],dtype=np.float32)
                    if self.duplicate(box,base):reject.append(ident);continue
                    track=eligible[ident]
                    extra.append([*box,track.track_id,track.score,track.cls,-1])
                for ident in reject:
                    self.predictor.remove_object(self.state,ident,need_output=False);self.blocked.add(ident);self.rejected+=1
        for row in base:self.observed[int(row[4])]=(frame,np.asarray(row[:4],dtype=np.float32).copy())
        live=active|set(eligible)
        self.observed={ident:value for ident,value in self.observed.items() if ident in live}
        assert all(requested<=at for at,requested in self.frames.requests)
        return np.asarray(extra,dtype=np.float32).reshape(-1,8)
