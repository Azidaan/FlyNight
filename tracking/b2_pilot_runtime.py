"""Применение замороженных ветвей Б2; GT и выбор кадров сюда не входят."""
from __future__ import annotations
import numpy as np
import torch
from ultralytics.trackers.basetrack import BaseTrack
from tracking.xywh_bytetrack import XYWHByteTrack
from tracking.offline_association import _DetSlice
from tracking.b1_learning import ReceptorHistory
from tracking.b2_causal_detector import CausalP3Branch


class TimedReceptorHistory(ReceptorHistory):
    dt=.02
    steps=10
    def step(self,state,previous,current):
        value=current[:,None]
        alpha=1-(1-self.alpha)**(self.dt*self.steps/.2)
        state=state+(value-state)*alpha
        return state,torch.cat((state,value-state),1)


class IndependentByteTracker(XYWHByteTrack):
    """Изолированный счётчик ID, штатная ассоциация и состояние XYWHByteTrack."""
    def __init__(self,args):
        saved=BaseTrack._count
        try:super().__init__(args)
        finally:BaseTrack._count=saved
        self.own_counter=0

    def reset(self):
        saved=BaseTrack._count
        try:super().reset()
        finally:BaseTrack._count=saved
        self.own_counter=0

    def step(self,boxes,scores,classes):
        saved=BaseTrack._count
        BaseTrack._count=self.own_counter
        try:
            out=np.asarray(self.update(_DetSlice(boxes,scores,classes)),np.float32).reshape(-1,8)
        finally:
            self.own_counter=BaseTrack._count
            BaseTrack._count=saved
        return out


def assert_tracker_state_equal(a,b):
    for name in ('tracked_stracks','lost_stracks','removed_stracks'):
        left,right=getattr(a,name),getattr(b,name)
        assert len(left)==len(right)
        assert all(x.track_id==y.track_id and np.array_equal(x.mean,y.mean) and
                   np.array_equal(x.covariance,y.covariance) for x,y in zip(left,right))


def load_branches(checkpoint,device):
    """Устройство задаёт отдельный процесс ДОимпорта FlyVis, явно проверяется."""
    from tracking.z13_readout import FlyBackbone,make_modules
    from tracking.b1_fast_stimulus import fast_init
    from tracking.b1_index_copy_stimulus import index_copy_step
    from scripts.z13_target_model import build_arm
    FlyBackbone.init=fast_init;FlyBackbone.step=index_copy_step
    models={};geometry=None
    for name,state in checkpoint['models'].items():
        if name in ('C','D') or geometry is None:
            _,geom,stock=build_arm('C',torch,None)
            geometry=geom
            assert stock.dev.type==torch.device(device).type
        if name in ('C','D'):
            bb=FlyBackbone(stock.est,trainable=False)
            bb.net.load_state_dict(state['backend'],strict=True)
            bb.net.eval().requires_grad_(False)
        elif name=='R1':
            bb=TimedReceptorHistory(device)
        else:
            _,Ordinary=make_modules(torch)
            bb=Ordinary(geometry.neighbors).to(device)
            bb.dev=torch.device(device)
            bb.load_state_dict(state['backend'],strict=True)
            bb.eval().requires_grad_(False)
        branch=CausalP3Branch(128,geometry.points,geometry.eye_size).to(device)
        branch.load_state_dict(state['branch'],strict=True)
        branch.eval().requires_grad_(False)
        models[name]=(branch,bb)
    assert set(models)=={'C','D','R1','B','last','mean'}
    return models


def modified_features(model,sequence,frames,fps,mode):
    branch,backbone=model
    return branch(sequence,backbone,frames,fps,mode if mode in ('last','mean') else 'history')
