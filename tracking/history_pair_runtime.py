"""Причинная поправка первого назначения с раздельной живой памятью вариантов."""
from collections import defaultdict,deque
from dataclasses import dataclass
import numpy as np
import torch
from ultralytics.trackers.basetrack import BaseTrack
from tracking.xywh_bytetrack import XYWHByteTrack
from tracking.offline_association import _DetSlice
from tracking.history_training_data import pair_geometry
from tracking.history_pair_features import make_features


@dataclass
class RealObservation:
    frame: int
    det: int
    box: np.ndarray
    quality: float
    embedding: np.ndarray
    camera: np.ndarray
    epoch: int


class HistoryAssociationTracker(XYWHByteTrack):
    """features — все доступные реальные области ТЕКУЩЕГО запроса, неGT.

    frame/camera и признаки передаёт общий причинный извлекатель. Внутри этого
    класса нет чтения разметки, оценки качества или изменения второй ассоциации.
    """
    def __init__(self,args,model,variant,alpha=.2):
        saved=BaseTrack._count
        try:super().__init__(args)
        finally:BaseTrack._count=saved
        self.model=model.eval();self.variant=variant;self.alpha=float(alpha)
        self.real_history=defaultdict(lambda:deque(maxlen=6))
        self._own_counter=0;self.first=False;self.context=None;self.shadow_pairs=0
        self.events=[]

    def reset(self):
        saved=BaseTrack._count
        try:super().reset()
        finally:BaseTrack._count=saved
        self.real_history.clear();self._own_counter=0;self.first=False
        self.context=None;self.shadow_pairs=0;self.events.clear()
        for name in ('raw_frame','raw_boxes','raw_scores'):
            self.__dict__.pop(name,None)

    def _first_association(self,*args):
        self.first=True
        try:return super()._first_association(*args)
        finally:self.first=False

    def get_dists(self,tracks,detections):
        matrix=super().get_dists(tracks,detections)
        if not self.first or self.context is None:return matrix
        selected=[];rows=[];K=6
        c=self.context
        for i,track in enumerate(tracks):
            past=[o for o in self.real_history[int(track.track_id)] if 0<self.raw_frame-o.frame<=60]
            if not past:continue
            for j,det in enumerate(detections):
                if matrix[i,j]>self.args.match_thresh or int(det.cls)!=int(track.cls):continue
                d=int(det.idx);assert d in c['rois']
                selected.append((i,j,d));rows.append((track,past,d))
        if not rows:return matrix
        n=len(rows);hf=np.full((n,K),-1,np.int32);hb=np.zeros((n,K,4),np.float32)
        hq=np.zeros((n,K),np.float32);valid=np.zeros((n,K),bool)
        he=np.zeros((n,K,256),np.float32);hc=np.zeros((n,K,2),np.float32);ep=np.full((n,K),-1,np.int32)
        current=[];pixels=[];geometry=[];boxes=[];cameras=[];epochs=[]
        for row,((track,past,d),(i,j,_)) in enumerate(zip(rows,selected)):
            roi=c['rois'][d];box=self.raw_boxes[d]
            current.append(roi['embedding']);pixels.append(roi['image']);boxes.append(box)
            cameras.append(c['camera']);epochs.append(c['epoch'])
            geometry.append(pair_geometry(track,box,float(self.raw_scores[d]),float(matrix[i,j]),self.raw_frame,len(past)))
            for t,o in enumerate(past):
                hf[row,t]=o.frame;hb[row,t]=o.box;hq[row,t]=o.quality;valid[row,t]=True
                he[row,t]=o.embedding;hc[row,t]=o.camera;ep[row,t]=o.epoch
        f=make_features(frame=np.full(n,self.raw_frame,np.int32),box=np.asarray(boxes,np.float32),geometry=np.asarray(geometry,np.float32),
            image=np.asarray(pixels,np.uint8),current_embedding=np.asarray(current,np.float32),
            history_frame=hf,history_box=hb,history_quality=hq,history_valid=valid,history_embedding=he,
            current_camera=np.asarray(cameras,np.float32),history_camera=hc,current_epoch=np.asarray(epochs,np.int32),
            history_epoch=ep,video_size=np.asarray(c['video_size']),variant=self.variant)
        device=next(self.model.parameters()).device
        with torch.no_grad():scores=self.model({k:torch.from_numpy(v).to(device) for k,v in f.items()}).cpu().numpy()
        assert np.isfinite(scores).all();self.shadow_pairs+=n
        corrected=matrix.copy()
        for row,(i,j,d) in enumerate(selected):
            delta=float(self.alpha*(.5-1/(1+np.exp(-np.clip(float(scores[row]),-80,80)))))
            if self.alpha!=0:corrected[i,j]=matrix[i,j]+delta
            self.events.append(dict(frame=self.raw_frame,track=int(tracks[i].track_id),det=d,original=float(matrix[i,j]),score=float(scores[row]),corrected=float(corrected[i,j]),history_frames=[o.frame for o in rows[row][1]]))
        return corrected

    def update_frame(self,frame,boxes,scores,classes,context=None):
        if hasattr(self,'raw_frame') and int(frame)<=self.raw_frame:
            raise ValueError('Strictly increasing observed frames required; reset before a new video')
        self.raw_frame=int(frame);self.raw_boxes=boxes;self.raw_scores=scores;self.context=context
        saved=BaseTrack._count;BaseTrack._count=self._own_counter
        try:
            output=np.asarray(self.update(_DetSlice(boxes,scores,classes)),np.float32).reshape(-1,8)
            self._own_counter=BaseTrack._count
        finally:BaseTrack._count=saved
        if context is not None:
            for row in output:
                d=int(row[7]);assert d in context['rois']
                roi=context['rois'][d]
                self.real_history[int(row[4])].append(RealObservation(frame=int(frame),det=d,
                    box=np.asarray(boxes[d]).copy(),quality=float(scores[d]),embedding=roi['embedding'].copy(),
                    camera=np.asarray(context['camera'],np.float32).copy(),epoch=int(context['epoch'])))
        alive={int(t.track_id) for t in self.tracked_stracks+self.lost_stracks}
        for ident in set(self.real_history)-alive:
            del self.real_history[ident]
        return output
