"""Адаптация BUSCA: проверять только дополнительную выдачу, не менять трекер.

Это не воспроизведение авторского полного трекера. Память — последние 11
уверенных реальных назначений. Нет компенсации камеры и глобального фильтра
плотности. Выбор кандидатов и отказ выполняет исходная обученная сеть BUSCA.
"""
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch


def valid_box(box,image_shape):
    box=np.asarray(box,dtype=np.float32)
    h,w=image_shape[:2]
    return (bool(np.isfinite(box).all()) and box[2]>box[0] and box[3]>box[1]
            and box[0]<w and box[1]<h and box[2]>0 and box[3]>0)


class BuscaOutputRecovery:
    def __init__(self,model,high_threshold=.25,memory_length=11,batch_size=4):
        self.model=model;self.high_threshold=high_threshold;self.memory_length=memory_length;self.batch_size=batch_size
        self.history={};self.calls=0;self.considered=0;self.accepted=0;self.strict_past=True

    def remember(self,frame,output,image):
        from busca.tracking import get_bbox_crop
        for row in output:
            if row[5]<self.high_threshold or not valid_box(row[:4],image.shape):continue
            ident=int(row[4]);history=self.history.setdefault(ident,deque(maxlen=self.memory_length))
            assert not history or history[-1][0]<frame
            box=np.array(row[:4],dtype=np.float32)
            crop=get_bbox_crop(image,box,normalize=False)
            history.append((frame,box,crop))

    def forget_removed(self,live_ids):
        self.history={ident:items for ident,items in self.history.items() if ident in live_ids}

    @staticmethod
    def object(crops,boxes):
        ltwh=np.array(boxes,dtype=np.float32).copy();ltwh[:,2:]-=ltwh[:,:2]
        return SimpleNamespace(images_mem=list(crops),tlwh_mem=list(ltwh),tlwh=ltwh[-1],scale=1.)

    def select(self,frame,lost,image,xyxy,conf,low_threshold=.1):
        from busca.tracking import get_bbox_crop
        eligible=[]
        for track in lost:
            history=self.history.get(track.track_id,())
            if len(history)!=self.memory_length or not valid_box(track.xyxy,image.shape):continue
            assert all(f<frame for f,_,_ in history),'Current/future region leaked into history'
            eligible.append((track,history))
        if not eligible:return []
        dets=[];det_boxes=[]
        for box,score in zip(xyxy,conf):
            if score>=low_threshold and valid_box(box,image.shape):
                det_boxes.append(np.asarray(box));dets.append(self.object([get_bbox_crop(image,box,normalize=False)],[box]))
        accepted=[]
        for offset in range(0,len(eligible),self.batch_size):
            chunk=eligible[offset:offset+self.batch_size];mem=[];proposals=[];centers=[]
            for track,history in chunk:
                mem.append(self.object([crop for _,_,crop in history],[box for _,box,_ in history]))
                box=np.asarray(track.xyxy);centers.append((box[:2]+box[2:])/2)
                proposals.append(self.object([get_bbox_crop(image,box,normalize=False)],[box]))
            centers=np.array(centers)
            dc=np.array([(box[:2]+box[2:])/2 for box in det_boxes]).reshape(-1,2)
            dists=np.sqrt(np.sum((centers[:,None]-dc[None])**2,axis=-1))
            with torch.inference_mode():
                probs,reliable=self.model.associate_embeddings(mem,dets,dists,seq_len=self.memory_length,num_candidates=5,
                    use_broader_memory=False,select_highest_candidate=True,highest_candidate_minimum_thresh=None,
                    extra_kalman_candidates=proposals,normalize_ims=True,plot_results=False)
            assert probs is not None and np.isfinite(probs).all()
            for i,(track,_) in enumerate(chunk):
                if reliable[i] and probs[i,len(dets)+i]>.5:accepted.append(track)
            self.calls+=1;self.considered+=len(chunk)
        self.accepted+=len(accepted)
        return accepted
