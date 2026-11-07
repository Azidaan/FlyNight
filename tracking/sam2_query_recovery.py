"""Та же FP32масочная ветвь, но нейронный перенос только по запросу выдачи.

Каждый исходный кадр наблюдается причинно. Между запросами сохраняются только
реальные назначения и недавние изображения. Вход не содержит меток объектов.
"""
import numpy as np
import torch
from tracking.sam2_mask_recovery import MaskOutputRecovery


class QueryMaskOutputRecovery(MaskOutputRecovery):
    @torch.inference_mode()
    def step(self,frame,image,base,lost,request_output=True):
        if request_output:return super().step(frame,image,base,lost)
        self.frames.append(frame,image)
        if self.state is None:self._init()
        active={int(row[4]) for row in base};eligible={track.track_id for track in lost}
        for ident in list(self.state['obj_ids']):
            if ident not in eligible:self.predictor.remove_object(self.state,ident,need_output=False)
        self.blocked.difference_update(active)
        for row in base:self.observed[int(row[4])]=(frame,np.asarray(row[:4],dtype=np.float32).copy())
        live=active|eligible
        self.observed={ident:value for ident,value in self.observed.items() if ident in live}
        assert all(requested<=seen for seen,requested in self.frames.requests)
        return np.zeros((0,8),dtype=np.float32)
