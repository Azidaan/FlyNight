"""Признаки текущего полного кадра MASA для прежнего правила реальных назначений."""
import hashlib
import numpy as np
from tracking.lazy_appearance_association import AppearanceServer,Observation

class FrameMasaServer(AppearanceServer):
    def __init__(self,encoder):
        super().__init__(encoder);self.feats=None;self.scale=None;self.finished=True;self.encoded_frames=0
    def reset(self):
        super().reset()
        self.feats=None;self.scale=None;self.finished=True;self.encoded_frames=0
    def start_frame(self,frame,image):
        assert self.finished,'Сначала сохранить признаки реальных наблюдений предыдущего кадра'
        super().start_frame(frame,image);self.feats=None;self.scale=None;self.finished=False
    def observation(self,box):
        b=np.asarray(box,np.float32);key=(self.frame,*b.tolist())
        if key not in self.tokens:self.tokens[key]=Observation(key,self.frame,b.copy())
        return self.tokens[key]
    def features(self,tokens,null_seed=None):
        assert all(t.frame<=self.frame for t in tokens)
        if null_seed is not None:
            for t in tokens:
                if null_seed not in t.nulls:
                    seed=int.from_bytes(hashlib.sha256((str(t.key)+'/'+str(null_seed)).encode()).digest()[:8],'little')
                    z=np.random.default_rng(seed).normal(size=256).astype(np.float32);t.nulls[null_seed]=z/np.linalg.norm(z)
            return np.stack([t.nulls[null_seed] for t in tokens])
        missing=list({id(t):t for t in tokens if t.vector is None}.values())
        if missing:
            assert all(t.frame==self.frame for t in missing),'Прошлые изображения не пересчитываются будущим контекстом'
            import torch
            from tracking.masa_r50 import preprocess
            with torch.inference_mode():
                if self.feats is None:
                    x,self.scale=preprocess(self.image,'cuda');self.feats=self.encoder.features(x);self.encoded_frames+=1
                for start in range(0,len(missing),16):
                    group=missing[start:start+16];boxes=torch.from_numpy(np.stack([t.crop for t in group])).to('cuda')
                    z=torch.nn.functional.normalize(self.encoder.embed_boxes(self.feats,boxes,self.scale),dim=1).cpu().numpy()
                    assert np.isfinite(z).all() and z.shape==(len(group),256)
                    for t,v in zip(group,z):t.vector=v
                    self.embedded+=len(group)
        return np.stack([t.vector for t in tokens])
    def finish_frame(self):
        # Любая фактически назначенная текущая область должна иметь сохранённый
        # вектор до следующего кадра. Общие признаки кадра далее освобождаются.
        pending=[t for t in self.tokens.values() if t.frame==self.frame and t.vector is None]
        if pending:self.features(pending)
        self.finished=True;self.feats=None;self.image=None
