"""Same causal query schedule and cost for two frozen appearance extractors."""
import numpy as np
from tracking.lazy_appearance_association import LazyAppearanceXYWH
from tracking.frame_masa_appearance import FrameMasaServer

class QueryAppearanceTracker(LazyAppearanceXYWH):
    def __init__(self,args,server=None,weight=0.,shadow=False):
        super().__init__(args,server,weight)
        self.query_active=False;self.query_history={};self.shadow=shadow;self.shadow_pairs=0
    def reset(self):
        super().reset()
        self.query_active=False;self.query_history.clear();self.shadow_pairs=0
    def init_track(self,results,img=None):
        if self.query_active:return super().init_track(results,img)
        server=self.server;self.server=None
        try:return super().init_track(results,img)
        finally:self.server=server
    def get_dists(self,tracks,detections):
        if not self.query_active:
            return super(LazyAppearanceXYWH,self).get_dists(tracks,detections)
        original=super(LazyAppearanceXYWH,self).get_dists(tracks,detections)
        result=super().get_dists(tracks,detections)
        if self.shadow:
            self.shadow_pairs+=int(np.count_nonzero(result!=original))
            return original
        return result
    def update_query_frame(self,detections,active):
        self.query_active=bool(active)
        if active:
            for track in [*self.tracked_stracks,*self.lost_stracks]:
                track.observation=self.query_history.get(track.track_id)
        result=np.asarray(super().update(detections),np.float32).reshape(-1,8)
        if active and self.weight:
            for track in self.tracked_stracks:
                token=getattr(track,'observation',None)
                if token is not None and token.frame==self.frame_id-1:
                    self.query_history[track.track_id]=token
        alive={t.track_id for t in [*self.tracked_stracks,*self.lost_stracks]}
        self.query_history={k:v for k,v in self.query_history.items() if k in alive}
        return result

class FrozenFrameServer(FrameMasaServer):
    def __init__(self,encoder,preprocessor):
        super().__init__(encoder);self.preprocessor=preprocessor;self.current_ledger={}
    def reset(self):
        super().reset();self.current_ledger.clear()
    def start_frame(self,frame,image):
        super().start_frame(frame,image);self.current_ledger={}
    def features(self,tokens,null_seed=None):
        assert null_seed is None
        import torch
        missing=list({id(t):t for t in tokens if t.vector is None}.values())
        if missing:
            assert all(t.frame==self.frame for t in missing),'History cannot be encoded from a later image'
            with torch.inference_mode():
                if self.feats is None:
                    image,self.scale=self.preprocessor(self.image,'cuda')
                    self.feats=self.encoder.features(image);self.encoded_frames+=1
                for start in range(0,len(missing),16):
                    group=missing[start:start+16]
                    boxes=torch.as_tensor(np.stack([t.crop for t in group]),device='cuda')
                    z=torch.nn.functional.normalize(self.encoder.embed_boxes(self.feats,boxes,self.scale),dim=1).cpu().numpy()
                    assert np.isfinite(z).all()
                    for token,vector in zip(group,z):
                        token.vector=vector;self.current_ledger[token.key]=vector.copy()
                    self.embedded+=len(group)
        return np.stack([t.vector for t in tokens])

def assert_states_equal(left,right):
    for field in ['tracked_stracks','lost_stracks','removed_stracks']:
        a,b=getattr(left,field),getattr(right,field)
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert x.track_id==y.track_id
            assert np.array_equal(x.mean,y.mean) and np.array_equal(x.covariance,y.covariance)
