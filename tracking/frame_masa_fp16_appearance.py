"""Отдельный смешанныйFP16 MASA; косинусная стоимость остаётсяFloat32."""
import numpy as np
from tracking.frame_masa_appearance import FrameMasaServer as Original
class FrameMasaServer(Original):
    def features(self,tokens,null_seed=None):
        if null_seed is not None:return super().features(tokens,null_seed)
        import torch
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            super().features(tokens,None)
        for t in tokens:
            if t.vector.dtype!=np.float32:t.vector=t.vector.astype(np.float32)
        return np.stack([t.vector for t in tokens])
