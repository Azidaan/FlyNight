import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_full_20261108.common import *


def main(obs):
    assert read(RECOVERY/'views_complete.json')['passed']
    from tracking.gpu_lock import GpuLock
    with GpuLock('flyvis_fovea_gpu'):
        import torch
        torch.cuda.init();assert torch.cuda.is_available()
        import research_singleton_free_20261108.full_features as module
        original=module.Receptors
        class ReadyViews(original):
            @property
            def gray(self):raise RuntimeError('A required view was not prepared. Full-video decoding fallback is forbidden.')
        module.Receptors=ReadyViews
        from research_bbox_normalized_full_20261108.features_logic import main as compute
        compute(obs)


if __name__=='__main__':main(guard())
