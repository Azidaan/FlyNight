import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_spatial_signals_20261108.common import *


def main(obs):
    from tracking.gpu_lock import GpuLock
    with GpuLock('flyvis_fovea_gpu'):
        import torch
        torch.cuda.init();assert torch.cuda.is_available()
        import research_singleton_verify_20261108.extract_features as source
        source.RUN=RUN;source.BASE=RUN/'reuse_pool';source.phase=phase;source.status=status;source.main(obs)
    write(RUN/'data_complete.json',dict(passed=True,features_sha256=sha(RUN/'features_complete.json')),new=True)


if __name__=='__main__':main(guard())
