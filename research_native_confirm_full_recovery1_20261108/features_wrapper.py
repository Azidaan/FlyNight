"""Use only prepared validated views; forbid fallback decoded-video regeneration."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_full_recovery1_20261108.common import *


def main(obs):
    recovery_phase();assert read(RECOVERY/'views_complete.json')['passed']
    from tracking.gpu_lock import GpuLock
    with GpuLock('flyvis_fovea_gpu'):
        import torch
        torch.cuda.init()
        import research_singleton_free_20261108.full_features as module
        original=module.Receptors
        class ReadyViews(original):
            @property
            def gray(self):raise RuntimeError('A required view was not prepared; decoded full-video fallback is forbidden in this recovery.')
        module.Receptors=ReadyViews
        import research_native_confirm_full_20261108.features as frozen
        try:frozen.main(obs)
        finally:module.Receptors=original
    recovery_phase()


if __name__=='__main__':main(guard())
