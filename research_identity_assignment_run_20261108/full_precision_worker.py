"""Isolated CPU or CUDA worker using the restored native-precision input path."""
import os,sys,argparse
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if 'cpu' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    p=argparse.ArgumentParser();p.add_argument('device',choices=['cpu','gpu']);p.add_argument('video');a=p.parse_args()
    verify_phase('full_precision_execution_phase.json')
    import torch
    from research_identity_assignment_run_20261108.full_precision_io import FULL,Representations,prepare_video
    from research_identity_assignment_run_20261108.full_precision_cache import recover
    import research_identity_assignment_run_20261108.full_run as runtime
    runtime.FULL=FULL;runtime.Representations=Representations;torch.set_num_threads(2)
    if a.device=='cpu':
        prepare_video(a.video,obs);recover(a.video)
        for arm in ['G','G_R1']:runtime.rollout(a.video,arm,obs)
        assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)
    else:
        from tracking.gpu_lock import GpuLock
        torch.use_deterministic_algorithms(True)
        with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
            from scripts.z13_target_model import build_arm
            _,_,backbone=build_arm('C',torch,None);weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
            for arm in ['G_FLYVIS','G_PERM']:
                runtime.rollout(a.video,arm,obs,backbone=backbone)
                assert all(torch.equal(v,weights[k]) for k,v in backbone.net.named_parameters())
    verify_phase('full_precision_execution_phase.json')


if __name__=='__main__':main(guard())
