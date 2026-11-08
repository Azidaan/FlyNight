"""A single externally guarded CUDA stage for two causal frozen-core arms."""
import os,sys,argparse
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    p=argparse.ArgumentParser();p.add_argument('video');a=p.parse_args()
    verify_phase('full_execution_recovery2_phase.json')
    import torch
    from tracking.gpu_lock import GpuLock
    from research_identity_assignment_run_20261108.full_run import rollout
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        from scripts.z13_target_model import build_arm
        _,_,backbone=build_arm('C',torch,None);weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
        for arm in ['G_FLYVIS','G_PERM']:
            rollout(a.video,arm,obs,backbone=backbone)
            assert all(torch.equal(v,weights[k]) for k,v in backbone.net.named_parameters())
    verify_phase('full_execution_recovery2_phase.json')


if __name__=='__main__':main(guard())
