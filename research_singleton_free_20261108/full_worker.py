"""One separately guarded CPU/CUDA process; no heavy nested subprocesses."""
import os,sys,argparse
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if 'cpu' in sys.argv or 'zero' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main(obs):
    parser=argparse.ArgumentParser();parser.add_argument('device',choices=['cpu','gpu','zero']);parser.add_argument('video');args=parser.parse_args()
    phase('full_phase.json')
    import torch
    from research_singleton_free_20261108.full_runtime import rollout
    from research_singleton_free_20261108.full_features import FULL
    torch.set_num_threads(2);p=read(FULL/'protocol.json')
    if args.device=='zero':rollout(args.video,'ZERO',obs,zero=True)
    elif args.device=='cpu':
        for arm in p['cpu_arms']:rollout(args.video,arm,obs)
    else:
        from tracking.gpu_lock import GpuLock
        from scripts.z13_target_model import build_arm
        torch.use_deterministic_algorithms(True)
        with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
            _,_,backbone=build_arm('C',torch,None);assert not backbone.net.training
            weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
            for arm in p['gpu_arms']:
                rollout(args.video,arm,obs,backbone)
                assert all(torch.equal(v,weights[k]) for k,v in backbone.net.named_parameters())
    if args.device!='gpu':assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)
    phase('full_phase.json')


if __name__=='__main__':main(guard())
