"""Same frozen rollout, with retried correctly routed progress only."""
import os,sys,argparse
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['CUDA_VISIBLE_DEVICES']='0' if 'gpu' in sys.argv else ''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_recovery1_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import for_run


def patch_engine():
    import research_singleton_verify_full_20261108.engine as engine
    engine.status=for_run(RUN);engine.feature_module.status=engine.status
    return engine


def main(obs):
    import torch
    phase('full_phase.json');recovery_phase();parser=argparse.ArgumentParser();parser.add_argument('role',choices=['cpu','gpu','zero']);parser.add_argument('video');a=parser.parse_args()
    torch.set_num_threads(2);p=read(FULL/'protocol.json')
    if a.role=='gpu':
        from tracking.gpu_lock import GpuLock
        torch.use_deterministic_algorithms(True)
        with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
            assert torch.cuda.is_available();torch.cuda.init()
            from tracking.flyvis_adapter import activate_flyvis
            activate_flyvis(device='cuda');engine=patch_engine()
            from scripts.z13_target_model import build_arm
            _,_,backbone=build_arm('C',torch,None);assert backbone.dev.type=='cuda' and not backbone.net.training
            weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
            for arm in p['gpu_arms']:
                engine.rollout(a.video,arm,obs,backbone)
                assert all(torch.equal(v,weights[k]) for k,v in backbone.net.named_parameters())
    else:
        engine=patch_engine()
        if a.role=='zero':engine.rollout(a.video,'ZERO',obs,zero=True);engine.rollout(a.video,'OFF_CHECK',obs,disabled=True)
        else:
            for arm in p['cpu_arms']:engine.rollout(a.video,arm,obs)
        assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)
    phase('full_phase.json');recovery_phase()


if __name__=='__main__':main(guard())
