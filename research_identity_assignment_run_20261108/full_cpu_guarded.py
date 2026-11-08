"""A single externally guarded CPU stage, with no subprocesses or neural core."""
import os,sys,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    p=argparse.ArgumentParser();p.add_argument('video');a=p.parse_args()
    import torch
    torch.set_num_threads(2);verify_phase('full_execution_recovery2_phase.json')
    from research_identity_assignment_run_20261108.full_retina_recovery1 import prepare_video
    from research_identity_assignment_run_20261108.full_run import rollout
    prepare_video(a.video,obs)
    for arm in ['G','G_R1']:rollout(a.video,arm,obs)
    assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)
    verify_phase('full_execution_recovery2_phase.json')


if __name__=='__main__':main(guard())
