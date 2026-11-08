"""Guarded fixed statistical analysis of the corrected-input pilot."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    verify_phase('full_post_registration.json')
    import research_identity_assignment_run_20261108.full_statistics as module
    module.FULL=RUN/'full_precision';module.main(obs)
    verify_phase('full_post_registration.json')


if __name__=='__main__':main(guard())
