"""Official evaluation as its own guarded CPU worker, no nested computation."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    verify_phase('full_execution_recovery2_phase.json')
    sys.argv.extend(['--parent-report',obs.read()['supervisor_report']])
    from research_identity_assignment_run_20261108.full_evaluate import main as evaluate
    evaluate();verify_phase('full_execution_recovery2_phase.json')


if __name__=='__main__':main(guard())
