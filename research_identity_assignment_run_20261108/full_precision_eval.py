"""Unchanged official evaluation, directed to corrected-input outputs."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    verify_phase('full_precision_execution_phase.json')
    from research_identity_assignment_run_20261108.full_precision_io import FULL
    import research_identity_assignment_run_20261108.full_evaluate as module
    module.FULL=FULL;sys.argv.extend(['--parent-report',obs.read()['supervisor_report']]);module.main()
    verify_phase('full_precision_execution_phase.json')


if __name__=='__main__':main(guard())
