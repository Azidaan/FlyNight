import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_20261108.common import *


def main(obs):
    phase('full_phase.json');configure_paths()
    import research_identity_assignment_run_20261108.full_evaluate as evaluate
    evaluate.FULL=FULL;evaluate.status=status;sys.argv.extend(['--parent-report',obs.read()['supervisor_report']]);evaluate.main();phase('full_phase.json')


if __name__=='__main__':main(guard())
