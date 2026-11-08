"""Correct only official CLEAR field spelling; preserve all numerical checks."""
import os,sys,importlib
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_recovery2_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import for_run


def main(obs):
    second_phase();group=sys.argv[1];name=sys.argv[2]
    if group=='own_audit':
        del sys.argv[1:3]
        import research_singleton_verify_full_20261108.common as c
        c.status=for_run(c.RUN);module=importlib.import_module('research_singleton_verify_full_recovery2_20261108.own_audit')
        module.main(obs)
    elif group=='age' and name=='full_audit':
        del sys.argv[1:3]
        import research_singleton_age_20261108.common as c
        c.status=for_run(c.RUN);module=importlib.import_module('research_singleton_verify_full_recovery2_20261108.age_audit')
        module.main(obs)
    else:
        from research_singleton_verify_full_recovery1_20261108.stage import main as preserved
        preserved(obs)
    second_phase()


if __name__=='__main__':main(guard())
