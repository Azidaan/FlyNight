"""Run preserved CPU stage bodies under the same guard and robust service status."""
import os,sys,importlib
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_recovery1_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import for_run


def main(obs):
    # Only strip dispatch arguments after the guardian's child is running.
    recovery_phase();group=sys.argv.pop(1);name=sys.argv.pop(1)
    if group=='own':
        import research_singleton_verify_full_20261108.common as c
        c.status=for_run(c.RUN);module=importlib.import_module('research_singleton_verify_full_20261108.'+name)
    elif group=='own_audit':
        import research_singleton_verify_full_20261108.common as c
        c.status=for_run(c.RUN);module=importlib.import_module('research_singleton_verify_full_audit_20261108.audit')
    elif group=='age':
        import research_singleton_age_20261108.common as c
        c.status=for_run(c.RUN);module=importlib.import_module('research_singleton_age_20261108.'+name)
    elif group=='diagnosis':
        import research_singleton_error_audit_20261108.common as c
        c.status=for_run(c.RUN);module=importlib.import_module('research_singleton_error_audit_20261108.worker')
    else:raise ValueError(group)
    module.main(obs);recovery_phase()


if __name__=='__main__':main(guard())
