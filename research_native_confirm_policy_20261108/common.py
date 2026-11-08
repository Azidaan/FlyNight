from research_identity_assignment_run_20261108.common import *
from research_singleton_verify_20261108.common import guard,sequence_key
from research_singleton_verify_full_recovery1_20261108.status_io import for_run
RUN=ROOT/'runs/native_confirm_policy_20261108'
TRACE=ROOT/'runs/singleton_native_confirm_20261108'
OWN=ROOT/'runs/singleton_verify_20261108'
BASE=ROOT/'runs/identity_assignment_20261108'
status=for_run(RUN)


def phase(name):
    p=read(RUN/name)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
