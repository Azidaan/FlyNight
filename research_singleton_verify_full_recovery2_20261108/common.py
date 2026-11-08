from research_singleton_verify_full_recovery1_20261108.common import *
RECOVERY1=RECOVERY
RECOVERY=RUN/'full_recovery2'


def second_phase():
    p=read(RECOVERY/'phase.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
