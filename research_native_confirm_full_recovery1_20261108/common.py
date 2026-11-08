from research_native_confirm_full_20261108.common import *
RECOVERY=RUN/'full_recovery1'


def recovery_phase():
    p=read(RECOVERY/'phase.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
