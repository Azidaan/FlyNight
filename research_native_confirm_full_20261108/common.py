from research_native_confirm_risk_20261108.common import *
FULL=RUN/'full'
PARENT=ROOT/'runs/singleton_verify_20261108/full'
status=for_run(FULL)


def full_phase():
    p=read(RUN/'full_phase.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
