from research_native_confirm_policy_20261108.common import *
PRIOR=RUN
RUN=ROOT/'runs/native_confirm_risk_20261108'
status=for_run(RUN)


def phase(name):
    p=read(RUN/name)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
