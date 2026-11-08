from research_bbox_normalized_signals_20261108.common import *
FULL=RUN/'full'
PARENT=ROOT/'runs/native_confirm_risk_20261108/full'
RECOVERY=FULL/'view_restore'
status=for_run(FULL)


def full_phase():
    p=read(RUN/'full_phase.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p


def recovery_phase():return full_phase()
