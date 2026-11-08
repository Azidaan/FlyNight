from research_bbox_singleton_pilot_20261108.common import *
PRIOR_PILOT=RUN
RUN=ROOT/'runs/bbox_cost_pilot_20261108'
status=for_run(RUN)


def phase():
    p=read(RUN/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
