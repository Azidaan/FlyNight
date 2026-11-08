from research_bbox_cost_pilot_20261108.common import *
COST=RUN
RUN=ROOT/'runs/bbox_spatial_readiness_20261108'
status=for_run(RUN)


def phase():
    p=read(RUN/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
