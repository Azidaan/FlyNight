from research_bbox_cost_pilot_20261108.common import *
COST=RUN
RUN=ROOT/'runs/bbox_normalized_signals_20261108'
READINESS=RUN
status=for_run(RUN)


def phase(name='protocol.json'):
    p=read(RUN/name)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
