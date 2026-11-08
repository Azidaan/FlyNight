from research_bbox_spatial_readiness_20261108.common import *
READINESS=RUN
RUN=ROOT/'runs/bbox_spatial_signals_20261108'
status=for_run(RUN)


def phase(name='protocol.json'):
    p=read(RUN/name)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
