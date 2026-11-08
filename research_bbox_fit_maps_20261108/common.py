from research_bbox_normalized_signals_20261108.common import *
PARENT_STUDY=RUN
RUN=ROOT/'runs/bbox_fit_maps_20261108'
READINESS=RUN
status=for_run(RUN)


def phase(name='protocol.json'):
    p=read(RUN/name)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
