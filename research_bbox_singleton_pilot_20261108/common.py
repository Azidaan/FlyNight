from research_bbox_confirmation_inventory_20261108.common import *
INVENTORY=RUN
RUN=ROOT/'runs/bbox_singleton_pilot_20261108'
status=for_run(RUN)


def phase():
    p=read(RUN/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
