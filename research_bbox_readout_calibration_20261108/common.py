from research_bbox_fit_maps_20261108.common import *
DATA=RUN
RUN=ROOT/'runs/bbox_readout_calibration_20261108'
status=for_run(RUN)


def phase():
    p=read(RUN/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
