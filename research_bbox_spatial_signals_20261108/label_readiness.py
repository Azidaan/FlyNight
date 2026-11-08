"""Stop restoration if the available maps cannot support the registered fit gate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_bbox_spatial_signals_20261108.common import *


def main():
    assert read(READINESS/'queue_result.json')['passed'];names=['runs/bbox_spatial_readiness_20261108/inventory.json','research_bbox_spatial_signals_20261108/label_readiness.py'];write(RUN/'label_readiness_protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in names},criterion='Before any restored neural maps, at least one actually known positive fit20edge must have spatial support. If zero, no support-only or fixed score veto can create a positive fit link, so the proposed comparison cannot pass its fit>0gate. This is a necessary logical readiness condition, not a quality claim.'),new=True)
    inventory=read(READINESS/'inventory.json');cache={};counts=defaultdict(Counter)
    for e in inventory['events']:
        if e['video'] not in cache:cache[e['video']]=read(PRIOR_PILOT/'labels'/(e['video']+'.json'))['rows']
        label=cache[e['video']][f"{e['frame']}|{e['old_tid']}|{e['new_tid']}"];kind='correct' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown';counts[e['split']]['all_'+kind]+=1
        if not e['unavailable']:counts[e['split']]['supported_'+kind]+=1
    proceed=counts['fit20']['supported_correct']>0;result=dict(passed=True,counts={k:dict(v) for k,v in counts.items()},restoration_logically_testable=bool(proceed),new_neural_calls=0,scope='Actual labels reused only to check whether any fit-positive spatial comparison exists. These descriptive labels do not become runtime inputs. Neither a chosen source nor new thresholds are selected from26.');write(RUN/'label_readiness.json',result,new=True);print(result)


if __name__=='__main__':main()
