"""Locate the exact old geometry rejection; no new image or model calculation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    import numpy as np
    run=ROOT/'runs/identity_evidence_20261108';inventory=ROOT/'runs/bbox_spatial_readiness_20261108/inventory.json';source=Path(__file__)
    write(run/'alignment_diagnostic_protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources={source.relative_to(ROOT).as_posix():sha(source),inventory.relative_to(ROOT).as_posix():sha(inventory)},scope='Decompose the existing alignment refusal on four known-positive fit edges. No activity, core, detector, metric or map-coverage causal test.'),new=True)
    rows=[];labels={}
    for e in read(inventory)['events']:
        if e['split']!='fit20':continue
        if e['video'] not in labels:labels[e['video']]=read(ROOT/'runs/bbox_singleton_pilot_20261108/labels'/(e['video']+'.json'))['rows']
        l=labels[e['video']][f"{e['frame']}|{e['old_tid']}|{e['new_tid']}"]
        if not l['known'] or not l['positive']:continue
        r=e['row_data'];mu=np.array(r['mean']);cov=np.array(r['covariance']);size=np.array(e['box'][2:])-e['box'][:2];old=np.array(r['history'][-1]['box']);axes=[]
        for axis in range(2):
            sd=float(np.sqrt(max(0,cov[axis+2,axis+2])));low=max(float(size[axis]),float(mu[axis+2]-2*sd));high=float(mu[axis+2]+2*sd)
            axes.append(dict(axis='x' if axis==0 else 'y',current_detection_extent=float(size[axis]),reference_observed_extent=float(old[axis+2]-old[axis]),predicted_extent=float(mu[axis+2]),predicted_extent_sd=sd,full_extent_lower=low,full_extent_upper=high,empty_size_interval=low>high))
        rows.append(dict(video=e['video'],frame=e['frame'],old_tid=e['old_tid'],new_tid=e['new_tid'],original_transform=e['transform'],axes=axes,cells_available=min(e['transform']['current_nodes'],e['transform']['reference_nodes'])>=3))
    assert len(rows)==4 and all(r['cells_available'] and not r['original_transform']['valid'] for r in rows)
    write(run/'alignment_diagnostic.json',dict(passed=True,rows=rows,known_positive_fit_edges=4,with_empty_predicted_full_size_interval=sum(any(a['empty_size_interval'] for a in r['axes']) for r in rows),no_claim_of_tracking_benefit=True),new=True)
    print([(r['video'],r['frame'],[(a['axis'],a['empty_size_interval']) for a in r['axes']]) for r in rows])


if __name__=='__main__':main()
