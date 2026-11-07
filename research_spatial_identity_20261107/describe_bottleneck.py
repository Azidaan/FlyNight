"""Descriptive analysis only: no threshold, feature or tracker intervention."""
import sys
import csv
import gzip
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_spatial_identity_20261107.run_common import RUN,read,sha,write


def main():
    rows=read(ROOT/'research_spatial_identity_20261107/readiness_rows.json')
    with gzip.open(ROOT/'runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz','rt') as f:
        labels={int(r['uid']):r for r in csv.DictReader(f) if r['exactly2']=='1' and r['label_reason']=='ok'}
    report={};details=[]
    for split in ['fit84','fit20','evaluate26']:
        counts=Counter();geo=Counter();byvideo={}
        for r in rows:
            if r['split']!=split:continue
            if r['video'] not in byvideo:byvideo[r['video']]=read(RUN/'tracker'/(r['video']+'.json'))
            v=byvideo[r['video']];t=next(t for t in v['rows'] if t['uid']==r['uid'])
            selected=t['native_assignment_raw_id'];candidates=t['candidate_raw_ids']
            category='none' if selected is None else 'first' if selected==candidates[0] else 'second' if selected==candidates[1] else 'other'
            counts[category]+=1;snapshot=v['snapshots'][str(t['frame'])];row=snapshot['track_ids'].index(t['tid'])
            with np.load(RUN/'tracker'/(r['video']+'.npz'),allow_pickle=False) as z:matrix=z['cost_'+str(t['frame'])]
            eligible=[];occupied=[]
            for raw,owner in zip(candidates,t['candidate_assigned_track_ids']):
                col=snapshot['detection_raw_ids'].index(raw);eligible.append(bool(matrix[row,col]<=snapshot['threshold']))
                occupied.append(owner is not None and owner!=t['tid'])
            counts['at_least_one_eligible']+=int(any(eligible));counts['at_least_one_eligible_occupied']+=int(any(a and b for a,b in zip(eligible,occupied)))
            counts['none_with_eligible']+=int(category=='none' and any(eligible))
            counts['none_with_eligible_occupied']+=int(category=='none' and any(a and b for a,b in zip(eligible,occupied)))
            if split=='fit84':continue
            s=read(RUN/'scores'/f"{r['uid']}.json");g=s['geometry'];truth_index=0 if int(labels[r['uid']]['target_side'])==1 else 1
            flags=[]
            for x in g:
                reason='size_grid_empty' if x['width_values']==0 or x['height_values']==0 else 'under_three_nodes' if min(x['current_nodes'],x['reference_nodes'])<3 else 'position_grid_empty' if x.get('position_transforms',0)==0 else 'available'
                flags.append(reason)
            rejection=any(x!='available' for x in flags)
            geo['rejected_pairs' if rejection else 'geometry_available_pairs']+=1
            for why in ['size_grid_empty','position_grid_empty','under_three_nodes']:geo['pairs_with_'+why]+=int(why in flags)
            geo['true_available_false_unavailable']+=int(flags[truth_index]=='available' and flags[1-truth_index]!='available')
            geo['true_unavailable_false_available']+=int(flags[truth_index]!='available' and flags[1-truth_index]=='available')
            geo['both_unavailable']+=int(all(x!='available' for x in flags))
            certificate=s.get('certificates',{}).get('flyvis_spatial')
            if certificate:geo['flyvis_certificate_'+certificate]+=1
            b0=np.array(t['history'][-1]['box']);crop0=np.array(t['history'][-1]['crop']);ct=np.array(t['current_crop'])
            points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'])
            hull=ConvexHull(points);equations=hull.equations
            def clipped(box,crop):
                x1,y1,x2,y2=box;corners=np.array([[x1,y1],[x1,y2],[x2,y1],[x2,y2]])
                eye=(corners-crop[:2])*391/(crop[2:]-crop[:2])
                return bool((eye@equations[:,:2].T+equations[:,2]>1e-8).any())
            refclip=clipped(b0,crop0);currentclip=[clipped(np.array(b),ct) for b in t['candidate_boxes']]
            details.append(dict(uid=r['uid'],split=split,video=r['video'],native=category,eligible=eligible,
                occupied=occupied,geometry_reasons=flags,true_candidate=truth_index,
                flyvis_certificate=certificate,reference_field_clipped=refclip,current_field_clipped=currentclip,
                observed_gap=t['frame']-t['history'][-1]['frame'],reference_nodes=g[0]['reference_nodes'],current_nodes=[x['current_nodes'] for x in g]))
        report[split]=dict(pairs=sum(1 for r in rows if r['split']==split),native=dict(counts),geometry=dict(geo))
    write(RUN/'bottleneck_rows.json',details,new=True)
    ev=[r for r in details if r['split']=='evaluate26']
    report['evaluate26']['coverage']=dict(reference_clipped=sum(r['reference_field_clipped'] for r in ev),
        any_candidate_clipped=sum(any(r['current_field_clipped']) for r in ev),
        reference_nodes_median=float(np.median([r['reference_nodes'] for r in ev])),
        current_nodes_median=float(np.median([x for r in ev for x in r['current_nodes']])),
        observed_gap_median=float(np.median([r['observed_gap'] for r in ev])))
    report['scope']='Post-hoc descriptive audit of frozen outcomes. No new method selection or threshold sweep. Native occupied counts concern the saved first high-confidence association stage.'
    write(RUN/'bottleneck_report.json',report,new=True)
    print(report)


if __name__=='__main__':main()
