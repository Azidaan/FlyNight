"""Describe the frozen result's bottlenecks and the two requested native cases."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from statistics import median
from research_identity_assignment_run_20261108.common import *


def main(obs):
    assert read(RUN/'independent_audit.json')['passed'];verify_phase('diagnostic_phase.json')
    report=read(RUN/'comparison_report.json');exports=read(RUN/'export_complete.json')['videos']
    geom=defaultdict(Counter);columns=Counter();features=defaultdict(Counter);support=[];case_rows={};strata={r['uid']:r for r in read(ROOT/'research_identity_assignment_protocol_20261108/evaluation_strata.json')['rows']}
    for video,entry in exports.items():
        data=read(ROOT/entry['path']);plans=read(RUN/'scores'/(video+'.json'))['frames']
        labs={r['uid']:r for r in read(ROOT/f'runs/z13_temporal_pair_readiness_20261025/supervision/{video}.json')['rows']}
        for frame,snapshot in data['snapshots'].items():
            for q in snapshot['queries']:
                if q['uid'] not in strata:continue
                row=next(r for r in snapshot['rows'] if r['tid']==q['tid']);group=q['split']+'|'+strata[q['uid']]['reason']
                if row['state']!=2:geom[group]['native_state_'+str(row['state'])]+=1;continue
                flags=[g['valid'] for g in row['geometry']];geom[group]['eligible_'+str(sum(flags))]+=1
                l=labs[q['uid']]
                if strata[q['uid']]['reason']=='ok':
                    for valid,identity in zip(flags,l['current_identity']):
                        geom[group][('true' if identity==l['target_identity'] else 'foreign')+('_eligible' if valid else '_invalid')]+=1
                if q['uid'] in [29017,29057]:
                    i=next(i for i,r in enumerate(snapshot['rows']) if r['tid']==q['tid'])
                    candidate=[]
                    for j in q['candidate_high_ids']:
                        owner=next((r for r,jj in snapshot['native_matches'] if jj==j),None)
                        candidate.append(dict(high_id=j,raw_id=snapshot['raw_ids'][j],owner_tid=snapshot['rows'][owner]['tid'] if owner is not None else None))
                    case_rows[str(q['uid'])]=dict(query=q,query_start_frame=row['start_frame'],candidate_owners=candidate,arms={})
            for j,col in snapshot['columns'].items():
                columns['proposed_columns']+=1
                if len(col['rows'])<2:columns['no_second_rival']+=1
                elif not col['geometry_valid']:
                    if 'items' not in col:
                        reason='cross_class_rival' if any(snapshot['rows'][i]['cls']!=snapshot['classes'][int(j)] for i in col['rows']) else 'missing_history'
                        columns[reason]+=1
                    elif any(not x['mapping']['valid'] for x in col['items']):columns['rival_geometry_failure']+=1
                    else:columns['fewer_than3_reference_matches']+=1
                elif not plans[frame]['columns'][j]['geometry_valid']:columns['DIS_domain_support_failure']+=1
                else:
                    columns['common_geometry_valid']+=1
                    support.extend(plans[frame]['columns'][j]['support'])
                    for arm,scores in plans[frame]['columns'][j]['scores'].items():
                        features[arm]['columns']+=1
                        if any(x is None for x in scores):features[arm]['undefined']+=1
                        elif max(scores)-min(scores)<=1e-12:features[arm]['equal_rival_scores']+=1
                        else:features[arm]['different_rival_scores']+=1
    for arm in report['parameters']:
        rows=read(RUN/'outcomes'/f'evaluate26_{arm}.json')['rows']
        for row in rows:
            if str(row['uid']) not in case_rows:continue
            case_rows[str(row['uid'])]['arms'][arm]={k:row[k] for k in ['outcome','first_high_index','duplicate_events','geometric_candidates']}
    write(RUN/'mechanism_report.json',dict(passed=True,geometry={k:dict(v) for k,v in geom.items()},
        competition_columns=dict(columns),feature_coverage={k:dict(v) for k,v in features.items()},requested_cases=case_rows,
        common_unique_reference_nodes=dict(pairs=len(support),minimum=min(support),median=median(support),maximum=max(support)) if support else None,
        comparison_sha256=sha(RUN/'comparison_report.json'),
        interpretation_boundary='A queried lost ID may be a later fragment of the same physical car as its rival. Returning that queried ID is not automatically preferable long-term identity continuity.'),new=True)
    verify_phase('diagnostic_phase.json')


if __name__=='__main__':main(guard())
