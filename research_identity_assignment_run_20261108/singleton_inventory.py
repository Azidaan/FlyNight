"""Locate sole-candidate refusals on the calibration cohort; no tracker or model rerun."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_identity_assignment_run_20261108.common import *


def main():
    import numpy as np
    queries=[q for q in read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows'] if q['split']=='fit20']
    strata={r['uid']:r for r in read(ROOT/'research_identity_assignment_protocol_20261108/evaluation_strata.json')['rows']}
    outcomes={a:{r['uid']:r for r in read(RUN/'outcomes'/('fit20_'+a+'.json'))['rows']} for a in ['B0','G','G_FLYVIS']}
    byvideo=defaultdict(list)
    for q in queries:byvideo[q['video']].append(q)
    exports=read(RUN/'export_complete.json')['videos'];rows=[];counts=Counter();bygroup=defaultdict(Counter);sources={}
    for video,qq in byvideo.items():
        item=exports[video];path=ROOT/item['path'];assert sha(path)==item['sha256'];data=read(path);sources[item['path']]=item['sha256']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            for q in qq:
                snap=data['snapshots'][str(q['frame'])];i=next(i for i,r in enumerate(snap['rows']) if r['tid']==q['tid']);row=snap['rows'][i]
                s=strata[q['uid']];counts['all_queries']+=1
                if row['state']!=2:counts['not_Lost']+=1;continue
                choices=row.get('choices',[]);geometry=row.get('geometry',[])
                if len(choices)!=2 or len(geometry)!=2:counts['not_two_geometry_cases']+=1;continue
                valid=[j for j,g in zip(choices,geometry) if g['valid']]
                if len(valid)!=1:counts['not_exactly_one_valid']+=1;continue
                j=valid[0];raw=snap['raw_ids'][j];owner=next((snap['rows'][ii]['tid'] for ii,jj in snap['native_matches'] if jj==j),None)
                query_match=next((jj for ii,jj in snap['native_matches'] if ii==i),None)
                group='same_native_match' if query_match==j else ('free_first_stage' if owner is None else 'owned_by_other_track')
                C=z['cost_'+str(q['frame'])];cost=float(C[i,j]);candidate_role='unknown'
                if s['primary_known']:
                    candidate_role='target' if s['reason']=='ok' and s['target_side']==choices.index(j) else 'foreign'
                measured={a:outcomes[a][q['uid']]['outcome'] for a in outcomes}
                r=dict(**q,primary_known=s['primary_known'],reason=s['reason'],candidate_high_index=j,raw_id=raw,candidate_role=candidate_role,
                    group=group,native_owner=owner,native_query_match=query_match,native_cost=cost,
                    passes_native_cost=cost<=.8,passes_G_cost=cost-.2<=.8,outcomes=measured)
                rows.append(r);key=('primary' if s['primary_known'] else 'shadow')+'|'+group;c=bygroup[key]
                c['N']+=1;c['candidate_'+candidate_role]+=1;c['native_cost_admissible']+=int(cost<=.8)
                for arm,o in measured.items():c[arm+'_'+o['kind']]+=1;c[arm+'_J']+=o['J']
                if s['primary_known']:
                    c['G_deltaJ']+=measured['G']['J']-measured['B0']['J'];c['F_deltaJ']+=measured['G_FLYVIS']['J']-measured['B0']['J']
    counts['sole_valid_Lost']=len(rows)
    result=dict(passed=True,cohort='fit20 only; no evaluate26 result used to select a new rule.',counts=dict(counts),groups={k:dict(v) for k,v in bygroup.items()},rows=rows,
        sources=sources,scope='Existing geometry and native ownership metadata; no completed geometric calculation, map inference or tracker update repeated.',
        interpretation='Free means unassigned in the first native solve. Later unconfirmed association or birth can still use that high-confidence detection.')
    write(RUN/'singleton_inventory_fit20.json',result,new=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','sources']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
