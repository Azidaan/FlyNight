"""Join all six previous direct cases through raw observations, never ID numbers."""
from evaluation_common import *
start(900)
import gzip,collections
import numpy as np
p=read(OUT/'protocol.json');phase=read(OUT/'case_followup_protocol.json');verify(phase)
assert read(OUT/'id_fate.json')['passed'] and all(read(OUT/(a+'_complete.json'))['passed'] for a in p['arms'])
previous=read(OLD/'id_fate.json');cases=[r for r in previous['records'] if r['arm']=='C34' and r['scored'] and r['kind']!='unknown'];assert len(cases)==6
records=[];summary={}
for video in sorted({r['video'] for r in cases}):
    for arm in p['arms']:
        path=OUT/'rollouts'/arm/(video+'.events.json.gz');receipt=read(OUT/'rollouts'/arm/(video+'.receipt.json'));assert sha(path)==receipt['artifacts'][path.relative_to(ROOT).as_posix()]
        with gzip.open(path,'rt',encoding='utf8') as f:data=json.load(f)
        published=collections.defaultdict(list)
        for a in data['assignments']:
            if a['emitted'] and a['raw_id']>=0:published[(a['frame'],a['raw_id'])].append(a['tid'])
        events={e['frame']:e for e in data['events']};counts=summary.setdefault(arm,collections.Counter())
        for case in [r for r in cases if r['video']==video]:
            frame=case['frame'];rf=case['reference_frame'];rid=case['raw_id'];rr=case['reference_raw_id'];owners=published.get((rf,rr),[])
            row=dict(video=video,arm=arm,frame=frame,kind=case['kind'],raw_id=rid,reference_frame=rf,reference_raw_id=rr,
                previous_numeric_ID_not_compared=case['old_id'],previous_head_accept=case['head_accept'],previous_global_selected=case['global_selected'],previous_public_oldID=case['old_id_emitted_on_candidate'],
                new_reference_owners=owners,new_candidate_public_owners=published.get((frame,rid),[]),kind_source='Previous exact direct GT endpoints, not every intermediate observation')
            counts['cases']+=1
            if len(owners)!=1:row.update(mapped=False,unknown_reason='Reference raw detection is not a unique public observation in this arm');counts['unmapped']+=1
            else:
                tid=owners[0];row.update(mapped=True,new_anchor_ID=tid,new_public_anchor_ID_on_candidate=tid in row['new_candidate_public_owners'])
                counts['public_anchor_ID_'+case['kind']]+=int(row['new_public_anchor_ID_on_candidate'])
                event=events.get(frame);pool=[] if event is None else [r for r in event['rows'] if r['tid']==tid]
                proposals=[] if event is None else [q for q in event['proposals'] if q['old_id']==tid and q['raw_id']==rid]
                assert len(proposals)<=1
                if proposals:
                    q=proposals[0];same=q['reference_frame']==rf and len(pool)==1 and pool[0]['history'][-1].get('raw_id')==rr
                    row.update(proposal_present=True,exact_same_reference_endpoint=same,new_reason=q['reason'],new_head_accept=q['accepted'],new_logit=q.get('logit'),
                        new_reference_frame=q['reference_frame'],new_original_cost=q['original_cost'],new_corrected_cost=event['corrected'][q['row']][q['column']],
                        new_global_selected=[q['row'],q['column']] in event['actual_first'],
                        column_winners=[dict(tid=event['rows'][i]['tid'],original=event['C'][i][j],corrected=event['corrected'][i][j]) for i,j in event['actual_first'] if j==q['column']])
                    counts['same_reference_endpoint']+=int(same)
                else:row.update(proposal_present=False,missing_proposal_reason='No matching saved proposal; do not infer whether detector,domain,state or support caused its absence')
                row['duplicates']=[] if event is None else [d for d in event['duplicates'] if tid in [d['tracked_id'],d['lost_id']]]
                row['current_anchor_assignments']=[a for a in data['assignments'] if a['frame']==frame and a['tid']==tid]
            records.append(row)
verify(phase)
write('previous_cases_followup.json',dict(passed=True,summary=summary,records=records,all_six_cases_all_five_arms=True,
    protocol_sha256=sha(OUT/'case_followup_protocol.json'),
    limits='Map IDs only through a unique emitted reference raw observation. The old six cases are diagnostic development examples. Different current histories need not yield the same proposal or ID number. Missing saved proposals have no invented cause. Followup does not select thresholds or replace aggregate and full-ID-fate results.',goal_achieved=False))
print('PREVIOUS_CASES_JOINED',len(records),flush=True)
