"""Describe immutable first-A outcomes only after independent audit passes."""
import json,hashlib
from collections import Counter
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def write(n,x):
    with (OUT/n).open('xb') as f:f.write((json.dumps(x,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
p=read('protocol.json');r=read('result.json');audit=read('audit.json');assert r['passed'] and audit['passed'];assert audit['selection_sha256']==p['selection_sha256'];cases=[];transitions=Counter();byvideo={};missing=[]
for ci,c in enumerate(p['cases']):
    registration=next(z for z in r['registrations'] if z['case_index']==ci);rr=[z for z in r['records'] if z['case_index']==ci and z['head']=='native'];values={z['coordinate']+'_'+z['reference_mode']:z['score'] for z in rr};accept={k:None if v is None else v>0 for k,v in values.items()};original=accept['original_index'];joint=accept.get('census_terminal_hold');kind='missing_repair' if joint is None else 'positive_to_positive' if original and joint else 'positive_to_negative' if original else 'negative_to_positive' if joint else 'negative_to_negative';transitions[kind]+=1;video=byvideo.setdefault(c['video'],Counter());video[kind]+=1
    desc=dict(ready=registration['ready'])
    if registration['ready']:
        costs=registration['cost_tuples'];winner=costs[registration['winner']];firstties=sum(z[0]==winner[0] for z in costs);secondties=sum(z[:2]==winner[:2] for z in costs);dx,dy=registration['translation'];lo_x,lo_y,hi_x,hi_y=registration['integer_rectangle'];desc.update(translation=[dx,dy],source_pixels=len(registration['source_pixels']),candidates=len(costs),winning_median_hamming=winner[0],winning_mean_hamming=winner[1]/len(registration['source_pixels']),median_ties=firstties,median_sum_ties=secondties,selected_on_pixel_rectangle_boundary=dx in [lo_x,hi_x] or dy in [lo_y,hi_y])
    else:
        desc['reason']=registration['reason'];missing.append(dict(case_key=c['case_key'],reason=registration['reason']))
    if joint is None and registration['ready']:missing.append(dict(case_key=c['case_key'],reason='constant_or_unavailable_readout'))
    cases.append(dict(case_key=c['case_key'],event_key=c['event_key'],video=c['video'],frame=c['frame'],old_id=c['old_id'],raw_id=c['raw_id'],scores=values,accepted=accept,sign_transition=kind,registration=desc,causal_limits='Sign change is not identity correctness. Boundary or ambiguous Census objective is a descriptive symptom, not proven physical failure.'))
out=dict(passed=True,scope='Immutable first A; all cases retained, no parameter/selection changes. B not read.',selection_sha256=p['selection_sha256'],cases=len(cases),videos=len(byvideo),registration_ready=sum(c['registration']['ready'] for c in cases),sign_transitions=dict(transitions),by_video={k:dict(v) for k,v in byvideo.items()},missing=missing,details=cases,baseline_max_channel_error=max(z['channel_error'] for z in r['baseline_witness']),baseline_max_score_error=max(z['score_error'] for z in r['baseline_witness']),audit_max_errors=audit['max_errors'],result_sha256=hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest(),audit_sha256=hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest(),missing_policy='Preserve missing values; frozen parent composition may report real execution/readiness failure and keep baseline. Never fabricate0 orlarge negative score.',quality_claim='No standalone identity/HOTA/generalization success claim; complete appearance, ownership and sameappearance-withoutnewcore comparisons are separate mandatory artifacts.')
write('summary_A.json',out)
print(json.dumps({k:out[k] for k in ['cases','videos','registration_ready','sign_transitions','missing','baseline_max_channel_error','baseline_max_score_error','audit_max_errors']},ensure_ascii=False))
