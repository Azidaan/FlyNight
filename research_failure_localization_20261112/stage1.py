from common import *
import numpy as np
from collections import Counter
p=verify();native=read(NATIVE/'protocol.json');records=[];trace=[]
for video in sorted({r['video'] for r in p['selected']}):
    d=events(video);evs={e['frame']:e for e in d['events']};ann=annotations(video)
    with np.load(ROOT/native['videos'][video]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    def rawhits(rid,frame):
        assert int(raw['frame_idx'][rid])==frame
        return hits(raw['xyxy'][rid],ann[frame]) if frame in ann else None
    def owner(tid,frame,target):
        obs=[a for a in d['assignments'] if a['tid']==tid and a['frame']<=frame and a['raw_id']>=0]
        labelled=[]
        for a in obs:
            h=rawhits(a['raw_id'],a['frame'])
            if h is not None:labelled.append(dict(frame=a['frame'],raw_id=a['raw_id'],emitted=a['emitted'],hits=h,kind='unknown' if len(h)!=1 else ('own' if h[0]['identity']==target else 'foreign')))
        return labelled[-3:]
    for r in [x for x in p['selected'] if x['video']==video]:
        e=evs[r['frame']];q=next(q for q in e['proposals'] if q['old_id']==r['old_id'] and q['raw_id']==r['raw_id'])
        a=rawhits(r['reference_raw_id'],r['reference_frame']);b=rawhits(r['raw_id'],r['frame'])
        assert len(a)==len(b)==1
        kind='own' if a[0]['identity']==b[0]['identity'] else 'foreign';assert kind==r['kind']
        row=e['rows'][q['row']];native_unmatched=all(i!=q['row'] for i,j in e['native_first'])
        gate=('not_car' if row['cls']!=2 else 'native_first_already_matched' if not native_unmatched else 'insufficient_reference_history' if len({h['frame'] for h in row['history']})<2 else 'outside_observed_training_gap' if r['frame']-row['history'][-1]['frame'] not in (5,6) else None)
        if not r['scored']:assert gate==r['reason']
        else:assert gate is None
        target=a[0]['identity']
        competitors=[dict(**w,owner_at_or_before=owner(w['tid'],r['frame'],target)) for w in r['column_global_winners']]
        failure=('gap_exclusion_of_own' if kind=='own' and gate else 'correct_rejection_of_foreign_by_gap' if gate else 'head_false_negative' if kind=='own' and not r['head_accept'] else 'head_false_positive_blocked_by_assignment' if kind=='foreign' and r['head_accept'] and not r['global_selected'] else 'head_true_positive_blocked_by_assignment' if kind=='own' and not r['global_selected'] else 'successful_own_assignment' if kind=='own' else 'foreign_assignment')
        records.append(dict(video=video,frame=r['frame'],old_id=r['old_id'],raw_id=r['raw_id'],kind=kind,reference_hits=a,current_hits=b,
          first_recorded_mechanism=failure,gate=gate,head_score=r['logit'],head_accept=r['head_accept'],selected=r['global_selected'],emitted=r['old_id_emitted_on_candidate'],
          row=q['row'],column=q['column'],shape=[len(e['C']),len(e['C'][0])],original_cost=e['C'][q['row']][q['column']],corrected_cost=e['corrected'][q['row']][q['column']],competitors=competitors,
          history=row['history'],reference_frame=r['reference_frame'],future_observations=r['future_public_old_ID_at_annotated_frames'],
          full_root_cause_closed=failure in ['correct_rejection_of_foreign_by_gap','successful_own_assignment'],
          limitation='First recorded blocking mechanism, not yet proof that lifting it repairs the complete trajectory.'))
    if video=='b1f4491b-cf446195':
        target=next(r for r in records if r['old_id']==1102)['reference_hits'][0]['identity']
        for a in d['assignments']:
            if a['tid']!=1102 or not 251<=a['frame']<=264:continue
            h=rawhits(a['raw_id'],a['frame']) if a['raw_id']>=0 else None
            e=evs.get(a['frame']);trace.append(dict(**a,annotated_frame=a['frame'] in ann,hits=h,
              kind='unlabelled' if h is None else 'ambiguous' if len(h)!=1 else 'own' if h[0]['identity']==target else 'foreign',
              matrix_saved=e is not None,box=None if a['raw_id']<0 else raw['xyxy'][a['raw_id']].tolist()))
    print('STAGE1',video,len(records),flush=True)
verify();save('stage1.json',dict(passed=True,records=records,counts=dict(Counter(r['first_recorded_mechanism'] for r in records)),trace1102=trace,
 script_sha256=sha(__file__),source_protocol_sha256=sha(OUT/'protocol.json'),limitations='Frame261 not annotated if marked false. No identity interpolated from endpoints.'))
journal('Первый этап локализации: независимо сверены 29 прямых меток и первый записанный механизм отказа; цепочка 1102 вынесена отдельно. Источник research_failure_localization_20261112/stage1.json. Следующий этап — подстановка разрешения на сохранённых матрицах, без изменения порогов.')
print('COUNTS',dict(Counter(r['first_recorded_mechanism'] for r in records)));print('TRACE',trace)
