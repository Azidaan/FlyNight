from official import *
import gzip,lap
from scipy.optimize import linear_sum_assignment
p=read(OUT/'protocol.json');ss=read(ROOT/'research_identity_repairs_20261112/review/assignment_snapshots.json')['snapshots'];targets={q['old_id']:q for q in read(ROOT/'research_weekly_causal_audit_20261112/native_five_cases.json')['rows']};rows=[];guard=.8
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def solve(c):
 a=np.ascontiguousarray(c,dtype='float64');n,m=a.shape;_,ii,_=lap.lapjv(a,extend_cost=True,cost_limit=guard);ps=[(i,int(j)) for i,j in enumerate(ii) if j>=0];cost=sum(a[i,j] for i,j in ps)+(n+m-2*len(ps))*.4
 C=np.zeros((n+m,n+m));C[:n,:m]=a;C[:n,m:]=C[n:,:m]=.4;x,y=linear_sum_assignment(C);assert abs(float(C[x,y].sum()-cost))<1e-8;return ps,float(cost)
for q in ss:
 case=q['old_case'];v=q['video'];fr=q['frame'];orig=targets[case];base=np.array(q['base_cost']);corr=np.array(q['corrected_cost']);ti=q['target_row'];cj=q['target_column'];tids=[o['tid'] for o in q['rows']];rid=q['target_raw_id'];ref_id=orig['reference_identity'];prev=orig['reference_frame'];hist=json.load(gzip.open(ROOT/f'runs/domain_matched_native_20261109/rollouts/C34/{v}.events.json.gz'));raw=np.load(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz');em=hist['assignments'];ev=q['evaluation'];clock=read(ROOT/ev['annotation_clock'])['anchors'];gt={r['frameIndex']:r for r in read(ROOT/ev['annotation_boxes'])};b0,p0=solve(base);pa,objective=solve(corr);assert b0==sorted(map(tuple,q['native_first'])) and pa==sorted(map(tuple,q['actual_first']));winning=next(i for i,j in pa if j==cj);owner=tids[winning];test={}
 for label,mt in [('corrected',corr),('without_all_votes',base)]:
  for intervention in ['none','remove_winner','remove_target','remove_target_adjustment','remove_winner_adjustment','force_target']:
   a=mt.copy();idx=list(range(len(a)));costshift=0
   if intervention=='remove_winner':idx.remove(winning)
   if intervention=='remove_target':idx.remove(ti)
   if intervention=='remove_target_adjustment':a[ti,cj]=base[ti,cj]
   if intervention=='remove_winner_adjustment':a[winning,cj]=base[winning,cj]
   if intervention=='force_target':
    if a[ti,cj]>=guard:test[label+'__'+intervention]=dict(eligible=False,cost=float(a[ti,cj]));continue
    ix=[i for i in idx if i!=ti];jx=[j for j in range(a.shape[1]) if j!=cj];pairs,value=solve(a[np.ix_(ix,jx)]);pp=[(ix[i],jx[j]) for i,j in pairs]+[(ti,cj)];val=value+a[ti,cj]
   else:
    pairs,val=solve(a[idx]);pp=[(idx[i],j) for i,j in pairs]
   test[label+'__'+intervention]=dict(eligible=True,winner=next((tids[i] for i,j in pp if j==cj),None),old_assigned=next((q['high_raw_ids'][j] for i,j in pp if i==ti),None),objective=float(val),changes_other_rows=sorted([(tids[i],q['high_raw_ids'][j]) for i,j in set(pp)-set(pa) if i!=ti]))
 def exact_hits(frame,box):
  if str(frame) not in clock:return None
  e=clock[str(frame)];r=gt[int(e['frameIndex'])];assert e['name']==r['name'];return sorted([(str(o['id']),float(iou(np.array(box),bbox(o)))) for o in r['labels'] if o['category']=='car' and not (o.get('attributes') or {}).get('crowd',False) and iou(np.array(box),bbox(o))>=.5],key=lambda a:(-a[1],a[0]))
 timeline=[];people=sorted({case,owner});observed={}
 for frame in range(max(0,fr-90),fr+31):
  ass=[a for a in em if a['frame']==frame and a['tid'] in people and a['emitted']]
  def obs(a):
   id0=a['raw_id'];assert int(raw['frame_idx'][id0])==frame;return dict(tid=a['tid'],raw=id0,start=a['start_frame'],hits=exact_hits(frame,raw['xyxy'][id0]))
  items=[obs(a) for a in ass];timeline.append(dict(frame=frame,exact=str(frame) in clock,owners=items))
 for tid in people:
  rr=[(r['frame'],a) for r in timeline if r['frame']<fr for a in r['owners'] if a['tid']==tid];xx=[dict(frame=f,identities=[h[0] for h in a['hits']]) for f,a in rr if a['hits'] is not None];own=[e['frame'] for e in xx if e['identities']==[ref_id]]
  observed[str(tid)]=dict(first_emitted_in_window=rr[0][0] if rr else None,last_emitted_before_query=rr[-1][0] if rr else None,all_exact_past=xx,unique_target_frames=own,other_unique_ids=sorted({e['identities'][0] for e in xx if len(e['identities'])==1 and e['identities'][0]!=ref_id}),ambiguous_or_unmatched=sum(len(e['identities'])!=1 for e in xx))
 for idx,key in [(ti,'old'),(winning,'winner')]:
  row=q['rows'][idx];b=np.array(row['predicted_xyxy']);obs=np.array(raw['xyxy'][rid]);expected=1-iou(b,obs)*float(raw['conf'][rid]);assert abs(expected-base[idx,cj])<1e-5,(case,key,expected,base[idx,cj]);observed[key+'_geometry']=dict(predicted_box=b.tolist(),raw_box=obs.tolist(),shape_iou=iou(b,obs),confidence=float(raw['conf'][rid]),computed_native_cost=expected,expected_cost_without_confidence=1-iou(b,obs),reconstruction_error=float(expected-base[idx,cj]),center_delta=((b[:2]+b[2:]-obs[:2]-obs[2:])/2).tolist())
 rows.append(dict(case=case,video=v,frame=fr,expected=ref_id,target_raw=rid,candidate_label=orig['kind'],literal_owner=owner,previous_stop=prev,target_base=float(base[ti,cj]),target_corrected=float(corr[ti,cj]),winner_base=float(base[winning,cj]),winner_corrected=float(corr[winning,cj]),original_selection=pa,objective=objective,counterfactual=test,provenance=observed,timeline=timeline))
save('evidence.json',dict(rows=rows,no_weight_or_threshold_changes=True,source_hashes_checked=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
