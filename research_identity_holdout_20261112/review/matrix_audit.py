"""Independent bounded assignment/evaluation on ALL54 frozen A events.

Decisions: {records:[{event_key,variant,edge_updates:[{tid,raw_id,cost}],
 append_columns:[{raw_id,costs:[nrows]}],freeze:'all'|'none',defer_tids:[],
 alias_groups:[[rawids]],reported_pairs:[{tid,raw_id}]}]}.
No policy selection or model/tracker instantiation occurs here.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,math,argparse
from pathlib import Path
import numpy as np,lap
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def solve(c):
 if not c.size:return []
 _,x,_=lap.lapjv(c,extend_cost=True,cost_limit=.8);return sorted((i,int(j)) for i,j in enumerate(x) if j>=0)
def objective(c,pairs):return math.fsum(float(c[i,j]) for i,j in pairs)+.4*(sum(c.shape)-2*len(pairs))
def independent(c,pairs):
 n,k=c.shape;a=np.zeros((n+k,n+k));a[:n,:k]=c;a[:n,k:]=.4;a[n:,:k]=.4;i,j=linear_sum_assignment(a);value=math.fsum(float(a[ii,jj]) for ii,jj in zip(i,j));error=abs(value-objective(c,pairs));assert error<1e-9;return error
def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));den=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter;return inter/den if den>0 else 0.
def annotation_objects(event):
 clock=read(ROOT/event['evaluation_sources']['clock'])['anchors'];key=str(event['frame'])
 if key not in clock:return None
 anchor=clock[key];labels=read(ROOT/event['evaluation_sources']['labels']);by={int(r.get('frameIndex',i)):r for i,r in enumerate(labels)};row=by[anchor['frameIndex']];assert row['name']==anchor['name']
 return [dict(identity=str(o['id']),box=[o['box2d'][k] for k in ['x1','y1','x2','y2']]) for o in row['labels'] if o['category']=='car' and not o.get('attributes',{}).get('crowd',False)]
def audit_event(event,decision,evaluation_rows):
 rowids=[r['tid'] for r in event['rows']];columns=list(event['high_raw_ids']);n=len(rowids);base=np.asarray(event['corrected_cost'],np.float32);before=solve(base);assert before==sorted(map(tuple,event['actual_first']));error=independent(base,before);c=base.copy()
 for addition in decision.get('append_columns',[]):
  assert addition['raw_id'] not in columns;z=np.asarray(addition['costs'],np.float32);assert z.shape==(n,) and np.isfinite(z).all();columns.append(addition['raw_id']);c=np.c_[c,z]
 for update in decision.get('edge_updates',[]):c[rowids.index(update['tid']),columns.index(update['raw_id'])]=float(update['cost'])
 assert np.isfinite(c).all();mode=decision.get('freeze','all');assert mode in ['all','none'];locked=before if mode=='all' else []
 for i,j in locked:assert c[i,j]==base[i,j]
 for tid in decision.get('defer_tids',[]):
  i=rowids.index(tid);assert i not in {ii for ii,jj in locked};c[i,:]=1e6
 rr=[i for i in range(n) if i not in {ii for ii,jj in locked}];cc=[j for j in range(c.shape[1]) if j not in {jj for ii,jj in locked}];residual=c[np.ix_(rr,cc)];partial=solve(residual);error=max(error,independent(residual,partial));after=sorted(locked+[(rr[i],cc[j]) for i,j in partial]);pairs={(rowids[i],columns[j]) for i,j in after};original={(rowids[i],event['high_raw_ids'][j]) for i,j in before};added=pairs-original;removed=original-pairs
 if 'reported_pairs' in decision:assert pairs=={(p['tid'],p['raw_id']) for p in decision['reported_pairs']}
 assigned={(int(t),int(r)) for t,r in event['assigned'].items()};ownership=[dict(tid=t,raw_id=r,saved_owner=owner) for t,r in added for owner,raw in assigned if raw==r and owner!=t];alias=[]
 for group in decision.get('alias_groups',[]):
  alias.extend(dict(tid=t,raw_id=r,saved_owner=owner,saved_raw=raw) for t,r in added if r in group for owner,raw in assigned if raw in group and owner!=t)
 protected_removed=[dict(tid=t,raw_id=r) for t,r in removed if (t,r) in assigned or event['rows'][rowids.index(t)]['state']==1]
 eligible={p['old_id'] for p in event['native_proposals'] if p.get('geometry') is not None};new_low_violations=[dict(tid=t,raw_id=r) for t,r in added if r not in event['high_raw_ids'] and (event['rows'][rowids.index(t)]['state']!=2 or t not in eligible)]
 current=annotation_objects(event);raw={r['raw_id']:r for r in event['raw_candidates']};eval_added=[]
 def target_hits(rid):return [] if current is None or rid not in raw else [o['identity'] for o in current if iou(raw[rid]['box'],o['box'])>=.5]
 for tid,rid in sorted(added):
  row=evaluation_rows[tid];hit=target_hits(rid);primary=row['last_reference_unique_identity'] if row['strict_last_reference_evaluable'] else None;anchored=row['expected_anchored_persistent_identity'] if row['anchored_history_evaluable'] else None
  def classify(expected):return 'unknown' if expected is None or len(hit)!=1 else 'correct' if expected==hit[0] else 'wrong'
  eval_added.append(dict(tid=tid,raw_id=rid,current_hits=hit,strict_last_reference=classify(primary),anchored_history_continuity=classify(anchored),last_reference_expected=primary,anchored_expected=anchored,source_anchor_frames=[h['frame'] for h in row['annotated_history_observations']],scope='AnchoredhistoryexpectedID isnotGTforunannotatedlastcrop.'))
 # A combined ledger preserves downstream native assignments that are outside
 # the first matrix, while replacing only that matrix's original pairs.
 before_all=assigned|original;after_all=(assigned-original)|pairs
 def duplicate_claims(ledger):
  claims={}
  for tid,rid in ledger:
   h=target_hits(rid)
   if len(h)==1:claims.setdefault(h[0],set()).add(tid)
  return {identity:sorted(ids) for identity,ids in claims.items() if len(ids)>1}
 olddup=duplicate_claims(before_all);newdup=duplicate_claims(after_all);extra_duplicates={identity:tids for identity,tids in newdup.items() if olddup.get(identity)!=tids}
 opportunities=[]
 for proposal in event['eligible_proposal_keys']:
  tid=proposal['old_id'];er=evaluation_rows[tid]
  for unit,expected in [('strict_last_reference',er['last_reference_unique_identity'] if er['strict_last_reference_evaluable'] else None),('anchored_history',er['expected_anchored_persistent_identity'] if er['anchored_history_evaluable'] else None)]:
   available=bool(expected is not None and current is not None and any(o['identity']==expected for o in current))
   def covered(ledger):return bool(available and any(rid in raw and target_hits(rid)==[expected] for _,rid in ledger))
   opportunities.append(dict(tid=tid,unit=unit,evaluable=expected is not None,annotated_identity_present=available,baseline_covered=covered(before_all) if available else None,repaired_covered=covered(after_all) if available else None))
 return dict(event_key=event['event_key'],video=event['video'],frame=event['frame'],variant=decision['variant'],matrix_shape=list(c.shape),baseline_reproduced=True,independent_objective_error=error,added=[dict(tid=t,raw_id=r) for t,r in sorted(added)],removed=[dict(tid=t,raw_id=r) for t,r in sorted(removed)],protected_assignments_removed=protected_removed,ownership_conflicts=ownership,alias_ownership_conflicts=alias,native_domain_or_state_violations=new_low_violations,new_duplicate_annotation_claims=extra_duplicates,added_identity_evaluation=eval_added,opportunities=opportunities,actual_pairs=[dict(tid=t,raw_id=r) for t,r in sorted(pairs)],new_full_tracker_calls=0)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--go',required=True);parser.add_argument('--decisions',required=True);parser.add_argument('--output',required=True);args,private=parser.parse_known_args();assert all(x.startswith('--audit-memory-worker=') for x in private);go=read(args.go);assert go['approved'] is True
 for n,h in go['inputs'].items():assert sha(ROOT/n)==h,n
 assert Path(args.decisions).resolve().relative_to(ROOT).as_posix() in go['inputs'] and Path(__file__).relative_to(ROOT).as_posix() in go['inputs']
 runtime=read(OUT/'A_runtime_manifest_v2.json');evaluation=read(OUT/'anchored_readiness.json');assert evaluation['runtime_manifest_sha256']==sha(OUT/'A_runtime_manifest_v2.json');decisions=read(args.decisions)['records'];variants=sorted({d['variant'] for d in decisions});eventkeys={e['event_key'] for e in runtime['events']};out=[]
 for n,h in evaluation['source_sha256'].items():assert sha(ROOT/n)==h,n
 for variant in variants:
  subset=[d for d in decisions if d['variant']==variant];assert len(subset)==len(eventkeys) and {d['event_key'] for d in subset}==eventkeys
  for d in subset:
   event=next(e for e in runtime['events'] if e['event_key']==d['event_key']);rows={r['tid']:r for r in evaluation['rows'] if r['event_key']==event['event_key']};out.append(audit_event(event,d,rows))
 summaries=[]
 for variant in variants:
  rr=[r for r in out if r['variant']==variant];added=[a for r in rr for a in r['added_identity_evaluation']];structure=sum(len(r[k]) for r in rr for k in ['protected_assignments_removed','ownership_conflicts','alias_ownership_conflicts','native_domain_or_state_violations','new_duplicate_annotation_claims'])
  identities={unit:{status:sum(a[unit]==status for a in added) for status in ['correct','wrong','unknown']} for unit in ['strict_last_reference','anchored_history_continuity']};op={}
  for unit in ['strict_last_reference','anchored_history']:
   oo=[o for r in rr for o in r['opportunities'] if o['unit']==unit];known=[o for o in oo if o['annotated_identity_present']];baseline_count=sum(o['baseline_covered'] for o in known);repaired_count=sum(o['repaired_covered'] for o in known)
   videos=sorted({r['video'] for r in rr});byvideo=[]
   for v in videos:
    vv=[o for r in rr if r['video']==v for o in r['opportunities'] if o['unit']==unit and o['annotated_identity_present']];byvideo.append([sum(int(o['repaired_covered'])-int(o['baseline_covered']) for o in vv),len(vv)])
   cluster=np.asarray(byvideo,float);rng=np.random.default_rng(20261112);draws=rng.integers(0,len(videos),size=(2000,len(videos)));sums=cluster[draws].sum(1);defined=sums[:,1]>0;vals=sums[defined,0]/sums[defined,1];ci=np.quantile(vals,[.025,.975]).tolist() if len(vals) else None;known_videos=sum(x[1]>0 for x in byvideo)
   op[unit]=dict(all_eligible_opportunities=len(oo),evaluable=sum(o['evaluable'] for o in oo),annotated_identity_present=len(known),baseline_covered=baseline_count,repaired_covered=repaired_count,paired_coverage_delta=(repaired_count-baseline_count)/len(known) if known else None,paired_video_bootstrap=dict(seed=20261112,draws=2000,videos=len(videos),videos_with_defined_opportunities=known_videos,undefined_draws=int((~defined).sum()),ci95=ci,limits='Conditionalonindependentlylabelledopportunities; missinglabelsnotzero. Clusterwholevideos; veryfewdefinedvideos limitsinference.'),positive_benefit_ci=bool(ci and ci[0]>0 and known_videos>=2))
  identity_gate=structure==0 and identities['strict_last_reference']['wrong']==0 and identities['strict_last_reference']['unknown']==0;benefit_gate=op['strict_last_reference']['positive_benefit_ci']
  summaries.append(dict(variant=variant,events=len(rr),new_assignments=len(added),identity=identities,structural_failures=structure,opportunities=op,deployment_identity_gate=identity_gate,primary_benefit_gate=benefit_gate,full_deployment_gate=bool(identity_gate and benefit_gate and len(added)>0),no_op_is_benefit_evidence=False,scope='PrimarystrictlastreferenceGT andseparateanchoredIDcontinuity; do notmergeunits orcallunknownwrong. Addedunknownassignmentscannotpassidentitydeploymentgate.'))
 for n,h in go['inputs'].items():assert sha(ROOT/n)==h,n
 for n,h in evaluation['source_sha256'].items():assert sha(ROOT/n)==h,n
 output=Path(args.output).resolve();output.relative_to(OUT.parent)
 with output.open('x',encoding='utf8') as f:json.dump(dict(passed=True,records=out,summaries=summaries,variants=variants,all_54_events_checked_each_variant=True,source_runtime_sha256=sha(OUT/'A_runtime_manifest_v2.json'),data_readiness=evaluation['summary'],scope='Exactsavedfirstmatrixwithdownstreamownershipledgercheck. No secondstage/lifecycle/futuretrajectory replay. Unknown labels remainunknown, cannotpassdeploymentnewactiongate.'),f,ensure_ascii=False,indent=2,allow_nan=False)
 print('HOLDOUT_MATRIX_AUDIT',len(out),len(variants))
if __name__=='__main__':main()
