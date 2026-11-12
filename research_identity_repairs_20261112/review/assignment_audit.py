"""Independent complete saved-matrix assignment audit; no tracker or model.

Input decisions schema:
{records:[{variant,video,frame,base:'base_cost'|'corrected_cost',
 edge_updates:[{tid,raw_id,cost}], append_columns:[{raw_id,costs:[nrows]}],
 freeze:'none'|'active'|'all', defer_tids:[], reported_pairs:[{tid,raw_id}],
 alias_groups:[[raw_id,...]]}]}.
An authorized GO JSON must bind inputs by SHA256 before this is executed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,argparse,math
from pathlib import Path
import numpy as np
import lap
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def solve(c):
 if not c.size:return []
 _,x,_=lap.lapjv(c,extend_cost=True,cost_limit=.8)
 return sorted((i,int(j)) for i,j in enumerate(x) if j>=0)
def objective(c,pairs):return math.fsum(float(c[i,j]) for i,j in pairs)+.4*(sum(c.shape)-2*len(pairs))
def independent(c,pairs):
 n,k=c.shape;a=np.zeros((n+k,n+k),float);a[:n,:k]=c;a[:n,k:]=.4;a[n:,:k]=.4
 i,j=linear_sum_assignment(a);value=math.fsum(float(a[ii,jj]) for ii,jj in zip(i,j));error=abs(value-objective(c,pairs));assert error<1e-6,(error,value,objective(c,pairs));return error
def constrained(c,locked):
 assert len({i for i,j in locked})==len(locked) and len({j for i,j in locked})==len(locked)
 rows=[i for i in range(c.shape[0]) if i not in {i for i,j in locked}];cols=[j for j in range(c.shape[1]) if j not in {j for i,j in locked}]
 residual=c[np.ix_(rows,cols)];partial=solve(residual);error=independent(residual,partial)
 return sorted(locked+[(rows[i],cols[j]) for i,j in partial]),error
parser=argparse.ArgumentParser();parser.add_argument('--go',required=True);parser.add_argument('--decisions',required=True);parser.add_argument('--output',required=True);args,private=parser.parse_known_args();assert all(arg.startswith('--audit-memory-worker=') for arg in private)
go=read(args.go);assert go['approved'] is True
for n,h in go['inputs'].items():assert sha(ROOT/n)==h,n
assert str(Path(args.decisions).resolve().relative_to(ROOT)).replace('\\','/') in go['inputs'];assert str(Path(__file__).relative_to(ROOT)).replace('\\','/') in go['inputs']
data=read(OUT/'assignment_snapshots.json');decisions=read(args.decisions);known=read(ROOT/'research_remaining_causes_20261112/independent_review/reid/low_protocol_v3.json');reports=[];maxerr=0.
for n,h in data['sources'].items():assert sha(ROOT/n)==h,n
for d in decisions['records']:
 s=next(s for s in data['snapshots'] if s['video']==d['video'] and s['frame']==d['frame']);rowids=[r['tid'] for r in s['rows']];originalcols=s['high_raw_ids'];base=np.asarray(s['corrected_cost'],np.float32)
 baseline=solve(base);assert baseline==sorted(map(tuple,s['actual_first'])),(s['video'],s['frame'],'baseline mismatch');maxerr=max(maxerr,independent(base,baseline));columns=list(originalcols)
 assert d['base'] in ['base_cost','corrected_cost'];changed=np.asarray(s[d['base']],np.float32).copy()
 for entry in d.get('append_columns',[]):
  assert entry['raw_id'] not in columns;costs=np.asarray(entry['costs'],np.float32);assert costs.shape==(len(rowids),) and np.isfinite(costs).all();columns.append(entry['raw_id']);changed=np.c_[changed,costs]
 for update in d.get('edge_updates',[]):
  i=rowids.index(update['tid']);j=columns.index(update['raw_id']);changed[i,j]=float(update['cost'])
 assert np.isfinite(changed).all()
 mode=d.get('freeze','none');assert mode in ['none','active','all'];locked=[(i,j) for i,j in baseline if mode=='all' or (mode=='active' and s['rows'][i]['state']==1)]
 for i,j in locked:assert changed[i,j]==base[i,j],('Frozen active pair cost changed',rowids[i],columns[j])
 # A deferred row cannot acquire a different edge as an accidental side effect.
 for tid in d.get('defer_tids',[]):
  i=rowids.index(tid);assert all(i!=ii for ii,jj in locked),('Cannot defer locked active row',tid);changed[i,:]=1e6
 after,error=constrained(changed,locked);maxerr=max(maxerr,error)
 pairs=[dict(tid=rowids[i],raw_id=columns[j],row=i,column=j,cost=float(changed[i,j]),state=s['rows'][i]['state']) for i,j in after]
 if 'reported_pairs' in d:assert sorted((p['tid'],p['raw_id']) for p in pairs)==sorted((p['tid'],p['raw_id']) for p in d['reported_pairs'])
 beforepairs={(rowids[i],originalcols[j]) for i,j in baseline};afterpairs={(p['tid'],p['raw_id']) for p in pairs}
 activebefore={(rowids[i],originalcols[j]) for i,j in baseline if s['rows'][i]['state']==1};active_stolen=sorted(activebefore-afterpairs)
 savedassigned={(int(tid),int(rid)) for tid,rid in s['assigned'].items()};newpairs=afterpairs-beforepairs
 ownership_conflicts=[dict(proposed_tid=tid,raw_id=rid,saved_owner=owner) for tid,rid in sorted(newpairs) for owner,ownedraw in savedassigned if ownedraw==rid and owner!=tid]
 alias_conflicts=[]
 for group in d.get('alias_groups',[]):
  for tid,rid in sorted(newpairs):
   if rid in group:
    alias_conflicts.extend(dict(proposed_tid=tid,raw_id=rid,saved_owner=owner,saved_raw_id=ownedraw,group=group) for owner,ownedraw in savedassigned if ownedraw in group and owner!=tid)
 newcolumn_state_violations=[p for p in pairs if p['column']>=len(originalcols) and p['state']!=2]
 candidate={r['raw_id']:r for r in s['raw_current_car_candidates']};identity=next(c['identity'] for c in known['cases'] if c['old_id']==s['old_case']);ownraw={rid for rid,r in candidate.items() if [h['identity'] for h in r['hits']]==[identity]}
 ownpairs=[p for p in pairs if p['raw_id'] in ownraw];beforeown=[(tid,rid) for tid,rid in beforepairs if rid in ownraw]
 def gtclaims(ps):
  claims={}
  for tid,rid in ps:
   rr=candidate.get(rid)
   if rr and len(rr['hits'])==1:claims.setdefault(rr['hits'][0]['identity'],[]).append(tid)
  return {gt:sorted(tids) for gt,tids in claims.items() if len(set(tids))>1}
 duplicates_before=gtclaims(beforepairs);duplicates_after=gtclaims(afterpairs)
 target_selected=any(p['tid']==s['target_tid'] and p['raw_id']==s['target_raw_id'] for p in pairs);own_recovered=any(p['tid']==s['target_tid'] and p['raw_id'] in ownraw for p in pairs)
 reports.append(dict(variant=d['variant'],video=s['video'],frame=s['frame'],old_case=s['old_case'],base=d['base'],freeze=mode,baseline_pairs=[dict(tid=tid,raw_id=rid) for tid,rid in sorted(beforepairs)],authoritative_saved_assigned=[dict(tid=tid,raw_id=rid) for tid,rid in sorted(savedassigned)],actual_pairs=pairs,target_edge_selected=target_selected,target_same_identity_recovered=own_recovered,foreign_target_rejected=(not target_selected) if s['label']=='foreign' else None,object_continuation_covered=bool(ownpairs),baseline_object_continuation_covered=bool(beforeown),existing_active_object_coverage=[p for p in ownpairs if p['state']==1 and p['tid']!=s['target_tid']],active_pairs_removed=[dict(tid=t,raw_id=r) for t,r in active_stolen],protected_assigned_conflicts=ownership_conflicts,occupied_alias_group_conflicts=alias_conflicts,new_column_nonlost_row_assignments=newcolumn_state_violations,new_duplicate_annotation_claims={gt:tids for gt,tids in duplicates_after.items() if duplicates_before.get(gt)!=tids},defer_tids=d.get('defer_tids',[]),added=[dict(tid=t,raw_id=r) for t,r in sorted(afterpairs-beforepairs)],removed=[dict(tid=t,raw_id=r) for t,r in sorted(beforepairs-afterpairs)],matrix_shape=list(changed.shape),matrix_changed_existing_cells=int(np.count_nonzero(changed[:,:len(originalcols)]!=base)),new_columns=columns[len(originalcols):],objective=float(objective(changed,after)),independent_residual_objective_error=error,scope='One complete saved first-assignment matrix with declared freeze/defer constraints. Saved final assigned map separately checks downstream/current ownership, including IDs outside first rows. This is not a replay of second association, public lifecycle or future trajectory. GT identities used only in evaluation after assignment.'))
for n,h in go['inputs'].items():assert sha(ROOT/n)==h,n
output=Path(args.output).resolve();output.relative_to(OUT.resolve())
with output.open('x',encoding='utf8') as f:json.dump(dict(passed=True,records=reports,max_independent_objective_error=maxerr,lap_version=lap.__version__,tracker_instantiations=0,model_calls=0,snapshot_sha256=sha(OUT/'assignment_snapshots.json')),f,ensure_ascii=False,indent=2,allow_nan=False)
print('ASSIGNMENT_AUDIT',len(reports),maxerr)
