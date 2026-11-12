from common import *
import psutil,shutil
initial=read(OUT/'initial_state.json')
for n,h in initial['old_science'].items():assert sha(ROOT/n)==h,n
audit_names=['geometry832/audit.json','review/bank_audit.json','recovery574/audit.json','recovery574/audit_lost_only.json','recovery574/audit_episode.json','review/component_checks.json','review/combined_assignment_result_v2.json','component_export_checks.json']
audits={n:read(OUT/n) for n in audit_names};assert all(a['passed'] for a in audits.values())
bound={};guards={};executions={}
for i in range(1,7):
 p=read(OUT/f'queue{i}.json');q=read(OUT/f'queue{i}_execution.json');assert q['complete']
 if i!=2:assert q['passed'],i
 executions[str(i)]=q
 for n,h in p['inputs'].items():
  assert n not in bound or bound[n]==h,n;bound[n]=h
 for job in q['jobs']:
  guards[job['token']]=job
  for pid,birth in job['result']['owned'].items():
   try:alive=psutil.Process(int(pid)).create_time()==birth
   except psutil.NoSuchProcess:alive=False
   assert not alive,(i,pid)
for n,h in bound.items():assert sha(ROOT/n)==h,n
data=read(OUT/'review/combined_assignment_result_v2.json');baseline=[r for r in data['records'] if r['variant']=='baseline'];safe=[r for r in data['records'] if r['variant']=='guarded_census_appearance_low_native_domain'];control=[r for r in data['records'] if r['variant']=='targetdecision_without_core_appearance_low_native_domain']
assert len(baseline)==len(safe)==len(control)==5
assert sum(r['object_continuation_covered'] for r in baseline)==4
assert all(r['object_continuation_covered'] for r in safe)
for r in safe:
 assert all(not r[k] for k in ['active_pairs_removed','protected_assigned_conflicts','occupied_alias_group_conflicts','new_duplicate_annotation_claims'])
 c=next(c for c in control if c['old_case']==r['old_case']);assert r['actual_pairs']==c['actual_pairs']
assert [p for r in safe for p in r['added']]==[dict(tid=574,raw_id=35280)]
d=OUT/'execution';d.mkdir(exist_ok=False)
for token,job in guards.items():
 original=Path(job['report']);assert read(original)==job['result'];shutil.copyfile(original,d/(token+'.json'));shutil.copyfile(original.with_name(original.stem+'.worker.log'),d/(token+'.log'))
science={}
for folder in ['geometry832','recovery574','review']:
 for f in (OUT/folder).glob('*'):
  if f.is_file() and f.suffix in ['.py','.json','.md','.npz','.csv']:science[f.relative_to(ROOT).as_posix()]=sha(f)
save('audit.json',dict(passed=True,checked=now(),audits=audits,verified_inputs=bound,scientific_artifacts=science,old_science_unchanged=True,all_owned_workers_ended=True,failed_original_evaluation_preserved='queue2_execution.json; recovery_v1_failure.json',event_coverage=dict(baseline=4,repaired=5,total=5),new_assignment=dict(tid=574,raw_id=35280),new_core_incremental_assignments=0,scope='Independent arithmetic/provenance of fixed cached features and complete saved first matrices. Historical states and other core votes remain fixed. No full tracker or public lifecycle.'))
print('FINAL_REPAIR_AUDIT_PASS',len(bound),len(audits))
