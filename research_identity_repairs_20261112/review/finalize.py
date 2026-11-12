"""Read-only evidence aggregation after registered parent execution."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
d=read(OUT/'combined_assignment_result_v2.json');assert d['passed'] and len(d['records'])==20
safe='guarded_census_appearance_low_native_domain';control='targetdecision_without_core_appearance_low_native_domain'
fields=['variant','old_case','target_edge_selected','target_same_identity_recovered','object_continuation_covered','baseline_object_continuation_covered','existing_active_object_coverage','active_pairs_removed','protected_assigned_conflicts','occupied_alias_group_conflicts','new_column_nonlost_row_assignments','new_column_native_domain_violations','new_duplicate_annotation_claims','added','removed','defer_tids','independent_residual_objective_error']
summaries=[]
for variant in sorted({r['variant'] for r in d['records']}):
 rows=[r for r in d['records'] if r['variant']==variant]
 summaries.append(dict(variant=variant,object_coverage=sum(r['object_continuation_covered'] for r in rows),cases=len(rows),added=[dict(old_case=r['old_case'],**x) for r in rows for x in r['added']],removed=[dict(old_case=r['old_case'],**x) for r in rows for x in r['removed']],active_pairs_removed=sum(len(r['active_pairs_removed']) for r in rows),ownership_conflicts=sum(len(r['protected_assigned_conflicts']) for r in rows),alias_conflicts=sum(len(r['occupied_alias_group_conflicts']) for r in rows)))
for case in {r['old_case'] for r in d['records']}:
 a=next(r for r in d['records'] if r['old_case']==case and r['variant']==safe);b=next(r for r in d['records'] if r['old_case']==case and r['variant']==control)
 assert sorted((p['tid'],p['raw_id']) for p in a['actual_pairs'])==sorted((p['tid'],p['raw_id']) for p in b['actual_pairs'])
 assert not any(a[k] for k in ['active_pairs_removed','protected_assigned_conflicts','occupied_alias_group_conflicts','new_column_nonlost_row_assignments','new_column_native_domain_violations','new_duplicate_annotation_claims'])
names=['assignment_snapshots.json','reference_extraction_plan.json','reference_bank.npz','bank_result.json','bank_audit.json','combined_protocol_v2.json','combined_adapter_v2.py','combined_worker_v2.py','assignment_audit_v2.py','combined_decisions_v2.json','combined_policy_notes_v2.json','combined_assignment_result_v2.json','component_checks.json']
sources={str((OUT/n).relative_to(ROOT)).replace('\\','/'):sha(OUT/n) for n in names}
for n in ['research_identity_repairs_20261112/recovery574/appearance_recovery_policy_v4.py','research_identity_repairs_20261112/recovery574/result_episode.json','research_identity_repairs_20261112/geometry832/result.json','research_identity_repairs_20261112/geometry832/audit.json']:sources[n]=sha(ROOT/n)
out=dict(passed=True,records=[{k:r[k] for k in fields} for r in d['records']],summaries=summaries,source_sha256=sources,independent_assignment_objective_error=d['max_independent_objective_error'],matched_target_decision_control_assignments_identical_all5=True,additional_new_core_assignment_effect=0,scope='20variant/event records on5knownscenes, not20independentsamples. One saved complete assignment only; historicalstates/otherC34votesfixed. These are current evidence fingerprints, not retroactively assigned historical checksums.')
with (OUT/'final_evidence.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(sha256=sha(OUT/'final_evidence.json'),summaries=summaries,assignment_sha256=sources['research_identity_repairs_20261112/review/combined_assignment_result_v2.json']),ensure_ascii=False))
