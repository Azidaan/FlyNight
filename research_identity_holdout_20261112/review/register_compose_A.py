"""Hash only metadata and existing stage outputs; do not inspect repair scores."""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


paths = dict(runtime='review/A_runtime_manifest_v2.json', clean_runtime='recovery/runtime_cases.json',
             bank='shared/descriptors_A.npz', recovery='recovery/result.json',
             recovery_directory='recovery', recovery_audit='recovery/audit.json',
             correspondence='correspondence/result.json', correspondence_audit='correspondence/audit.json')
paths = {k: str((BASE/v).relative_to(ROOT)).replace('\\', '/') for k, v in paths.items()}
names = [OUT/n for n in ['compose_adapter_A.py', 'compose_worker_A.py', 'audit_compose_A.py', 'register_compose_A.py', 'matrix_audit.py', 'anchored_readiness.json']]
pending = [paths['correspondence_audit']]
names += [ROOT/n for k, n in paths.items() if k != 'recovery_directory' and n not in pending]
names += [ROOT/'tracking'/n for n in ['identity_repair_adapter.py', 'appearance_recovery_policy.py', 'correspondence_repair_guard.py']]
names += [BASE/'shared/evaluation_protocol.json', BASE/'recovery/protocol.json', BASE/'correspondence/protocol.json']
for receipt in read(BASE/'recovery/result.json')['events']:
    path = BASE/'recovery'/receipt['path']
    assert sha(path) == receipt['sha256']
    names.append(path)
inputs = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in names}
# Include exact-label provenance for the downstream audit, never compose inputs.
inputs.update(read(OUT/'anchored_readiness.json')['source_sha256'])
p = dict(prepared=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(), inputs=inputs, paths=paths,
         pending_parent_binding=pending,
         events=54, expected_decision_records=216,
         variants=[dict(name='baseline', operation='Literal corrected_cost, no changes, freeze none.'),
                   dict(name='universal_census_negative_control', operation='Only selected native target vote replaced by Census/terminal_hold/native positive=>base_cost-.2 else base_cost. No ownership lock; negative control.'),
                   dict(name='guarded_census_appearance_low_native_domain', operation='Frozen production guarded target decision; all row references/all raw cars competition. Fixed borrowed appearance/proximity/alias gates, canonical geometry eligibility; freeze actual_first and all assigned; append supported low only.'),
                   dict(name='targetdecision_without_core_appearance_low_native_domain', operation='Matched target decision from appearance/ownership, same frozen eligibility/competition/low branch. Other historical C34 votes and saved states unchanged; not whole tracker without core.')],
         decision_schema=dict(root='records[54*4]', record='event_key,variant,edge_updates[{tid,raw_id,cost}],append_columns[{raw_id,costs:Nrows}],freeze,defer_tids,alias_groups; optional execution_failure'),
         missing_scores='No None=>0. Frozen production float-casts BOTH scores before branching. Any missing/nonfinite score means every nonbaseline variant execution_failure and exact baseline fallback, including appearance-only API. All54 retained; execution failure blocks deployment. No policy fixes on A.',
         component_binding='Combined low outputs must be subset of fixed_repair component proposals from same event/bank, because actual_first adds ownership protection. No labels passed to pure compose.',
         identity_addendum='Post-recovery-result descriptive target-tid→unique current expected GT, distinct from preregistered any-tid physical-object coverage. Existing coverage by another tid is not target-ID continuation; no choice/tuning use.',
         parent_GO_required=True, authorize_registered_downstream_audit_required=True,
         no_GT_for_decisions=True, no_new_variants=True, tracker_calls=0, GPU_calls=0,
         outputs=['review/decisions_A.json', 'review/compose_notes_A.json', 'review/matrix_GO_A.json', 'review/matrix_result_A.json', 'review/compose_audit_A.json'],
         deadline='2026-11-13T07:35:00+03:00')
with (OUT/'proposed_compose_protocol.json').open('x', encoding='utf8', newline='\n') as f:
    json.dump(p, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')
print('COMPOSE_A_PROTOCOL_READY', sha(OUT/'proposed_compose_protocol.json'))
