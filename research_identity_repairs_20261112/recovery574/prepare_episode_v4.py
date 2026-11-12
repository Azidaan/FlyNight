import ast, hashlib, json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    source = '\n'.join(line.rstrip() for line in value.splitlines()).rstrip() + '\n'
    ast.parse(source)
    with (OUT / name).open('xb') as stream:
        stream.write(source.encode('utf8'))

policy = (OUT / 'appearance_recovery_policy_v3.py').read_text(encoding='utf8')
policy = policy.replace('*, competition_complete, enabled=True)', '*, competition_complete, eligible_tids=(), enabled=True)')
policy = policy.replace('    decisions = []', '    eligible_tids = frozenset(int(tid) for tid in eligible_tids)\n    decisions = []', 1)
policy = policy.replace("        elif not groups:", "        elif row['tid'] not in eligible_tids:\n            decision['reason'] = 'outside_observable_native_recovery_episode'\n        elif not groups:", 1)
policy += '''
def observable_episode_tids(native_proposals):
    """Recover the observable pre-core episode domain from native receipts.

    old_id is the native receipt name for an ordinary track identifier. No
    identifier value or annotation is tested. Both neural head outcomes are
    included, and evidence_valid is deliberately not a condition. Geometry
    exists only after native car/unmatched/history/gap/region/support checks.
    """
    eligible = set()
    for proposal in native_proposals:
        if proposal.get('geometry') is None:
            continue
        if proposal.get('reason') not in {'head_accept', 'head_refuse'}:
            raise ValueError('Unexpected native stage for geometry-bearing receipt')
        eligible.add(int(proposal['old_id']))
    return frozenset(eligible)
'''
write('appearance_recovery_policy_v4.py', policy)
worker = (OUT / 'worker_v3.py').read_text(encoding='utf8')
worker = worker.replace('from appearance_recovery_policy_v3 import confirm_recovery', 'from appearance_recovery_policy_v4 import confirm_recovery, observable_episode_tids')
worker = worker.replace('from appearance_recovery_policy_v3 import overlap', 'from appearance_recovery_policy_v4 import overlap')
worker = worker.replace("assert (OUT / 'GO_lost_only.json').exists()", "assert (OUT / 'GO_episode.json').exists()")
worker = worker.replace("protocol, binding = verify()", "protocol, binding = verify()\nepisode_addendum = read(OUT / 'episode_addendum_v4.json')\nfor name, digest in episode_addendum['new_sources'].items():\n    assert sha(OUT / name) == digest, name", 1)
worker = worker.replace("    case = next(c for c in lowprotocol['cases']", "    eligible_tids = observable_episode_tids(snapshot['proposals'])\n    case = next(c for c in lowprotocol['cases']", 1)
worker = worker.replace("frozen, competition_complete=mode != 'missing_competitors', enabled=mode != 'disabled')", "frozen, competition_complete=mode != 'missing_competitors', eligible_tids=eligible_tids, enabled=mode != 'disabled')")
worker = worker.replace("rows=rows, candidates=candidate_input", "rows=rows, eligible_tids=sorted(eligible_tids), candidates=candidate_input")
worker = worker.replace("'evaluation_arrays_lost_only.npz'", "'evaluation_arrays_episode.npz'")
worker = worker.replace("'result_lost_only.json'", "'result_episode.json'")
write('worker_v4.py', worker)
audit = (OUT / 'audit_v3.py').read_text(encoding='utf8')
audit = audit.replace("'result_lost_only.json'", "'result_episode.json'")
audit = audit.replace("'evaluation_arrays_lost_only.npz'", "'evaluation_arrays_episode.npz'")
audit = audit.replace("'appearance_recovery_policy_v3.py'", "'appearance_recovery_policy_v4.py'")
audit = audit.replace('"[\'old_id\']", ', '')
audit = audit.replace("        rows = record['rows']", "        eligible_tids = set()\n        for receipt in snapshot['proposals']:\n            if receipt.get('geometry') is not None:\n                assert receipt.get('reason') in ['head_accept', 'head_refuse']\n                eligible_tids.add(int(receipt['old_id']))\n        assert record['eligible_tids'] == sorted(eligible_tids)\n        rows = record['rows']", 1)
audit = audit.replace("            elif not groups:", "            elif row['tid'] not in eligible_tids:\n                reason = 'outside_observable_native_recovery_episode'\n            elif not groups:", 1)
audit = audit.replace("with (OUT / 'audit_lost_only.json')", "with (OUT / 'audit_episode.json')")
audit = audit.replace("scope_addendum = read(OUT / 'scope_addendum_v3.json')", "episode_addendum = read(OUT / 'episode_addendum_v4.json')\nfor name, digest in episode_addendum['new_sources'].items():\n    assert sha(OUT / name) == digest, name\nscope_addendum = read(OUT / 'scope_addendum_v3.json')")
write('audit_v4.py', audit)
addendum = dict(created=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(), primary_protocol_sha256=sha(OUT / 'proposal_protocol.json'), lost_scope_addendum_sha256=sha(OUT / 'scope_addendum_v3.json'),
                variant='FINAL SECONDARY engineering scope correction after broad primary produced3unknown proposals; primary and lost-only outputs preserved. Not heldout or a threshold search.',
                scientific_change='Only native-observable exposed recovery episode rows may EMIT newlow recovery proposals, and only when state2lost. All107 row references and all222 raw candidates remain forward/reverse nearest-neighbor competitors. No GT anchor or evaluation outcome filters the eligible set.',
                canonical_guard='appearance_recovery_policy_v4.py::observable_episode_tids(native_proposals): geometry present/nonNone; assert reason head_accept ORhead_refuse, include BOTH; do NOT require evidence_valid; map ordinary native old_id identifier to tid. Receipt geometry appears after pre-core car/unmatched/history/gap5or6/singletonregion/support>=3 checks.',
                source_verification='Parent independent reviewer verified the native guard precedes core/readout in runs/full_template_native_20261109/native_runtime.py and research_identity_assignment_run_20261108/full_planner.py. Snapshot receipts and source hashes retained.',
                api='confirm_recovery(..., competition_complete=bool, eligible_tids=(), enabled=True). Empty eligibility defers all unassigned rows; caller must pass canonical observable episode IDs. No hardcoded identifiers.',
                unchanged='Original borrowed appearance0.8/proximity0.5/NMS0.7/nativehigh0.25/nativelow0.1, descriptors, ownershipfreeze, all6negativecontrols and all5completepool acceptance criteria.',
                after_outcomes_context='Registered in response to primary1own+3unknown proposals and lost-only scope correction. No guard was chosen from GT anchor availability or proposal correctness. This is the LASTscope variant; no further gate search.',
                new_sources={name: sha(OUT / name) for name in ['appearance_recovery_policy_v4.py', 'worker_v4.py', 'audit_v4.py', 'prepare_episode_v4.py']},
                additional_inputs={name: sha(ROOT / name) for name in ['runs/full_template_native_20261109/native_runtime.py', 'research_identity_assignment_run_20261108/full_planner.py']},
                output_files=['result_episode.json', 'evaluation_arrays_episode.npz', 'audit_episode.json'], preserve_prior_failures=True)
with (OUT / 'episode_addendum_v4.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(addendum, ensure_ascii=False, indent=2).rstrip() + '\n')
print('EPISODE_ADDENDUM', sha(OUT / 'episode_addendum_v4.json'))
