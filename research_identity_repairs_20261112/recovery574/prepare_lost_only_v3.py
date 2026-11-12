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

policy = (OUT / 'appearance_recovery_policy.py').read_text(encoding='utf8')
policy = policy.replace('state (1 tracked/2 lost)', 'state (2 lost is the only recovery-eligible state)')
policy = policy.replace("row['state'] not in (1, 2)", "row['state'] != 2")
policy = policy.replace('row_not_tracked_or_lost', 'row_not_lost')
write('appearance_recovery_policy_v3.py', policy)
worker = (OUT / 'worker_v2.py').read_text(encoding='utf8')
worker = worker.replace('from appearance_recovery_policy import', 'from appearance_recovery_policy_v3 import')
worker = worker.replace("assert (OUT / 'GO.json').exists()", "assert (OUT / 'GO_lost_only.json').exists()")
worker = worker.replace("protocol, binding = verify()", "protocol, binding = verify()\nscope_addendum = read(OUT / 'scope_addendum_v3.json')\nfor name, digest in scope_addendum['new_sources'].items():\n    assert sha(OUT / name) == digest, name", 1)
worker = worker.replace("'evaluation_arrays.npz'", "'evaluation_arrays_lost_only.npz'")
worker = worker.replace("write('result.json'", "write('result_lost_only.json'")
write('worker_v3.py', worker)
audit = (OUT / 'audit.py').read_text(encoding='utf8')
audit = audit.replace("'result.json'", "'result_lost_only.json'")
audit = audit.replace("'evaluation_arrays.npz'", "'evaluation_arrays_lost_only.npz'")
audit = audit.replace("'appearance_recovery_policy.py'", "'appearance_recovery_policy_v3.py'")
audit = audit.replace("row['state'] not in [1, 2]", "row['state'] != 2")
audit = audit.replace('row_not_tracked_or_lost', 'row_not_lost')
audit = audit.replace("with (OUT / 'audit.json')", "with (OUT / 'audit_lost_only.json')")
audit = audit.replace("for name, digest in protocol['sources'].items():", "scope_addendum = read(OUT / 'scope_addendum_v3.json')\nfor name, digest in scope_addendum['new_sources'].items():\n    assert sha(OUT / name) == digest, name\nfor name, digest in protocol['sources'].items():")
write('audit_v3.py', audit)
addendum = dict(created=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(), parent_protocol_sha256=sha(OUT / 'proposal_protocol.json'), technical_addendum_sha256=sha(OUT / 'worker_addendum_v2.json'),
                variant='SECONDARY scope correction, not a silently substituted primary',
                registration_context='Parent fixed lost-only secondary after original worker computed a partial first-case232 policy and failed during evaluation with missing GT anchor533. No final primary result file existed at failure. Primary registered tracked/lost rule remains unchanged and must be evaluated/reported. Lost-only choice is a pre-outcome scope decision for this variant, not gate tuning; parent may execute primary while preparing this file.',
                change='Only unassigned state2 lost rows may receive new recovery proposals. State1 tracked rows now defer with row_not_lost. All5 target rows were already state2 as established by metadata before full outcomes.',
                unchanged='All appearance/proximity/NMS/native thresholds, raw/reference descriptors, competition, frozen assigned ownership, negative controls, evaluation criteria and noGT decision boundary remain unchanged.',
                new_sources={name: sha(OUT / name) for name in ['appearance_recovery_policy_v3.py', 'worker_v3.py', 'audit_v3.py', 'prepare_lost_only_v3.py']},
                output_files=['result_lost_only.json', 'evaluation_arrays_lost_only.npz', 'audit_lost_only.json'], primary_output_preserved=True)
with (OUT / 'scope_addendum_v3.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(addendum, ensure_ascii=False, indent=2).rstrip() + '\n')
context = dict(recorded=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(), parent_report='Original worker autoexecuted20:00:38 and failed KeyError533 while evaluating a non-target reference without GT clock anchor. Policy calculations for part of232 had already occurred; no completed result.json was produced.',
               correction='The earlier worker_addendum_v2 repair_outcomes_computed:false field reflected incomplete agent knowledge and must not be read as proof no computation occurred. Preserve that file/hash; this note supersedes its factual interpretation. V2 fixes only unavailable reference label evaluation to unknown; V3 is a separately registered lost-only secondary.', no_policy_threshold_change=True)
with (OUT / 'execution_failure_context.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(context, ensure_ascii=False, indent=2).rstrip() + '\n')
print('LOST_ONLY_ADDENDUM', sha(OUT / 'scope_addendum_v3.json'))
