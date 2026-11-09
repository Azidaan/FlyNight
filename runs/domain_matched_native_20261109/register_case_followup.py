"""Fix six-case followup before the new C34 trajectory exists."""
from evaluation_common import *
assert not (OUT/'rollouts/C34').exists()
write('case_followup_protocol.json',dict(registered=datetime.now(MSK).isoformat(),new_C34_native_not_started=True,
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OUT/'compare_previous_cases.py',OUT/'evaluation_common.py',OUT/'execution_protocol.json',OUT/'protocol.json',OUT/'previous_contexts_check.json',OLD/'id_fate.json']},
    question='For every old directly labeled C34 case, does the new full trajectory actually fix the old failure, through raw reference correspondence rather than matching numeric IDs?',
    selection='Allsixpreviousdirectcases,allfivefixedarms, no favorable selection. Head/global/dedup/publicID stages separated. Absence of a proposal stays unexplained unless the actual saved context shows it. This is development diagnosis, not independent validation.'))
print('SIX_CASE_FOLLOWUP_REGISTERED',flush=True)
