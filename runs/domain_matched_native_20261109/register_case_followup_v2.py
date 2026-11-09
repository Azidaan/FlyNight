"""Register descriptive matching after C34 started; retain the failed precheck."""
from evaluation_common import *
assert (OUT/'rollouts/C34').exists() and not (OUT/'comparison_report.json').exists() and not (OUT/'id_fate.json').exists()
selection=read(OUT/'previous_contexts_check.json');started=read(OUT/'native_C34.started.json')
assert selection['new_C34_native_not_started'] and len(selection['records'])==6
write('case_registration_attempt.json',dict(passed=False,scientific_failure=False,failed_source='register_case_followup.py',source_sha256=sha(OUT/'register_case_followup.py'),
    reason='C34 started while the descriptive followup adapter was being written. Original assertion correctly stopped before any protocol/output write. No scientific work repeated; no pre-start registration claimed for the later matching adapter.'))
write('case_followup_protocol.json',dict(registered=datetime.now(MSK).isoformat(),new_C34_native_not_started=False,
    previous_failed_precheck='case_registration_attempt.json',case_selection_before_C34=True,matching_rules_registered_after_C34_started=True,
    native_stage_at_registration=read(OUT/'status.json'),
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OUT/'register_case_followup.py',OUT/'case_registration_attempt.json',OUT/'compare_previous_cases.py',OUT/'evaluation_common.py',OUT/'execution_protocol.json',OUT/'protocol.json',OUT/'previous_contexts_check.json',OLD/'id_fate.json']},
    question='For every old directly labeled C34 case, does the new full trajectory fix the old failure when IDs are linked through a reference raw observation?',
    selection='Sixcases alreadyfixed in previous_contexts_check before C34. Allfivearms, no favorable selection. Matching adapter registered after C34 began, before aggregate evaluation/fate exists. Descriptive diagnosis only; no new primary success criterion and no threshold selection. Missing proposals have no invented cause.'))
print('SIX_CASE_DESCRIPTIVE_FOLLOWUP_REGISTERED_AFTER_NATIVE_START',flush=True)
