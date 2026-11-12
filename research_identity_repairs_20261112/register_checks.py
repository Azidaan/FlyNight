from common import *
names=['required_checks.py','run_queue.py','audit.json','text_review.json','РЕЗУЛЬТАТ.md']
inputs={(OUT/n).relative_to(ROOT).as_posix():sha(OUT/n) for n in names}
for n in ['tests/test_identity_repair_components.py','tracking/census_correspondence.py','tracking/reference_template_history.py','tracking/correspondence_repair_guard.py','tracking/appearance_recovery_policy.py','tracking/identity_repair_adapter.py','experiments/project_history.json','experiments/system_problems.json','experiments/useful_literature.json','experiments/obsidian_experiment_topics.json']:inputs[n]=sha(ROOT/n)
save('queue7.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=(OUT/'required_checks.py').relative_to(ROOT).as_posix(),args=[mode],timeout=900) for mode in ['validate','tests']],deadline='2026-11-13T07:35:00+03:00'))
print('REGISTERED_CODE_AND_CATALOG_CHECKS')
