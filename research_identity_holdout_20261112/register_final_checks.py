from common import *
inputs={}
for n in ['experiments/project_history.json','experiments/metrics_glossary.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','AGENTS.md','scripts/validate_project_history.py','tests/test_project_history.py','tests/test_identity_repair_components.py','research_identity_holdout_20261112/required_checks.py']:
    inputs[n]=sha(ROOT/n)
save('queue6.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=(OUT/'required_checks.py').relative_to(ROOT).as_posix(),args=[m],timeout=900) for m in ['validate','tests']],deadline='2026-11-13T07:35:00+03:00'))
print('FINAL_CHECKS_REGISTERED')
