from common import *
assert 'FAIL' not in (OUT/'catalog_validation.log').read_text(encoding='utf8')
names=['required_checks.py','run_queue.py','audit.json','РЕЗУЛЬТАТ.md','text_review_v2.json','catalog_validation.log']
inputs={}
for n in names:
 f=OUT/n;inputs[f.relative_to(ROOT).as_posix()]=sha(f)
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json']:inputs[n]=sha(ROOT/n)
save('queue7.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=(OUT/'required_checks.py').relative_to(ROOT).as_posix(),timeout=900)],deadline='2026-11-13T07:35:00+03:00'))
print('REGISTERED_REQUIRED_CHECKS')
