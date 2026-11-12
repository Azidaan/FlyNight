from common import *
import shutil
q=read(OUT/'queue7_execution.json');assert q['passed'] and q['complete'];job=q['jobs'][0];guard=Path(job['report']);log=guard.with_name(guard.stem+'.worker.log');s=log.read_text(encoding='utf8');assert '58 passed' in s
shutil.copyfile(log,OUT/'tests.log');shutil.copyfile(guard,OUT/'execution'/(job['token']+'.json'));shutil.copyfile(log,OUT/'execution'/(job['token']+'.log'))
assert 'OK:' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
a=read(OUT/'audit.json')
for n,h in a['scientific_artifacts'].items():assert sha(ROOT/n)==h,n
save('validation_receipt.json',dict(passed=True,recorded=now(),tests_passed=58,test_seconds=181.62,test_log_sha256=sha(OUT/'tests.log'),catalog_log_sha256=sha(OUT/'catalog_validation.log'),legacy_warnings=2,scientific_artifacts_unchanged=True,large_tensor_archives='Preserved locally with exact hashes in audit.json; excluded from publication. Original source inputs/weights remain outside this task commit.'))
journal('Проверка записей разбора832/574: validate_project_history.py — OK, новых ошибок нет; два прежних предупреждения относятся к старому fly_input_norm_hota_v1. Все58обязательных тестов каталога пройдены за181,62с. Независимые первичные проверки и контрольные суммы сохранены, собственные расчётные процессы завершены. Источники research_remaining_causes_20261112/validation_receipt.json,tests.log,catalog_validation.log. Копия графа будет проверена перед применением; гипотезы, настройки и чужие незавершённые правки сохраняются.')
inputs={}
for n in ['publish.py','run_queue.py','validation_receipt.json','audit.json']:
 f=OUT/n;inputs[f.relative_to(ROOT).as_posix()]=sha(f)
for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/obsidian_experiment_topics.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:inputs[n]=sha(ROOT/n)
save('queue8.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=(OUT/'publish.py').relative_to(ROOT).as_posix(),args=['prepare'],timeout=600)],deadline='2026-11-13T07:35:00+03:00'))
print('VALIDATED_AND_REGISTERED_PUBLICATION')
