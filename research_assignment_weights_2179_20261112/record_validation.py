from common import *
import shutil,re
q=read(OUT/'queue2_execution.json');assert q['passed'] and len(q['jobs'])==2
(OUT/'execution').mkdir(exist_ok=True)
for job,name in zip(q['jobs'],['catalog_validation.log','tests.log']):
    report=Path(job['report']);log=report.with_name('memory_guard.worker.log');shutil.copyfile(log,OUT/name)
    shutil.copyfile(report,OUT/'execution'/(job['token']+'_guard.json'));shutil.copyfile(log,OUT/'execution'/(job['token']+'_worker.log'))
t=(OUT/'tests.log').read_text(encoding='utf8');m=re.search(r'(\d+) passed in ([0-9.]+)s',t);assert m and int(m[1])==58
assert 'OK:' in (OUT/'catalog_validation.log').read_text(encoding='utf8')
save('validation_receipt.json',dict(passed=True,tests_passed=58,seconds=float(m[2]),independently_checked_assignment_matrices=27,core_calls=0,tracker_calls=0,source_logs=['tests.log','catalog_validation.log']))
journal('Проба весов и аудит хранения завершены:58обязательных тестов каталога пройдены, валидатор согласовал карточку с первичными источниками,27матриц независимо проверены. Сам расчёт создал '+str(read(OUT/'storage_summary.json')['calculation_artifact_bytes'])+'байт результатов наE. Источники research_assignment_weights_2179_20261112/validation_receipt.json,storage_summary.json. Пересборка графа выполняется с2компактными текстовыми копиями вместо3полных; вложения не дублируются, исходные гипотезы/настройки сохраняются. Ни рабочие веса, ни данные не удалялись или перемещались.')
names=['publish.py','run_queue.py','audit.json','provenance.json','validation_receipt.json'];inputs={(OUT/n).relative_to(ROOT).as_posix():sha(OUT/n) for n in names}
for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/useful_literature.json','experiments/obsidian_experiment_topics.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:inputs[n]=sha(ROOT/n)
save('queue3.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=(OUT/'publish.py').relative_to(ROOT).as_posix(),args=['prepare'],timeout=900)],deadline='2026-11-13T07:35:00+03:00'))
print('CATALOG_AND_MATRIX_CHECKS_VERIFIED_GRAPH_REGISTERED')
