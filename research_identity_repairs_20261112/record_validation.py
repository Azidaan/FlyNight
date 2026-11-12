from common import *
import re,shutil
q=read(OUT/'queue7_execution.json');assert q['complete'] and q['passed'];assert len(q['jobs'])==2
for job,name in zip(q['jobs'],['catalog_validation.log','tests.log']):
 report=Path(job['report']);log=report.with_name(report.stem+'.worker.log');shutil.copyfile(log,OUT/name);shutil.copyfile(report,OUT/'execution'/(job['token']+'.json'));shutil.copyfile(log,OUT/'execution'/(job['token']+'.log'))
v=(OUT/'catalog_validation.log').read_text(encoding='utf8');t=(OUT/'tests.log').read_text(encoding='utf8');assert 'OK:' in v
match=re.search(r'(\d+) passed in ([0-9.]+)s',t);assert match and int(match[1])==79,t[-1000:]
a=read(OUT/'audit.json')
for n,h in a['scientific_artifacts'].items():assert sha(ROOT/n)==h,n
save('validation_receipt.json',dict(passed=True,checked=now(),tests_passed=int(match[1]),test_seconds=float(match[2]),new_component_tests=21,catalog_tests=58,tests_log_sha256=sha(OUT/'tests.log'),catalog_log_sha256=sha(OUT/'catalog_validation.log'),scientific_artifacts_unchanged=True,legacy_catalog_warnings=2,live_tracker_calls=0))
journal('Проверка реализации исправлений завершена:79тестов, включая21новый тест компонентов и58обязательных тестов каталога, пройдены; каталог согласован с источниками. Два прежних предупреждения относятся к старому опыту fly_input_norm_hota_v1. Новые тесты проверяют перенос без эталонных рамок, защиту занятой группы, конкурентов вне области выдачи, отсутствие признаков, сохранение правильного исходного решения и запрет подмены неизвестного решения. Научные артефакты не изменились после аудита. Источники research_identity_repairs_20261112/validation_receipt.json,tests.log,catalog_validation.log. Производные графа проверяются на копии; прежние гипотезы и чужие правки сохраняются.')
names=['publish.py','run_queue.py','validation_receipt.json','audit.json'];inputs={(OUT/n).relative_to(ROOT).as_posix():sha(OUT/n) for n in names}
for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/useful_literature.json','experiments/obsidian_experiment_topics.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:inputs[n]=sha(ROOT/n)
save('queue8.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=(OUT/'publish.py').relative_to(ROOT).as_posix(),args=['prepare'],timeout=600)],deadline='2026-11-13T07:35:00+03:00'))
print('79_TESTS_VERIFIED_PUBLICATION_REGISTERED')
