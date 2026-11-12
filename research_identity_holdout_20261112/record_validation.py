from common import *
import re,shutil,psutil
q=read(OUT/'queue6_execution.json');assert q['complete'] and q['passed'] and len(q['jobs'])==2
for job,name in zip(q['jobs'],['catalog_validation.log','tests.log']):
    report=Path(job['report']);log=report.with_name(report.stem+'.worker.log');shutil.copyfile(log,OUT/name);shutil.copyfile(report,OUT/'execution'/(job['token']+'.json'));shutil.copyfile(log,OUT/'execution'/(job['token']+'.log'))
    for pid,birth in job['result']['owned'].items():
        try:alive=psutil.Process(int(pid)).create_time()==birth
        except psutil.NoSuchProcess:alive=False
        assert not alive,pid
v=(OUT/'catalog_validation.log').read_text(encoding='utf8');t=(OUT/'tests.log').read_text(encoding='utf8');assert 'OK:' in v
match=re.search(r'(\d+) passed in ([0-9.]+)s',t);assert match and int(match[1])==79,t[-1000:]
for n,h in read(OUT/'audit.json')['scientific_artifacts'].items():assert sha(ROOT/n)==h,n
save('validation_receipt.json',dict(passed=True,checked=now(),tests_passed=int(match[1]),test_seconds=float(match[2]),component_tests=21,catalog_tests=58,tests_log_sha256=sha(OUT/'tests.log'),catalog_log_sha256=sha(OUT/'catalog_validation.log'),scientific_artifacts_unchanged=True,live_tracker_calls=0))
journal('Итоговая проверка новойAзавершена:79существующих тестов компонентов и каталога пройдены, валидатор подтвердил источники и карточки. Это проверка реализации, не доказательство пользы для личности. Научные исходыAи прежние результаты неизменны. Источники research_identity_holdout_20261112/validation_receipt.json,tests.log,catalog_validation.log. Предстоящая пересборка графа сохранит гипотезы и настройки, положительный прежний574останется зелёным, новые неопределённые проверки—фиолетовыми.')
names=['publish.py','run_queue.py','validation_receipt.json','audit.json','audit_addendum.json'];inputs={(OUT/n).relative_to(ROOT).as_posix():sha(OUT/n) for n in names}
for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/useful_literature.json','experiments/obsidian_experiment_topics.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:inputs[n]=sha(ROOT/n)
save('queue7.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=(OUT/'publish.py').relative_to(ROOT).as_posix(),args=['prepare'],timeout=600)],deadline='2026-11-13T07:35:00+03:00'))
print('79_TESTS_VERIFIED_GRAPH_PUBLICATION_REGISTERED')
