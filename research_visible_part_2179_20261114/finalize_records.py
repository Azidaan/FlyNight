from common import *
c=read(OUT/'checks_v2.json');assert c['passed']
assert all(sha(ROOT/n)==h for n,h in c['inputs'].items())
assert read(OUT/'audit.json')['passed']
assert '58 passed' in (OUT/'history_tests_v2.log').read_text(encoding='utf8')
journal('После уточнения интерпретации прежних карточек повторно пройдены валидатор каталога и58 проверок истории. Все зарегистрированные входы, исходные веса и сохранённые признаки неизменны; независимые пересчёты192 покомпонентных и72 масштабных векторов завершены. Источники: '+OUT.relative_to(ROOT).as_posix()+'/checks_v2.json; audit.json. Предупреждения валидатора относятся к двум прежним шаблонным путям fly_input_norm_hota_v1, не к этому опыту. Готовится синхронизация производного графа и сохранение только файлов этой задачи; полный трекер и ядро не запускались.')
save('final_record_checks.json',dict(passed=True,at=now(),checks_sha256=sha(OUT/'checks_v2.json'),audit_sha256=sha(OUT/'audit.json'),history_tests=58,all_checked_inputs_unchanged=True))
print('FINAL_RECORDS_READY')
