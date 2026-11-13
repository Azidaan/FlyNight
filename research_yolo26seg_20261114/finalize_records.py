from common import *
c=read(OUT/'checks.json');assert c['passed'] and read(OUT/'audit.json')['passed']
assert all(sha(ROOT/n)==h for n,h in c['inputs'].items())
assert '58 passed' in (OUT/'history_tests.log').read_text(encoding='utf8')
journal('После итогов двух проверок пройдены валидатор каталога и58проверокистории. Все исходные кадры, карты и веса сохранили зарегистрированные контрольные суммы; четыре очередиGPU завершены. Независимо пересчитаны192вектора сравнения,450замероввремени и15таблиц качества малых считывателей. Источники '+OUT.relative_to(ROOT).as_posix()+'/audit.json; checks.json; flyvis_probe/independent_assignment_review.json. Валидатор сохранил два прежних предупреждения о шаблонных путях fly_input_norm_hota_v1 и предупреждение о60-секундном лимите необязательной статистикиGit; согласованность научных источников прошла. Новые варианты не прошли научные допуски; чужие рабочие изменения сохраняются.')
save('final_record_checks.json',dict(passed=True,checked=now(),checks_sha256=sha(OUT/'checks.json'),audit_sha256=sha(OUT/'audit.json'),history_tests=58))
print('FINAL_RECORDS_READY')
