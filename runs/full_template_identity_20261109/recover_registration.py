"""Recover only an atomic journal write; registered science is not repeated."""
from geometry import *
import time

p=read(OUT/'protocol.json');verify(p)
cp=read(OUT/'comparison_protocol.json');verify(cp)
prior=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/e8c8ce7f8912483db9298abd2c5cf6cd'
assert read(prior/'memory_guard.json')['returncode']==1
log=(prior/'memory_guard.worker.log').read_text(encoding='utf8')
assert 'OSError: [Errno 22]' in log and 'journal' in log
assert not (OUT/'features.json').exists() and not (OUT/'predictions.json').exists()
g=read(OUT/'geometry_results.json');assert g['passed'] and g['gate']
write('registration_recovery_protocol.json',dict(registered=datetime.now(MSK).isoformat(),science_protocol_unchanged=True,
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),'runs/full_template_identity_20261109/protocol.json':sha(OUT/'protocol.json'),'runs/full_template_identity_20261109/comparison_protocol.json':sha(OUT/'comparison_protocol.json')},
    recovery='Only append factual geometry result and complete stage registration via atomic file replacement. No geometry, maps, features, model or criteria rerun. Failed process receipt retained.'))
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();text=before.decode('utf8');normalized=text.replace('\r\n','\n')
assert normalized.count('## Журнал изменений\n')==1
assert 'Зафиксирована проверка целого шаблона вместо пересечения всех сдвигов' in normalized
stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')
entry=f'''**{stamp}** — Геометрия целого шаблона проверена; восстановлена только запись этапа

*Получено (`{OUT.name}`).* Пройдены все 6 математических тестов. Прежние ячейки и координаты воспроизведены точно: максимальная ошибка 0. В 90 из 91 пары опора больше; медианы 23 → 49 ячеек при медиане 77 ячеек прошлой рамки. Медиана сохранённой доли прошлого шаблона 0,302632 → 0,704762. Это медианы отношений, не отношение медиан. Источник: `runs/{OUT.name}/geometry_results.json`.

*Вывод.* Усечение всеми сдвигами действительно удаляет пространственные измерения; их польза для личности ещё не проверена.

*Решение.* Условия сравнения сигналов уже зафиксированы до их вычисления: `runs/{OUT.name}/comparison_protocol.json`. Готовые 114 карт, прежние признаки и геометрические оценки переиспользуются. Функция штрафа, нулевой порог, примеры и сдвиги не меняются; одинаковые контроли сохранены. После успешного сохранения протокола процесс регистрации завершился с OSError22 на открытии Единого плана. Исходный план остался цел; восстановлена только эта запись атомарной заменой, без научного повтора. Ошибка и её прежний журнал сохранены; паспорт восстановления — `runs/{OUT.name}/registration_recovery_protocol.json`.

'''
new=normalized.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1)
temporary=path.with_name('ЕДИНЫЙ_ПЛАН.full_template_recovery.tmp')
assert not temporary.exists()
temporary.write_bytes(new.encode('utf8'))
for attempt in range(20):
    try:
        assert path.read_bytes()==before,'Concurrent journal edit; preserve both instead of replacing'
        temporary.replace(path)
        break
    except (PermissionError,OSError):
        if attempt==19:raise
        time.sleep(.25)
assert path.read_text(encoding='utf8')==new
write('registration_recovery.json',dict(passed=True,no_science_repeated=True,conditions_unchanged=True,
    original_plan_sha256=hashlib.sha256(before).hexdigest(),result_plan_sha256=sha(path),append_only_text=True,
    preserved_failed_registration_token=prior.name))
print('Registered conditions verified; journal write recovered without computation.',flush=True)
