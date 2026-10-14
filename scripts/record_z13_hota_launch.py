"""Record verified launch and independent deadline protection, not an outcome."""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def main():
    status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
    launch = json.loads((RUN / 'launch_process.json').read_text(encoding='utf-8-sig'))
    auth = json.loads((RUN / 'full_authorization.json').read_text(encoding='utf-8'))
    guards = list(RUN.glob('deadline_guard_*/started.json'))
    armed = [json.loads(p.read_text(encoding='utf-8')) for p in guards]
    assert status['state'] == 'running'
    assert launch['guard_test_passed']
    assert any(g['state'] == 'armed' and g['deadline'] == auth['deadline'] for g in armed)
    log = Path(status['log']).read_text(encoding='utf-8', errors='replace')
    assert 'кадров' in log
    fixture = list((RUN / 'guard_verified_test').glob('deadline_guard_*/result.json'))
    assert len(fixture) == 1
    test = json.loads(fixture[0].read_text(encoding='utf-8'))
    assert test['state'] == 'deadline_reached'
    now = datetime.now().astimezone()
    proof = {'verified_at': now.isoformat(), 'launch': launch, 'status_snapshot': status,
             'guard_armed': True, 'deadline': auth['deadline'],
             'native_termination_fixture': str(fixture[0].relative_to(ROOT)),
             'progress_in_worker_log': True, 'hota_measured': False}
    (RUN / 'launch_verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    card = next(c for c in catalog['experiments'] if c['id'] == 'fly_spatial_hota_policy_z13_v1')
    assert card['state'] == 'running'
    ref = 'runs/z13_spatial_hota_v1_20261014/launch_verification.json'
    assert not any(e.get('ref') == ref for e in card['evidence'])
    card['evidence'].append({'kind': 'json', 'ref': ref,
                             'claim': 'Запуск подтверждён обработкой кадров; независимый контроль срока проверен на собственном процессе'})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — полный прогон HOTA З13 v2 запущен, контроль срока проверен

*Получено (`fly_spatial_hota_policy_z13_v1`).* `runs/z13_spatial_hota_v1_20261014/launch_verification.json`: запуск {launch['launched']}, обработка кадров первого видео B подтверждена журналом, один GPU-работник. Веса, пороги, данные и код зафиксированы в `inference_identity.json`; критерии не менялись. Завершённые задания имеют отдельные маркеры и сохраняются для продолжения.

*Контроль срока.* Разрешённое окно до {auth['deadline']}; независимый Python-процесс заранее запросит остановку и завершит только своё проверенное дерево процессов. Реальное завершение собственного тестового процесса проверено, первичный файл `{fixture[0].relative_to(ROOT).as_posix()}`. Первая служебная проба PowerShell была заблокирована политикой подписания, а запуск без повышенных разрешений не мог завершить собственный тестовый процесс. Политика не обходилась; производственный контроль реализован на Python и проверен в разрешённом запуске с необходимыми правами.

*Решение.* Выполняется одно согласованное сравнение B/C/D на dev35 с прежней MASA; без обучения, скачиваний и отложенных видео. Новых результатов HOTA ещё нет, процент к стенду пока неизвестен.

'''
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    assert '## Журнал изменений\n\n' in text
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False))


if __name__ == '__main__':
    main()
