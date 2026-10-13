"""Record the verified infrastructure recovery without changing research criteria."""
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_spatial_v2_20261013'


def main():
    proof_path = RUN / 'recovery_20261013_io/report.json'
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    status = json.loads((RUN / 'B/fold0/status.json').read_text(encoding='utf-8'))
    controller = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
    assert proof['state'] == 'recovered' and proof['all_training_state_exactly_preserved']
    assert status['step'] > proof['step'] and status['checkpoint_verified']
    assert controller['state'] == 'running'
    proof['verified_resume'] = {'job_status': status, 'controller_status': controller}
    proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    catalog_path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    card = next(c for c in catalog['experiments'] if c['id'] == 'fly_spatial_quality_z13_v2')
    evidence = {'kind': 'json', 'ref': proof_path.relative_to(ROOT).as_posix(),
                'claim': 'Восстановление после WinError 5: сохранён шаг 3610, веса/оптимизатор/генераторы точно сохранены; продолжение подтверждено новым сохранением.'}
    assert not any(e.get('ref') == evidence['ref'] for e in card['evidence'])
    card['evidence'].append(evidence)
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    now = datetime.now().astimezone()
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — З13 v2: восстановление после отказа Windows в записи прогресса

*Получено (`fly_spatial_quality_z13_v2`).* В 21:46 контроллер остановился на B/fold0: `PermissionError: [WinError 5]` при замене `status.json`; источник — `runs/z13_spatial_v2_20261013/controller_20261013_204411/log.txt`. Последний файл прогресса показывал 3600, но проверенное сохранение содержит шаг 3610 и всю историю до него. Отказ не является отрицательным результатом обучения; процесс, удерживавший файл, не установлен.

*Исправление.* Для атомарной замены файлов добавлены ограниченные повторы при отказе доступа; контроллер повторяет прерванное задание при такой ошибке, максимум два раза, только из сохранения. Проверки `tests/test_z13_atomic_retry.py` и `tests/test_z13_spatial.py`: 6 пройдены, включая настоящую блокировку Windows читателем. Формулы обучения, данные, разбиения, число шагов и критерии не изменены. До продолжения сохранена исходная копия checkpoint; изменение подписи трёх служебных файлов проведено отдельно, с точным сравнением всех остальных полей, весов, оптимизатора и генераторов. Источник — `runs/z13_spatial_v2_20261013/recovery_20261013_io/report.json`.

*Решение.* В 22:50 обучение возобновлено с шага 3610. Новое подтверждённое сохранение: {status['step']} из 4000, B/fold0; снимок состояния приложен к источнику. Один рабочий процесс GPU, низкий приоритет. Продолжается ранее разрешённое обучение B/C/D и выбор порога только на train; dev, MASA и отложенная проверка не запускались. HOTA не измерялась, процента прироста к стенду нет.

'''
    path = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = path.read_text(encoding='utf-8')
    path.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print('Recorded resume at', status['step'])


if __name__ == '__main__':
    main()
