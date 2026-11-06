"""Append verified engineering records without reformatting existing history."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/flyvis_repairs_20261106'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    now = datetime.now(timezone(timedelta(hours=3)))
    parity = json.loads((OUT / 'lifecycle_parity.json').read_text(encoding='utf-8'))
    assert parity['passed'] and parity['outputs_and_all_tracker_states_exact'] and parity['events_exact']
    card = dict(
        id='flyvis_system_lifecycle_repair_20261106',
        title='Авторская система — безопасный сброс трекеров и очистка истории',
        area='infra', state='infrastructure', started='2026-11-06', done='2026-11-06',
        question='Не сохраняется ли история предыдущего видео после reset и не сбиваются ли идентификаторы соседнего трекера?',
        hypothesis='Техническая проверка воспроизведённых нарушений состояния; это не новая научная гипотеза.',
        expected='Сброшенный трекер ведёт себя как новый; внешний счётчик и общие серверы сохраняются; обычные выходы не меняются.',
        acceptance=[
            dict(criterion='Сброс всех затронутых производных трекеров очищает их собственное состояние; внешний счётчик сохраняется. Технический критерий записан в каталог после проверки, научной предварительной регистрацией не является.', met=True,
                 evidence='runs/flyvis_repairs_20261106/lifecycle_before.log; runs/flyvis_repairs_20261106/lifecycle_after_v2.log'),
            dict(criterion='На фиксированной последовательности с ненулевой поправкой нейронной головы все выходы, состояния и решения совпадают с сохранённым исходным кодом; история удалённых объектов очищается, потерянных сохраняется.', met=True,
                 evidence='runs/flyvis_repairs_20261106/lifecycle_parity.json')],
        obtained='До исправления восемь проверок завершились ошибкой (runs/flyvis_repairs_20261106/lifecycle_before.log). После исправления 28 проверок пройдены (runs/flyvis_repairs_20261106/lifecycle_after_v2.log). На 100 искусственных кадрах все выходы и 292 события сопоставления совпали с исходным кодом; число сохранённых историй снизилось с 6 до 2 (runs/flyvis_repairs_20261106/lifecycle_parity.json).',
        interpretation='Подтверждены дефекты сброса и удаления устаревшей истории. Сравнение проверяет отсутствие изменения обычных решений на данном техническом сценарии.',
        decision='Исправить сброс собственных состояний и изоляцию счётчика; хранить историю только объектов, остающихся отслеживаемыми или потерянными. Продолжить аудит зависимых компонентов.',
        cannot_claim='Не измерено улучшение HOTA, не переобучены модели, не доказано отсутствие всех возможных ошибок на любых видео.',
        metrics=[], evidence=[
            dict(kind='json', ref='runs/flyvis_repairs_20261106/parent_baseline.json', claim='Состояние файлов до исправления и исходный коммит'),
            dict(kind='json', ref='runs/flyvis_repairs_20261106/lifecycle_parity.json', claim='Сравнение исходного и исправленного исполнения'),
            dict(kind='file', ref='tests/test_tracker_lifecycle_reset.py', claim='Воспроизводимые проверки сброса и истории')])
    catalogue = ROOT / 'experiments/project_history.json'
    before = catalogue.read_bytes()
    data = json.loads(before)
    if any(c['id'] == card['id'] for c in data['experiments']):
        raise RuntimeError('Record already exists; refusing to rewrite history')
    text = before.decode('utf-8')
    match = re.search(r'"experiments"\s*:\s*\[', text)
    assert match
    start = match.end() - 1
    _, end = json.JSONDecoder().raw_decode(text, start)
    assert text[end-1] == ']'
    insert = ',\n' + '\n'.join('    '+line for line in json.dumps(card, ensure_ascii=False, indent=2).splitlines()) + '\n  '
    # Insert after the last object and before existing trailing whitespace.
    content_end = len(text[:end-1].rstrip())
    updated = text[:content_end] + insert + text[content_end:]
    parsed = json.loads(updated)
    assert parsed['experiments'][:-1] == data['experiments']
    assert {k:v for k,v in parsed.items() if k != 'experiments'} == {k:v for k,v in data.items() if k != 'experiments'}
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    old_plan = plan.read_bytes()
    marker = '## Журнал изменений'.encode()
    assert old_plan.count(marker) == 1
    entry = f'''\n\n**{now:%d.%m.%Y, %H:%M} MSK** — Аудит авторской системы и исправление управления состоянием

*Постановка.* Автор разрешил автономно находить и исправлять подтверждённые ошибки до 07.11.2026, 01:00 MSK. Главный критерий — исправление не создаёт новых ошибок; сохранение прежних режимов само по себе не доказывает безопасность. Проверяются ядро FlyVis, полный модуль и остальные компоненты системы. Исходные файлы сохранены в `runs/flyvis_repairs_20261106/parent_baseline.zip`; исходное состояние и хеши — `runs/flyvis_repairs_20261106/parent_baseline.json`.

*Получено (`flyvis_system_lifecycle_repair_20261106`).* Первичный аудит сохранён в `runs/flyvis_system_audit_20261106/AUDIT_REPORT.txt`; проверены загрузка весов, связи, время и геометрия. Для сброса трекеров восемь исходных проверок дали ошибку; после исправления 28 проверок пройдены (`runs/flyvis_repairs_20261106/lifecycle_before.log`, `lifecycle_after_v2.log`). На 100 искусственных кадрах при ненулевой поправке головы все выходы и 292 события совпали с исходным исполнением; сохранённых историй осталось 2 вместо 6 (`runs/flyvis_repairs_20261106/lifecycle_parity.json`). История объектов со статусом «потерян» сохраняется. Исправлены сброс собственного состояния, изоляция счётчиков идентификаторов и удаление истории объектов, окончательно ушедших из трекера.

*Вывод.* Подтверждены технические дефекты управления состоянием. Совпадение решений ограничено проверенным сценарием; HOTA и польза для качества трекинга не измерены. Подозрение к коэффициенту нормализации BUSCA отклонено: значение соответствует официальному выбранному режиму и протоколу.

*Решение.* Продолжить исправления загрузки, геометрии и времени с проверкой зависимых компонентов. Старые веса, результаты и защищённое дерево гипотез не изменять. Научные выводы из технических тестов не делать.
'''.encode('utf-8')
    new_plan = old_plan.replace(marker, marker + entry, 1)
    assert new_plan.replace(marker + entry, marker, 1) == old_plan
    catalogue.write_bytes(updated.encode('utf-8'))
    plan.write_bytes(new_plan)
    receipt = dict(at=now.isoformat(), card=card['id'], catalogue_before=digest(before), catalogue_after=digest(catalogue.read_bytes()),
                   plan_before=digest(old_plan), plan_after=digest(new_plan), previous_cards_unchanged=True, previous_plan_bytes_preserved=True)
    (OUT / 'record_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
