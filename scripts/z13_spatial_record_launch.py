"""Record observed technical acceptance and live training, never inferred success."""
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_spatial_v2_20261013'


def main():
    technical = json.loads((RUN / 'technical/report.json').read_text(encoding='utf-8'))
    status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
    assert technical.get('passed') is True
    assert status['state'] == 'running' and status['stage'] != 'technical'
    catalog_path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    card = next(c for c in catalog['experiments'] if c['id'] == 'fly_spatial_quality_z13_v2')
    card['state'] = 'running'
    card['acceptance'][0].update(met=True, evidence='runs/z13_spatial_v2_20261013/technical/report.json: passed=true, arms.B/C/D')
    for ref, claim in [('runs/z13_spatial_v2_20261013/technical/report.json', 'Технические проверки и цена короткой пробы'),
                       ('runs/z13_spatial_v2_20261013/status.json', 'Состояние последовательного контроллера')]:
        if not any(e['ref'] == ref for e in card['evidence']):
            card['evidence'].append({'kind': 'json', 'ref': ref, 'claim': claim})
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    now = datetime.now().astimezone()
    entries = technical['arms']
    seconds = '/'.join(f'{entries[a]["seconds"]:.2f}'.replace('.', ',') for a in ('B', 'C', 'D'))
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — З13 v2: техническая проба пройдена, обучение запущено

*Получено (`fly_spatial_quality_z13_v2`).* `runs/z13_spatial_v2_20261013/technical/report.json`: B/C/D по 32 шага, время {seconds} с; все технические условия пройдены. Во всех вариантах считыватель содержит {entries['B']['readout_parameters']} обучаемых параметров. Подтверждены координаты, влияние замены образца на пространственный ответ, ненулевой градиент объектной ошибки в ядре D, точное восстановление весов/оптимизатора/генераторов из сохранения. Ответ после восстановления совпадает с абсолютным и относительным допуском 1e-5, учитывающим отдельно измеренную численную вариацию GPU. Это проверка исполнения, не доказательство полезной локализации или HOTA.

*Решение.* Последовательный контроллер начал основной этап (источник `runs/z13_spatial_v2_20261013/status.json`). B/C/D, два вспомогательных разбиения и полные 104 train; настройки и критерии из protocol.json не менялись. Каждая модель — 4000 шагов. После прогнозов вне обучения — обучение оценки качества на CPU, затем точный выбор порога на прежних 26 train. Автоматической оценки dev/MASA/75 val нет. Сохранения включают веса, оптимизатор, шаг, генераторы и проверку версий; сторож ограничивает перезапуски только собственных зависших заданий. Сводка: HOTA не измерялась, прирост к стенду и доля оракула не оценены.

'''
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    txt = plan.read_text(encoding='utf-8')
    marker = '## Журнал изменений\n\n'
    if 'З13 v2: техническая проба пройдена, обучение запущено' not in txt:
        plan.write_text(txt.replace(marker, marker + entry, 1), encoding='utf-8')
    print('Recorded technical acceptance and live launch')


if __name__ == '__main__':
    main()
