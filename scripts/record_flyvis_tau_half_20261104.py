"""Сохранить проверенный итог опыта и дополнить существующую проблему С7."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/flyvis_tau_half_20261104'
SOURCE = 'runs/flyvis_tau_half_20261104/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
    chain = json.loads((OUT / 'chain_result.json').read_text(encoding='utf-8'))
    ref = json.loads((OUT / 'streaming_reference.json').read_text(encoding='utf-8'))
    tau = json.loads((OUT / 'tau_intervention.json').read_text(encoding='utf-8'))
    assert result['complete'] and result['passed'] and not result['scientific_gate']
    assert chain['complete'] and chain['all_owned_ended'] and ref['passed']
    assert (tau['clamped_count'], tau['parameter_count']) == (51, 65)
    g, low = result['groups']['1.0'], result['groups']['0.25']
    assert g['valid'] == low['valid'] == 16
    def mean_delay(contrast, arm):
        rows = [r for r in result['rows'] if r['contrast'] == contrast and r['event'] == 'reverse'
                and r['mode'] == 'hold' and r['aggregation'] == 'last']
        return statistics.mean(r[arm]['latency_frames'] for r in rows)
    assert mean_delay(1.0, 'stock') == 2.0625 and mean_delay(1.0, 'fast') == 1.0625
    assert g['nominal_epe_fast'] > g['nominal_epe_stock'] and low['nominal_epe_fast'] > low['nominal_epe_stock']
    fact = ('В диагностической серии фиксированное уменьшение временных констант ядра вдвое сократило '
            'среднюю задержку разворота штатного модуля с 2,0625 до 1,0625 кадра при обычном контрасте '
            '(парная разница 1,0 кадра, 95% интервал [1,0; 1,0] по четырём исходным изображениям). '
            'При слабом контрасте задержка сократилась с 3,0625 до 2,5625 кадра. '
            'При этом номинальная ошибка оценки движения выросла с 1,8166 до 2,0642 при обычном '
            'и с 2,4254 до 2,4597 при слабом контрасте. Заранее заданный общий критерий не пройден; '
            'набор охватывает искусственное движение на четырёх изображениях и одну модель.')
    report_md = ('# Уменьшение временных констант FlyVis\n\n' + fact + '\n\n'
        'Из 65 параметров у 51 уменьшенное значение достигло штатного нижнего ограничения шагом '
        'интегрирования 1/60 с; остальные веса и считыватель сохранены. Модель и считыватель не обучались. '
        '64 входные последовательности и штатный контроль переиспользованы по хешам; все сохранённые '
        'результаты и вычисления метрик проверены. Один полный случай дополнительно совпал с отдельным '
        'проходом через потоковый интерфейс на CPU (максимальное расхождение 0 при допуске 0,00001). '
        'Это не независимый повтор всей сети на 64 случаях.\n\n'
        'Проблема остаётся открытой: сокращение задержки получено ценой роста ошибки. '
        'Данные не показывают пользу для локализации отдельных объектов или HOTA трекера. '
        'Адаптивное изменение динамики не проверено; вывод относится к одной фиксированной правке.\n\n'
        'Первичные источники: `report.json`, `tau_intervention.json`, `streaming_reference.json`, '
        '`protocol.json`, `chain_result.json`.\n')
    (OUT / 'РЕЗУЛЬТАТ.md').write_text(report_md, encoding='utf-8')
    catalog = ROOT / 'experiments/project_history.json'
    problems = ROOT / 'experiments/system_problems.json'
    for path in (catalog, problems):
        shutil.copy2(path, OUT / (path.stem + '_before_result.json'))
    data = json.loads(catalog.read_text(encoding='utf-8'))
    card = next(c for c in data['experiments'] if c['id'] == 'flyvis_tau_half_v1')
    assert card['state'] == 'planned'
    card.update(state='completed_negative', done='2026-11-04', obtained=fact + ' Источник: ' + SOURCE + 'report.json.',
        interpretation='Однократное фиксированное изменение ускорило отклик в контролируемой постановке, но увеличило ошибку; основной составной критерий не выполнен.',
        decision='Не считать проблему С7 решённой и не внедрять эту правку как готовое решение; сохранить частное уменьшение задержки и ухудшение точности.',
        cannot_claim='Не оценивались реальные траектории объектов, HOTA, другие модели ансамбля или адаптивное правило; четыре исходных изображения не являются 64 независимыми видео.')
    card['acceptance'][0].update(met=True, evidence=SOURCE+'report.json:groups.1.0')
    card['acceptance'][1].update(met=False, evidence=SOURCE+'report.json:groups')
    card['evidence'] += [dict(kind='json', ref=SOURCE+'report.json', claim='Парное сравнение всех условий и проверка метрик'),
                         dict(kind='json', ref=SOURCE+'streaming_reference.json', claim='Одна полная независимая потоковая сверка')]
    catalog.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    data = json.loads(problems.read_text(encoding='utf-8'))
    item = next(p for p in data['problems'] if p['id'] == 'С7')
    item['cases'].append(dict(date='2026-11-04', fact=fact, evidence=[SOURCE+'report.json', SOURCE+'tau_intervention.json', SOURCE+'streaming_reference.json', SOURCE+'protocol.json']))
    for name, claim in [('report.json','Парная проверка: задержка уменьшилась, ошибка выросла'),
                        ('tau_intervention.json','Изменённая группа параметров и ограничение шага'),
                        ('streaming_reference.json','Отдельная потоковая сверка одного случая'),
                        ('protocol.json','Критерии до расчёта')]:
        item['evidence'].append(dict(source=SOURCE+name, sha256=digest(OUT/name), claim=claim))
    item['cards'].append('flyvis_tau_half_v1')
    item['next_check'] = ('Первая фиксированная правка ускорила реакцию ценой роста ошибки. '
        'До следующей проверки зафиксировать одну обоснованную альтернативу с согласованным контролем '
        'точности и влияния шага интегрирования; затем, при успехе, проверить отложенные реальные последовательности.')
    item['limitations'] += (' Испытана только фиксированная половина временных констант; 51 из 65 '
                            'ограничена численным шагом. Потоковая сверка всей нейросети выполнена на одном случае.')
    problems.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    now = datetime.now(ZoneInfo('Europe/Moscow'))
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    raw = plan.read_bytes(); mark = '## Журнал изменений'.encode(); nl = b'\r\n' if mark+b'\r\n' in raw else b'\n'
    entry = f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Проверка уменьшения временных констант FlyVis\n\n{fact} Источник `{SOURCE}report.json`; протокол `{SOURCE}protocol.json`; отдельная потоковая сверка одного случая `{SOURCE}streaming_reference.json`. Из 65 констант 51 ограничена шагом 1/60 с (`{SOURCE}tau_intervention.json`). Вывод: составной критерий не выполнен, С7 остаётся открытой; адаптивная динамика и польза для HOTA не проверены.\n\n'
    assert raw.count(mark+nl) == 1
    plan.write_bytes(raw.replace(mark+nl, mark+nl+entry.encode(), 1))
    roadmap = ROOT / 'FLYVIS_TRACKER_RESEARCH_ROADMAP.md'
    with roadmap.open('a', encoding='utf-8') as f:
        f.write('\n\n## 04.11.2026 — фиксированное ускорение динамики FlyVis\n\n' + fact + ' Источник: `' + SOURCE + 'report.json`. С7 открыта; адаптивная динамика и эффект для трекера не проверены.\n')
    print('Результат и случай С7 сохранены.')


if __name__ == '__main__':
    main()
