"""Record only completed primary probe reports, with their actual scope."""
from datetime import datetime
from pathlib import Path
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'experiments/project_history.json'
catalog = json.loads(path.read_text(encoding='utf-8'))
now = datetime.now().astimezone()
entries = []
for eid, run, arm in (
    ('z13_cache_balanced_v1', 'runs/z13_cache_balanced_v1_20261015', 'D'),
    ('z13_cache_followup_c_v1', 'runs/z13_cache_followup_c_v1_20261015', 'C'),
    ('z13_cache_clip_v1', 'runs/z13_cache_clip_v1_20261015', 'D, фрагмент')):
    directory = ROOT / run
    report_path = directory / 'report.json'
    status_path = directory / 'status.json'
    card = next(c for c in catalog['experiments'] if c['id'] == eid)
    if card['state'] not in ('planned', 'running'):
        continue
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    if status.get('state') == 'running':
        card['state'] = 'running'
        continue
    if not report_path.exists() or status.get('state') != 'completed':
        continue
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['passed']
    if 'summary' in report:
        positive = all(s['cache_benefit_supported_within_this_sequence'] for s in report['summary'].values())
        rows = []
        for batch, s in report['summary'].items():
            rows.append(f"{arm}, пакет {batch}: медиана Triton {s['median_seconds']['triton'] * 1000:.3f} мс, кэш {s['median_seconds']['cache'] * 1000:.3f} мс; экономия {s['saving_percent']:.3f}% [{s['paired_block_interval_percent'][0]:.3f}; {s['paired_block_interval_percent'][1]:.3f}], {run}/report.json:summary.{batch}")
        api = {name: float(np.median(values)) * 1e6 for name, values in report['parameter_api'].items()}
        rows.append(f"API параметров (медиана, мкс): {api}; возврат с проверками {report['guarded_return_microseconds']:.3f} мкс, без {report['plain_return_microseconds']:.3f} мкс; {run}/report.json:parameter_api,guarded_return_microseconds,plain_return_microseconds")
        interpretation = 'Экономия в нейронной последовательности подтверждена повторными блоками на фиксированных входах; это не независимые видео и не полный конвейер. Кэш исключает повторную подготовку параметров; проверки версий существенно дешевле обычной подготовки.'
    else:
        positive = report['benefit_in_both_pairs']
        rows = [json.dumps(report['runs'], ensure_ascii=False) + '; ' + json.dumps(report['pairs'], ensure_ascii=False) + f'; {run}/report.json:runs,pairs']
        interpretation = 'Две пары короткого фрагмента измеряют реальный конвейер до MASA. Проверено совпадение номеров, счётчиков решений и координат в допуске. Полный набор и HOTA этим не проверены.'
    card.update(state='infrastructure' if positive else 'inconclusive', done=now.strftime('%Y-%m-%d'),
                obtained='; '.join(rows), interpretation=interpretation,
                decision='При успехе всех трёх проверок подготовить отдельное исполнение полного прогона с кэшем; веса, пороги, набор и срок 07:40 сохраняются. Иначе продолжить исходный Triton.')
    card['acceptance'][0].update(met=True, evidence=run + '/report.json:correctness' if 'summary' in report else run + '/report.json:pairs')
    card['acceptance'][1].update(met=positive, evidence=run + '/report.json:summary' if 'summary' in report else run + '/report.json:benefit_in_both_pairs')
    card['evidence'].append({'kind': 'json', 'ref': run + '/report.json', 'claim': 'Первичные времена, повторные блоки и совпадение вычислений'})
    formatted_rows = re.sub(r'(?<=\d)\.(?=\d)', ',', '; '.join(rows))
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — итог технической проверки `{eid}`

*Получено.* {formatted_rows}

*Вывод.* {interpretation} Критерий экономии {'выполнен' if positive else 'не выполнен'}. Источник — `{run}/report.json`.

*Решение.* {card['decision']} HOTA не измерялась; проценты относятся к времени, доля измеренного потолка времени неизвестна.

'''
    entries.append(entry)
path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
if entries:
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + ''.join(entries), 1), encoding='utf-8')
print('Recorded completed probes:', len(entries))
