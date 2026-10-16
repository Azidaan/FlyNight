"""Append bounded interpretation of the completed spatial-target pilot."""
from pathlib import Path
import json
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'runs/z13_distribution_pilot_20261015'
assert not (out / 'interpretation_recorded.json').exists()
report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
assert report['all_gates_passed']
now = datetime.now(timezone(timedelta(hours=3)))
source = 'runs/z13_distribution_pilot_20261015/report.json'
note = ('Все три критерия функционального пилота выполнены, но сравнение с исходной D не подтверждает улучшение: '
        'D original 17,58%, D hard 12,88%, D spatial 18,81% полезных положений. '
        'D spatial−original +1,23 п.п., ДИ [−1,05; +3,43]; D spatial−C spatial +2,52 п.п., '
        'ДИ [−0,53; +5,60]. Преимущество над hard частично связано с ухудшением hard относительно исходной модели; '
        'причина ухудшения этим опытом не установлена. Все интервалы описательные на прежних train26. '
        'Ядра при продолжении фиксированы; это сравнение целей обучения считывателя, а не новое обучение ядра. '
        'Полезное положение — геометрическое предложение на примере с присутствующей известной целью, '
        'не точность вмешательств после отказа и не HOTA. Источник: ' + source + '.')
path = ROOT / 'experiments/project_history.json'
cat = json.loads(path.read_text(encoding='utf-8'))
card = next(c for c in cat['experiments'] if c['id'] == 'z13_distribution_pilot_v1')
card.setdefault('revisions', []).append(dict(date=now.date().isoformat(), commit='', field='interpretation',
    old=card['interpretation'], reason='Добавлены сравнения с исходной D и C; предотвращение завышенного вывода по ухудшившемуся hard-контролю.'))
card['interpretation'] += ' ' + note
path.write_text(json.dumps(cat, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
text = plan.read_text(encoding='utf-8')
anchor = '## Журнал изменений\n\n'
assert text.count(anchor) == 1
entry = f'**{now:%d.%m.%Y, %H:%M} MSK** — граница положительного функционального пилота З13\n\n{note}\n\nРешение: сохранить удачное парное сравнение как основание для подготовки применения, но не объявлять улучшение исходной модели или стенда. Новые научные расчёты этим монитором не запускаются.\n\n'
plan.write_text(text.replace(anchor, anchor + entry, 1), encoding='utf-8')
research = ROOT / 'research_flyvis_decision_trees_2026-10-15/СВОДКА.md'
old = research.read_text(encoding='utf-8')
research.write_text(old + '\n## 7. Итог продолжения, 15.10.2026\n\n'
    'Все шесть продолжений по 500 шагов завершены в 19:16 MSK; сохранены веса, состояния оптимизаторов и прогнозы на всех прежних 26 видео train. '
    'D spatial−hard: +5,94 п.п. полезных положений [ +3,98; +7,97 ]; среднее медиан относительной ошибки снизилось с 0,3939 до 0,3280. '
    'D spatial−B spatial: +7,17 п.п. [ +3,42; +11,05 ]. Три критерия выполнены.\n\n'
    + note + '\n\n'
    'Проба деревьев остаётся отрицательной. Пространственная цель прошла свой функциональный допуск; '
    'следующий содержательный шаг при продолжении — подготовка применения сохранённого считывателя с проверкой качества отказа на train, '
    'без перебора ширины распределения и без переноса прежнего оценщика качества как заведомо подходящего. '
    'Это предложение, не выполненное обучение или разрешённый этим монитором новый опыт. '
    'Прежняя постановка в разделе 5 сохранена. HOTA нового варианта не измерена, процент к стенду и доля оракула неизвестны.\n', encoding='utf-8')
(out / 'interpretation_recorded.json').write_text(json.dumps({'recorded': now.isoformat(), 'source': source}, indent=2), encoding='utf-8')
print(note)
