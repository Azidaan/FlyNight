"""Record compiler results with primary provenance, without HOTA claims."""
from datetime import datetime
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_triton_hota_v1_20261014'


def main():
    report = json.loads((RUN / 'report.json').read_text(encoding='utf-8'))
    # Preserve available full-video timing even when a gate aborts the probe.
    timings = {}
    video = '00067cfb-e535423e'
    for arm, reference in [('D', RUN / 'technical/eager_D'),
                           ('B', ROOT / 'runs/z13_spatial_hota_v1_20261014/B/signatures')]:
        eager = reference / video / 'signatures.json'
        compiled = RUN / f'technical/compiled_{arm}' / video / 'signatures.json'
        if eager.is_file() and compiled.is_file():
            old = json.loads(eager.read_text(encoding='utf-8'))
            new = json.loads(compiled.read_text(encoding='utf-8'))
            timings[arm] = {'eager_seconds': old['seconds'], 'compiled_seconds': new['seconds'],
                            'wall_speedup': old['seconds'] / new['seconds'],
                            'same_event_counts': old['z13_events'] == new['z13_events'],
                            'sources': [str(eager.relative_to(ROOT)), str(compiled.relative_to(ROOT))]}
    (RUN / 'video_timing_evidence.json').write_text(json.dumps(timings, indent=2) + '\n', encoding='utf-8')
    path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    card = next(c for c in catalog['experiments'] if c['id'] == 'z13_triton_execution_v1')
    assert card['state'] in ('planned', 'running')
    now = datetime.now().astimezone()
    card['done'] = now.strftime('%Y-%m-%d')
    card['state'] = 'infrastructure' if report['passed'] else 'completed_negative'
    card['obtained'] = 'runs/z13_triton_hota_v1_20261014/report.json: ' + json.dumps(report, ensure_ascii=False) + '; runs/z13_triton_hota_v1_20261014/video_timing_evidence.json: ' + json.dumps(timings, ensure_ascii=False)
    card['interpretation'] = ('Отдельное выполнение сохраняет проверенные решения и ускоряет одно парное видео. Это техническая проверка, не результат HOTA.' if report['passed']
                              else 'Технический критерий этой реализации не выполнен; причина указана в первичном файле. Возможность ускорения других узлов не опровергнута.')
    card['decision'] = ('Допущен отдельный полный HOTA-прогон с теми же весами/порогами и прежним сроком до 15.10 08:46:23 MSK.' if report['passed']
                        else 'Полный прогон на этом исполнении не ставится; обычные результаты и возможность их продолжения сохранены.')
    card['acceptance'][0].update(met=len(report.get('models', {})) == 3,
        evidence='runs/z13_triton_hota_v1_20261014/sequence_report.json: сравнение состояния, вероятностей и решений B/C/D на одинаковой последовательности')
    card['acceptance'][1].update(met=bool(report['passed']),
        evidence='runs/z13_triton_hota_v1_20261014/report.json: совпадение номеров/рамок/событий и парное время полного видео B/D')
    card['evidence'].append({'kind': 'json', 'ref': 'runs/z13_triton_hota_v1_20261014/report.json',
                             'claim': 'Одна техническая проверка отдельного исполнения; скорость и совпадение, без HOTA'})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    content = plan.read_text(encoding='utf-8')
    summary = json.dumps(report, ensure_ascii=False) + '; `video_timing_evidence.json`: ' + json.dumps(timings, ensure_ascii=False)
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — итог технической пробы отдельного исполнения Triton

*Получено (`z13_triton_execution_v1`).* Первичный файл `runs/z13_triton_hota_v1_20261014/report.json`: {summary}. Первая попытка не нашла TinyCC, поскольку зависимость установлена в отдельный каталог, а поиск шёл по стандартному platlib. Задан явный путь CC к комплектному компилятору; первичная ошибка сохранена в `report.attempt1.json`. Следующая попытка достигла лимита перекомпиляций из-за переменного числа глаз (1–4); ошибка сохранена в `report.attempt2.json`, для той же функции включены динамические размеры. Математика, допуски и критерии не менялись. При служебном запуске исправлены установка Idle после Start-Process и формат времени PowerShell (7 десятичных знаков несовместимы с Python 3,10); исходный срок 20-минутной пробы не продлевался. Независимый ограничитель затем запущен для собственного дерева.

*Вывод.* {card['interpretation']}

*Решение.* {card['decision']} Новая HOTA не измерена, процент к стенду неизвестен. Ускорение узла не считается улучшением качества FlyVis.

'''
    plan.write_text(content.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(card['state'])


if __name__ == '__main__':
    main()
