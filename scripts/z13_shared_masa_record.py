"""Record the primary parity/timing result without changing scientific criteria."""
from datetime import datetime
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json
RUN = ROOT / 'runs/z13_shared_masa_probe_v1_20261015'


def main():
    now = datetime.now().astimezone()
    report = json.loads((RUN / 'report.json').read_text(encoding='utf-8'))
    assert json.loads((RUN / 'status.json').read_text(encoding='utf-8'))['state'] == 'completed'
    assert report['passed'] and report['embedding_gate_passed'] and report['fixed_stitcher_input_gate_passed']
    path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    card = next(c for c in catalog['experiments'] if c['id'] == 'z13_shared_masa_probe_v1')
    assert card['state'] == 'running'
    for field in ('cannot_claim', 'decision'):
        card.setdefault('revisions', []).append({'date': now.strftime('%Y-%m-%d'), 'commit': '',
            'field': field, 'old': card[field], 'reason': 'Получен результат заранее зафиксированной технической пробы; прежняя постановка сохранена.'})
    for criterion in card['acceptance']:
        criterion.update(met=True, evidence='runs/z13_shared_masa_probe_v1_20261015/report.json')
    def ru(value):
        return f'{value:.3f}'.replace('.', ',')
    source = 'runs/z13_shared_masa_probe_v1_20261015/report.json'
    card.update(state='completed_positive', done=now.strftime('%Y-%m-%d'),
        obtained=f'Техническая проба на одном полном видео 00067cfb-e535423e: отдельные B/C/D {ru(report["reference_total_seconds"])} с, совместный {ru(report["shared"]["including_model_load_seconds"])} с, экономия {ru(report["reduction_percent"])}%; {source}. Все first/last fp16 массивы совпали побитово; max_abs=0 для B/C/D. Кандидатов B 447406, C 446969, D 444999; индексы пар и полные матрицы признаков совпали точно, pair_feature_equality. Вызовов признаков кадра 1182 вместо 3545, плюс один контрольный; shared.feature_calls.',
        interpretation='Повторная обработка одинаковых кадров убрана; точность модели, порядок и размер обработки рамок каждой ветки сохранены. На этой пробе вход фиксированной сшивки не изменился. Экономия включает также однократную загрузку модели, декодирование видео и чтение массивов.',
        cannot_claim='Полный HOTA здесь не пересчитывался. Результат скорости относится к одному полному видео и только этапу MASA. Не доказано ускорение всего опыта на 64,5%, улучшение HOTA или вклад FlyVis.',
        decision='Все заранее записанные условия выполнены. По прямому условному разрешению автора продолжить фиксированный HOTA с общими признаками MASA в отдельном каталоге; 36 готовых сигнатур сохранить, дедлайн 15.10 07:40 не менять.')
    card['evidence'].append({'kind': 'json', 'ref': source, 'claim': 'Время, побитовое равенство эмбеддингов и точные входы сшивки'})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; text = plan.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — общий MASA прошёл скорость и строгую сверку

*Получено (`z13_shared_masa_probe_v1`).* На полном видео 00067cfb-e535423e отдельные B/C/D: {ru(report['reference_total_seconds'])} с; совместный проход {ru(report['shared']['including_model_load_seconds'])} с; экономия {ru(report['reduction_percent'])}%. Первичный источник `runs/z13_shared_masa_probe_v1_20261015/report.json`, reference_total_seconds, shared.including_model_load_seconds, reduction_percent. Все массивы first/last fp16 побитово одинаковы, max_abs=0 в трёх вариантах. Все индексы пар и матрицы признаков совпали: B 447406, C 446969, D 444999 пар, pair_feature_equality. Признаки кадра считались 1182 раза вместо 3545, плюс одна контрольная проверка. Тест с CPU сверкой завершён внутри разрешённых 15 минут.

*Вывод.* Неизменный фиксированный считыватель сшивки получает те же входы; его решения в этой пробе сохраняются. HOTA отдельно не пересчитывалась. Это техническое ускорение MASA, не улучшение FlyVis; выигрыш включает однократную загрузку/чтение/декодирование и не переносится на весь HOTA. На следующих видео предусмотрена отдельная проверка равенства признаков первого общего кадра, ошибка прерывает расчёт.

*Решение.* Условное разрешение автора на переход выполнено; продолжить без повторного запроса. Готовые сигнатуры сохраняются, модель/пороги/dev35/сшивка неизменны, один GPU-процесс и прежний абсолютный предел 15.10 07:40.

'''
    assert '## Журнал изменений\n\n' in text
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    atomic_json(RUN / 'result_recorded.json', {'at': now.isoformat(), 'id': card['id'], 'all_gates_passed': True})
    print(json.dumps({'recorded': card['id'], 'reduction_percent': report['reduction_percent']}))


if __name__ == '__main__':
    main()
