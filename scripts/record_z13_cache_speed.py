"""Record the measured speed table and limitations; no HOTA inference."""
from datetime import datetime
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_cache_speed_v1_20261014'


def main():
    report = json.loads((RUN / 'report.json').read_text(encoding='utf-8'))
    now = datetime.now().astimezone()
    path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    card = next(c for c in catalog['experiments'] if c['id'] == 'z13_cached_params_speed_v1')
    assert card['state'] == 'running'
    card['done'] = now.strftime('%Y-%m-%d')
    complete = report.get('passed', False)
    card['state'] = 'infrastructure' if complete else 'inconclusive'
    lines = []
    table = '| вариант | Triton, с | Triton + кэш, с | сокращение времени |\n|---|---:|---:|---:|\n'
    for arm in ('B', 'D'):
        t = report.get('timings', {}).get(arm, {})
        if all(k in t for k in ('triton', 'triton_cached')):
            old, new = t['triton']['processing_seconds'], t['triton_cached']['processing_seconds']
            reduction = 100 * (1 - new / old)
            f = lambda x: f'{x:.3f}'.replace('.', ',')
            table += f'| {arm} | {f(old)} | {f(new)} | {f(reduction)}% |\n'
            lines.append(f"{arm}: {f(old)} → {f(new)} с, сокращение {f(reduction)}%; первичные {t['triton']['source']} и {t['triton_cached']['source']}")
    cpu = report.get('npz_cpu', {})
    if cpu:
        lines.append('CPU-подготовка заданий MASA: ' + json.dumps(cpu, ensure_ascii=False) + '; источник runs/z13_cache_speed_v1_20261014/report.json:npz_cpu')
    card['obtained'] = 'runs/z13_cache_speed_v1_20261014/report.json: ' + '; '.join(lines)
    card['interpretation'] = ('Парный технический замер одного видео завершён. B не содержит этой правки; его изменение времени показывает колебания замера. Ускорение D, если есть, не подтверждено на всём наборе.' if complete
                              else 'Проба не завершена полностью; доступные измерения сохранены. Ускорение и совпадение неподсчитанных выходов не заявляются.')
    card['decision'] = 'Возобновить прежний проверенный Triton HOTA, без вставки новой оптимизации в его зафиксированный код. Новая оптимизация остаётся отдельной; предел до 15.10 08:46:23 MSK не продлевается.'
    card['acceptance'][0].update(met=True if complete else None,
         evidence='runs/z13_cache_speed_v1_20261014/report.json:sequence/timings, совпадение номеров, счётчиков событий и координат в допуске')
    card['acceptance'][1].update(met=report.get('D_faster') if complete else None,
         evidence='runs/z13_cache_speed_v1_20261014/report.json:timings.D.reduction_percent, один полный парный прогон')
    card['evidence'].append({'kind': 'json', 'ref': 'runs/z13_cache_speed_v1_20261014/report.json', 'claim': 'Фактическое время одного видео и отдельной CPU-подготовки; ограничения замера сохранены'})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — таблица времени однократной подготовки параметров

*Получено (`z13_cached_params_speed_v1`).* `runs/z13_cache_speed_v1_20261014/report.json`: {'проба завершена' if complete else 'неполная проба'}. Одинаковое видео, веса и пороги; в обеих колонках уже используется Triton, новая переменная — кэш неизменных параметров FlyVis. Время соответствует прежней таблице (обработка после создания модели); полное внешнее время также сохранено в JSON.

{table}

Источники чисел: {card['obtained']}

*Вывод.* {card['interpretation']} CPU-чтение архива измерено отдельно; его процент не переносится на всё время MASA. Первый служебный запуск был отклонён автоматической проверкой из-за отсутствия доказательства освобождения GPU; затем проверены paused, завершение всех процессов собственного дерева и свободный мьютекс, `stop_verified.json`, и запуск разрешён. Первая попытка измерения достигла лимита вариантов компилятора при загрузке следующей модели в том же процессе; ошибка и готовый замер B сохранены в `report.attempt1.json`. Перед новой моделью добавлена очистка внутренних графов компилятора, как при отдельном процессе на видео; веса, входы и допуски не менялись. Счётчик бюджета уточнён: суммарно до 20 минут существования двух вычислительных процессов, включая CPU внутри них, исключая подготовку и подтверждённую паузу между попытками. Первые 235,092 с + не более 964,908 с второй попытки; новый предел 15.10 00:21:50 MSK, `active_budget.json`; прежние записи срока сохранены в `*.before_active_accounting.json`. Общий предел HOTA 08:46 не менялся. HOTA не считалась.

*Решение.* {card['decision']} Измеренного потолка времени нет, долю не считаем; изменения HOTA неизвестны.

'''
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(table)


if __name__ == '__main__':
    main()
