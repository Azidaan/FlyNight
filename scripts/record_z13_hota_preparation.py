"""Record the integration probe and measured preparation price; no HOTA verdict."""
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def main():
    probe = json.loads((RUN / 'technical/report.json').read_text(encoding='utf-8'))
    masa = json.loads((RUN / 'technical/masa_price/report.json').read_text(encoding='utf-8'))
    assert probe['passed'] and masa['passed']
    signature_hours = probe['active_D_seconds'] * 105 / 3600
    masa_hours = masa['seconds_including_model_load'] * 105 / 3600
    timing = {'steps_per_frame': 2, 'dt': 1 / 60, 'source_rate_for_neurons': 30,
              'reason': 'Match saved training; measured video rates vary around 30 Hz. Retina and original environment keep measured video rate.',
              'status': 'clarified before full run and before any HOTA',
              'probe_note': 'Streaming parity probe already used exactly 30 Hz. No-intervention output cannot depend on neural correction values; timing measurement remains a rough price estimate.'}
    (RUN / 'timing_contract.json').write_text(json.dumps(timing, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    price = {'source': 'technical/report.json + technical/masa_price/report.json',
             'signature_hours_if_all_arms_like_D_video': signature_hours,
             'masa_hours_if_all_videos_like_probe': masa_hours,
             'subtotal_hours': signature_hours + masa_hours,
             'proposed_wall_budget_hours': 10,
             'caveat': 'One video; differing object counts and B speed, process startup and CPU HOTA cost can change total. Not a guaranteed upper bound.',
             'full_run_started': False}
    (RUN / 'price.json').write_text(json.dumps(price, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    card = next(c for c in catalog['experiments'] if c['id'] == 'fly_spatial_hota_policy_z13_v1')
    assert card['state'] == 'planned'
    card['state'] = 'ready'
    card['acceptance'][0]['met'] = True
    card['acceptance'][0]['evidence'] = 'runs/z13_spatial_hota_v1_20261014/technical/report.json: all models loaded, streaming parity <=1e-5, no-intervention outputs exactly equal. CPU contract tests cover center-only/lost-only/chunking/memory.'
    card['evidence'].extend([
        {'kind': 'json', 'ref': 'runs/z13_spatial_hota_v1_20261014/technical/report.json', 'claim': 'Technical probe, no HOTA or annotation reading'},
        {'kind': 'json', 'ref': 'runs/z13_spatial_hota_v1_20261014/technical/masa_price/report.json', 'claim': 'One-video MASA inference price'},
        {'kind': 'json', 'ref': 'runs/z13_spatial_hota_v1_20261014/price.json', 'claim': 'Arithmetic workload estimate, not guaranteed bound'},
        {'kind': 'json', 'ref': 'runs/z13_spatial_hota_v1_20261014/timing_contract.json', 'claim': 'Neuron timing matches training; environment kept unchanged'}])
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    now = datetime.now().astimezone()
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — подготовка HOTA З13 v2: подключение проверено, цена измерена

*Получено (`fly_spatial_hota_policy_z13_v1`).* `runs/z13_spatial_hota_v1_20261014/technical/report.json`: все B/C/D загружаются; последовательное считывание совпадает с обучающим (максимум различия вероятностей 0 / 1,788e−7 / 1,788e−7). На одном dev-видео `00067cfb-e535423e` без вмешательств все выходы трекера, жизни и внешность точно совпали со стендом, вход глаза — с допуском 1e−3. Прототип D с включёнными поправками: 15592 шага глаза, 55 вмешательств, 0 нечисловых отказов; 217,52 с на видео. Это исполнение и цена, HOTA не считалась. Первая проба закончилась ошибкой смешения CPU/CUDA в проверке геометрии; размещение проверочных тензоров и оценщика качества сделано явным, прежний статус и журнал ошибки сохранены. Также до полного прогона явно выровнен шаг нейронов с обучением (30 Гц, 2 шага по 1/60 с); сетчатка и окружение используют прежнюю измеренную частоту видео, `timing_contract.json`. Проверка совпадения считывания уже выполнялась при 30 Гц.

*Цена.* `technical/masa_price/report.json`: MASA одного видео с загрузкой модели 94,80 с. При том же темпе для всех 105 видео-прогонов: сигнатуры {signature_hours:.2f} ч, MASA {masa_hours:.2f} ч, сумма {signature_hours + masa_hours:.2f} ч, без части запуска процессов и CPU-оценки HOTA. Это арифметический прогноз по одному видео, не верхняя граница; B может быть быстрее, другие видео тяжелее. `price.json` предлагает ограниченное окно 10 ч с сохранением завершённых видео и возможностью продолжения.

*Решение.* Подготовлены отдельное подключение, последовательный контроллер и оценщик с прежней сшивкой база+MASA. Неиспользуемый DINO не пересчитывается: совместимые нулевые столбцы общим читателем вычисляются, но не входят в фиксированную голову base_masa. Карточка `ready`; полный запуск требует подтверждения измеренного бюджета. Весов/порогов по HOTA не меняли. Исходный отрицательный итог 90%/50 сохранён, HOTA ещё не измерялась; процент к стенду неизвестен.

'''
    # Russian journal decimal convention.
    entry = entry.replace(f'{signature_hours:.2f}', f'{signature_hours:.2f}'.replace('.', ','))
    entry = entry.replace(f'{masa_hours:.2f}', f'{masa_hours:.2f}'.replace('.', ','))
    entry = entry.replace(f'{signature_hours + masa_hours:.2f}', f'{signature_hours + masa_hours:.2f}'.replace('.', ','))
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps(price))


if __name__ == '__main__':
    main()
