"""Journal the read-only overlap calculation; no new GPU experiment or plan edit."""
from datetime import datetime
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
proof = json.loads((ROOT / 'runs/z13_masa_overlap_readonly_20261015.json').read_text(encoding='utf-8'))
now = datetime.now().astimezone()
plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
text = plan.read_text(encoding='utf-8')
entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — чтение сохранённых запросов MASA для ответа о скорости

По просьбе автора выяснить самые долгие этапы прочитаны текущий status.json, price.json и код сборки MASA. Основной прогон продолжается; GPU для этой справки не запускался. `runs/z13_masa_overlap_readonly_20261015.json`: на одном существующем видео из 1206 кадров B требует признаки 1182 кадров, D — 1181; пересечение 1181, объединение 1182. Раздельные вычисления признаков всего кадра дают 2363 вызова, совместное чтение потребовало бы 1182: устранимо около 50% таких вызовов для этой пары. Это счётчик вызовов, не измеренная экономия времени; C не проверен.

Код `tracking/masa_r50.py:MasaR50.features` не получает рамки; рамки используются в `embed_boxes`. Поэтому общий расчёт признаков кадра для B/C/D — технический кандидат, сохраняющий смысл алгоритма, но совпадение выходов и фактическое время нужно проверить перед применением. Реализация конвейера не менялась, новый опыт и пересмотр научного плана не запускались. Текущий крупный этап — 105 выходов трекера; следующая оценка MASA около 2,8 часа из `runs/z13_spatial_hota_v1_20261014/price.json` остаётся грубой экстраполяцией одного видео.

'''
plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
print('Read-only finding recorded; no pipeline changes')
