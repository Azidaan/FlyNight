"""Append the verified device failure; do not alter frozen scripts or outcomes."""
from pathlib import Path
from datetime import datetime
import json

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/z13_distribution_quality_20261015'
marker=out/'failure_recorded.json'
assert not marker.exists()
verification=json.loads((out/'failure_verification.json').read_text(encoding='utf-8'))
assert len(verification['completed'])==6 and verification['sources_unchanged']
now=datetime.now().astimezone()
plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
text=plan.read_text(encoding='utf-8')
anchor='## Журнал изменений\n\n'
assert text.count(anchor)==1
entry=f'''**{now:%d.%m.%Y, %H:%M} MSK** — техническая остановка оценки качества и очереди новых узлов

*Получено.* `runs/z13_distribution_quality_20261015/stderr.log`: после сохранения всех шести заданий B/C/D fold0/fold1 вызов `quality` завершился на первой операции `torch.randint`: генератор CPU несовместим с унаследованным устройством CUDA. Ошибка до первого обновления функции качества. Первичные веса и предсказания проверены на процессоре: `runs/z13_distribution_quality_20261015/failure_verification.json`, все шесть контрольных точек имеют шаг 500, хеши предсказаний совпадают, исходники и данные неизменны. Это ошибка интеграции CPU-этапа после GPU, не измерение эффективности.

*Решение.* Оба собственных контроллера и рабочий процесс завершены. Новая очередь остановлена защитой до запуска рабочего процесса: `runs/z13_candidates_20261016/controller_result.json`, код 76, `worker_started=false`. Не подменять код завершения успешным и не обходить условие очереди. Сохранённое обучение повторять не требуется. Точка технического исправления — явное устройство CPU для оценки качества; замороженные источники не изменены. Для продолжения нужен отдельный проверенный способ возобновления с сохранением истории сбоя. Новые узлы не проверены, их эффективность неизвестна. HOTA не измерялась, процент к стенду не определён.

'''
plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
marker.write_text(json.dumps({'recorded':now.isoformat(),'verification':'failure_verification.json'},indent=2),encoding='utf-8')
print('Verified failure appended without modifying frozen sources')
