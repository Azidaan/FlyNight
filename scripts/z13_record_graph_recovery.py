"""Append the graph preservation failure and exact recovery to the journal."""
from datetime import datetime
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_shared_masa_hota_v1_20261015/full'
proof = json.loads((RUN / 'graph_recovery.json').read_text(encoding='utf-8'))
archive = ROOT / proof['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == proof['sha256']
assert proof['exact_old_hash_recovered']
current = ROOT / proof['current_result_note']
assert current.is_file() and 'completed_negative' in current.read_text(encoding='utf-8')
now = datetime.now().astimezone()
plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
text = plan.read_text(encoding='utf-8')
entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — исправлена защита прежних заметок при пересборке графа

*Получено.* После записи итогового HOTA обёртка `scripts/z13_build_vault_preserving_edits.py` обнаружила нарушение своей проверки сохранения: она защищала локально изменённые заметки от записи, но не от удаления устаревшего пути генератором. При переходе карточки из «идёт» в итоговый статус генератор переместил `fly_spatial_hota_policy_z13_v1` из блока 09 в блок 01. Прежний текст точно восстановлен официальным генератором по прежней версии карточки; совпадение со старым заранее сохранённым SHA256 побитовое, `runs/z13_shared_masa_hota_v1_20261015/full/graph_recovery.json`. Архив: `{proof['archive']}`. Новая итоговая заметка — `{proof['current_result_note']}`. В обёртку добавлена защита `Path.unlink` для локально изменённых файлов; научный код, метрики и результаты не менялись.

*Решение.* Сохранён точный архив старой заметки, а текущий итог остаётся на новом пути, созданном генератором. Пересборку повторить с исправленной обёрткой и проверкой сохранности остальных локальных изменений. GPU и HOTA повторно не запускать. Каталог уже согласован с первичными числами, 58 тестов пройдены. Git по-прежнему блокирует `.git/index.lock`; блокировка не удалялась.

'''
assert '## Журнал изменений\n\n' in text
plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
print('Exact old generated note archived; journal records recovery and wrapper fix.')
