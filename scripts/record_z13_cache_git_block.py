"""Preserve the concrete Git blocker without removing a shared index lock."""
from datetime import datetime
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
now = datetime.now().astimezone()
lock = ROOT / '.git/index.lock'
assert lock.is_file()
proof = {'at': now.isoformat(), 'commit_created': False, 'push_done': False,
         'reason': 'git add failed: .git/index.lock already exists',
         'lock_size': lock.stat().st_size, 'lock_removed': False,
         'hota_running_unaffected': True}
out = ROOT / 'runs/z13_cache_speed_v1_20261014/git_record_status.json'
out.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
text = plan.read_text(encoding='utf-8')
entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — запись оптимизации сохранена, Git заблокирован

Каталог согласован с источниками, 58 тестов пройдены; граф пересобран официальным генератором. Попытка добавить только собственные файлы в Git не прошла из-за существующей `.git/index.lock`. Блокировка не удалялась, чужие процессы не останавливались. Доказательство — `runs/z13_cache_speed_v1_20261014/git_record_status.json`. Новый коммит и push не состоялись; файлы результатов и журнал сохранены локально. HOTA продолжает работать на исходном Triton с пределом 07:40.

'''
plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
print(json.dumps(proof, ensure_ascii=False))
