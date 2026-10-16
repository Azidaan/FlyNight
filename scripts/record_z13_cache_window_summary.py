"""Record actual probe process windows and restoration of the approved HOTA."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
rows = []
for relative in ('runs/z13_cache_balanced_v1_20261015',
                 'runs/z13_cache_followup_c_v1_20261015', 'runs/z13_cache_clip_v1_20261015'):
    directory = ROOT / relative
    launch = json.loads((directory / 'launch_process.json').read_text(encoding='utf-8'))
    result = json.loads(max(directory.glob('deadline_guard_*/result.json'), key=lambda p:p.stat().st_mtime).read_text(encoding='utf-8'))
    start = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(launch['launcher_start_filetime']) / 10)
    ended = datetime.fromisoformat(result['ended'])
    rows.append({'run': relative, 'from_creation_to_guard_result_seconds': (ended - start).total_seconds(),
                 'start': start.isoformat(), 'guard_result': result['ended'], 'state': result['state']})
total = sum(r['from_creation_to_guard_result_seconds'] for r in rows)
proof = {'probe_windows': rows, 'total_seconds': total,
         'scope': 'Process creation to guard completion, including CPU setup inside each process; not hardware active-kernel time.',
         'hota_resume_source': 'runs/z13_cache_clip_v1_20261015/resume_verified.json'}
path = ROOT / 'runs/z13_cache_balanced_v1_20261015/window_summary.json'
path.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
now = datetime.now().astimezone()
plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; text = plan.read_text(encoding='utf-8')
entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — три пробы завершены, бюджет и продолжение проверены

Все три измерительных процесса завершены штатно, независимо ограничены по времени; `runs/z13_cache_balanced_v1_20261015/window_summary.json`: суммарно {total:.3f} с от создания каждого процесса до результата его ограничителя, включая CPU внутри процессов. Это не время активных ядер GPU. Между пробами шла подготовка кода на CPU. Исходный HOTA возобновлён, независимый ограничитель `runs/z13_triton_hota_v1_20261014/full/deadline_guard_20261015_010150/started.json` вооружён на 07:40. Отдельный код с кэшем готов, но не запущен; синхронизация готового B — `scripts/z13_cached_reuse_b.py`, 21 результат сохранён в подготовленном каталоге по `reuse_proof.json`. Каталог согласован, 58 тестов пройдены, граф пересобран. Git по-прежнему заблокирован существующей `.git/index.lock`; блокировка не удалялась, результаты сохранены локально.

'''
plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
print(json.dumps(proof, ensure_ascii=False))
