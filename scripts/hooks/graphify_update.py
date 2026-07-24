"""PostToolUse-хук: инкрементальное обновление графа graphify после правки кода.

Вызывается Claude Code после каждого Write/Edit (exec-форма, stdin — JSON
с tool_input). Обновляет граф только когда правился .py-файл проекта вне
служебных директорий. Дедупликация конкурентных запусков — lock-файл.

Работает на stdlib; интерпретатор — python из tools/graphify/.venv.
Лог срабатываний: graphify-out/update_hook.log.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r'E:\Dissertation')
GRAPHIFY = PROJECT_ROOT / 'tools' / 'graphify' / '.venv' / 'Scripts' / 'graphify.exe'
LOCK = PROJECT_ROOT / 'graphify-out' / '.update.lock'
LOG = PROJECT_ROOT / 'graphify-out' / 'update_hook.log'
LOCK_STALE_SEC = 300

# Директории, правки в которых граф не касаются (совпадает с .graphifyignore).
EXCLUDED = {'data', 'ods_probe', 'weights', 'runs', 'wheelhouse', 'installers',
            'python310', 'third_party', 'tools', '.venv', '.cache', '.claude',
            '.vscode', 'ui', 'graphify-out', 'figs', 'experiments', 'notes'}


def log(msg: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')
    except OSError:
        pass


def main() -> int:
    try:
        # stdin читаем байтами: PowerShell-пайпы добавляют BOM, utf-8-sig
        # снимает его; чистый UTF-8 от харнесса декодируется так же.
        raw = sys.stdin.buffer.read()
        payload = json.loads(raw.decode('utf-8-sig'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        log('failed to parse stdin JSON')
        return 0
    tool_input = payload.get('tool_input') or {}
    tool_response = payload.get('tool_response') or {}
    fp = tool_input.get('file_path') or tool_response.get('filePath') or ''
    if not fp.lower().endswith('.py'):
        return 0
    try:
        rel = Path(fp).resolve().relative_to(PROJECT_ROOT)
    except (ValueError, OSError):
        return 0  # файл вне проекта
    if rel.parts and rel.parts[0].lower() in EXCLUDED:
        return 0

    # Дедупликация: если lock свежий — другой апдейт уже идёт/только что был.
    try:
        if LOCK.exists() and time.time() - LOCK.stat().st_mtime < LOCK_STALE_SEC:
            log(f'skip (lock) {rel}')
            return 0
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(str(os.getpid()), encoding='utf-8')
    except OSError:
        return 0

    try:
        t0 = time.time()
        proc = subprocess.run(
            [str(GRAPHIFY), '.', '--update', '--no-label', '--no-viz'],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=240)
        tail = (proc.stdout or '').strip().splitlines()[-1:] or ['(no output)']
        log(f'update after {rel}: rc={proc.returncode} {time.time()-t0:.1f}s {tail[0]}')
    except subprocess.TimeoutExpired:
        log(f'update after {rel}: TIMEOUT')
    finally:
        try:
            LOCK.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
