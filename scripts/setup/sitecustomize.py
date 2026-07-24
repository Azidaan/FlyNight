"""Локализация кэшей внутри папки проекта.

Модуль site автоматически импортирует sitecustomize при старте интерпретатора,
поэтому переменные действуют для любого запуска из .venv — pip, скрипты, IDE,
без необходимости активировать окружение.

Мастер-копия: scripts/setup/sitecustomize.py.
Рабочая копия: .venv/Lib/site-packages/sitecustomize.py.
При пересоздании .venv скопировать мастер-копию заново
(см. notes/environment.md, процедура восстановления).
"""
import os
import sys
from pathlib import Path

# Принудительный UTF-8 для вывода: без этого кириллица в логах превращается
# в кракозябры при захвате вывода (консоль cp866/cp1251 vs UTF-8-исходники).
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8')
        except (ValueError, OSError):
            pass
os.environ.setdefault('PYTHONUTF8', '1')

# .venv/Lib/site-packages/sitecustomize.py -> корень проекта на 3 уровня выше.
# Для мастер-копии (scripts/setup/) корень на 2 уровня выше — берём ту папку,
# в которой лежит .venv или сводный план.
_here = Path(__file__).resolve()
_root = None
for _parent in _here.parents:
    if (_parent / 'СВОДНЫЙ_ПЛАН1.md').exists() or (_parent / '.venv').exists():
        _root = _parent
        break
if _root is None:
    _root = _here.parents[3] if len(_here.parents) > 3 else _here.parents[-1]

_cache = _root / '.cache'

os.environ.setdefault('PIP_CACHE_DIR', str(_cache / 'pip'))
os.environ.setdefault('TORCH_HOME', str(_cache / 'torch'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(_cache / 'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(_cache / 'matplotlib'))
