"""Ограниченный обход артефактов: таблицы, JSON, отчёты.

Никакой рекурсии по гигантским каталогам изображений, кэшей детекций и
весов: список корней задан явно, глубина и число файлов ограничены, тела
файлов читаются только для поиска значений по адресу (lookup_value) и
только если файл не больше лимита.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from .safe_paths import resolve_inside, to_rel

# (относительный корень, максимальная глубина от него)
ROOTS: tuple[tuple[str, int], ...] = (
    ('.', 0),                       # отчёты *.md в корне проекта
    ('experiments/results', 1),
    ('experiments', 0),
    ('runs', 2),                    # runs/<эксперимент>/<combined>/summary.json
    ('dissertation', 1),
    ('notes', 1),
)
EXTENSIONS = ('.csv', '.json', '.md')
SKIP_DIRS = {'det_cache_ctl', 'det_cache_ctl_960', 'det_cache_q', 'det_cache_over',
             'weights', 'images', 'labels', 'backup_20260907', '__pycache__',
             'subset_samples', '_archive'}
DEFAULT_MAX_FILES = 3000
DEFAULT_MAX_BYTES = 2_000_000
# Каталог с сотнями однотипных файлов (повидео-выходы трекера) описывается
# одной записью с числом файлов, а не тысячей строк.
DEFAULT_MAX_PER_DIR = 30
DEFAULT_MAX_SUBDIRS = 40
# повидео-выход трекера: <id видео BDD100K>.json — тысячи файлов, не артефакт
RE_PER_VIDEO = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{8}(_[a-z0-9]+)?\.json$')


def scan(root: Path, max_files: int = DEFAULT_MAX_FILES,
         max_bytes: int = DEFAULT_MAX_BYTES,
         max_per_dir: int = DEFAULT_MAX_PER_DIR,
         max_subdirs: int = DEFAULT_MAX_SUBDIRS) -> list[dict]:
    root = Path(root)
    out: list[dict] = []
    seen: set[str] = set()
    for rel, depth in ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        stack = [(base, 0)]
        while stack and len(out) < max_files:
            d, lvl = stack.pop()
            files: list[os.DirEntry] = []
            subdirs: list[os.DirEntry] = []
            try:
                with os.scandir(d) as it:
                    for e in it:
                        try:
                            if e.is_dir(follow_symlinks=False):
                                if (lvl < depth and e.name not in SKIP_DIRS
                                        and not e.name.startswith('.')):
                                    subdirs.append(e)
                            elif (os.path.splitext(e.name)[1].lower() in EXTENSIONS
                                  and not RE_PER_VIDEO.match(e.name)):
                                files.append(e)
                        except OSError:
                            continue
            except OSError:
                continue
            files.sort(key=lambda e: e.name)
            subdirs.sort(key=lambda e: e.name)
            if len(files) > max_per_dir:
                out.append({'path': to_rel(root, d), 'kind': 'dir', 'size': 0,
                            'mtime': 0.0, 'too_big': False, 'count': len(files),
                            'note': f'{len(files)} однотипных файлов, перечислены не все'})
                files = files[:5]
            if len(subdirs) > max_subdirs:
                out.append({'path': to_rel(root, d), 'kind': 'dir', 'size': 0,
                            'mtime': 0.0, 'too_big': False, 'count': len(subdirs),
                            'note': f'{len(subdirs)} подкаталогов, раскрыты не все'})
                subdirs = subdirs[:max_subdirs]
            for e in reversed(subdirs):
                stack.append((Path(e.path), lvl + 1))
            for e in files:
                if len(out) >= max_files:
                    break
                relp = to_rel(root, Path(e.path))
                if relp in seen:
                    continue
                seen.add(relp)
                try:
                    st = e.stat()
                except OSError:
                    continue
                out.append({'path': relp, 'kind': os.path.splitext(e.name)[1].lower()[1:],
                            'size': st.st_size, 'mtime': st.st_mtime,
                            'too_big': st.st_size > max_bytes})
    return out


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().replace('−', '-')
        try:
            if s.lower() in ('true', 'false'):
                return s.lower() == 'true'
            return float(s.replace(',', '.')) if any(ch.isdigit() for ch in s) else s
        except ValueError:
            return s
    return v


def lookup_value(path: Path, locator: str, max_bytes: int = DEFAULT_MAX_BYTES) -> Any:
    """Значение по адресу внутри JSON («a.b.0») или CSV
    («row=1;col=значение» либо «<колонка>=<значение>;col=значение»).
    None — если файла нет, он велик или адрес не найден."""
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
    except OSError:
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == '.json':
            data = json.loads(path.read_text(encoding='utf-8'))
            cur: Any = data
            for part in [p for p in locator.split('.') if p != '']:
                if isinstance(cur, list) and part.lstrip('-').isdigit():
                    cur = cur[int(part)]
                elif isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur
        if suffix == '.csv':
            spec = dict(kv.split('=', 1) for kv in locator.split(';') if '=' in kv)
            col = spec.pop('col', None)
            with open(path, encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            if col is None:
                return None
            if 'row' in spec:
                i = int(spec['row']) - 1
                return _coerce(rows[i].get(col)) if 0 <= i < len(rows) else None
            for r in rows:
                if all(str(r.get(k, '')).strip() == v.strip() for k, v in spec.items()):
                    return _coerce(r.get(col))
            return None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    return None


def value_matches(found: Any, expected: Any, tol: float = 1e-4) -> bool | None:
    """Совпадает ли значение из источника с записанным в каталоге.
    None — сравнить нельзя (не числа)."""
    try:
        a, b = float(found), float(str(expected).replace(',', '.').replace('−', '-'))
    except (TypeError, ValueError):
        return None
    scale = max(1.0, abs(b))
    return abs(a - b) <= tol * scale


def check_metric_source(root: Path, source: str, locator: str, expected: Any) -> tuple[bool | None, str]:
    """(совпало?, пояснение). None — проверить нельзя."""
    p = resolve_inside(root, source) if source else None
    if p is None or not p.exists():
        return None, 'источник не найден'
    if not locator:
        return None, 'адрес значения не указан'
    found = lookup_value(p, locator)
    if found is None:
        return None, f'адрес {locator!r} не найден в {source}'
    ok = value_matches(found, expected)
    if ok is None:
        return None, f'в источнике {found!r}, сравнить нельзя'
    return ok, ('совпадает с источником' if ok else f'в источнике {found!r}, в каталоге {expected!r}')
