"""Проверка путей и hash до любого обращения к диску или Git.

Путь считается допустимым, только если после Path.resolve() он лежит
внутри корня проекта по Path.relative_to — сравнение строк через
startswith пропускало бы E:\\Dissertation_fake.
"""
from __future__ import annotations

import re
from pathlib import Path

HASH_RE = re.compile(r'^[0-9a-f]{7,40}$')


def looks_like_hash(text: str) -> bool:
    return bool(text) and HASH_RE.match(text) is not None


def resolve_inside(root: Path, rel: str) -> Path | None:
    """Абсолютный путь внутри root или None, если путь выходит наружу."""
    if not rel or '\x00' in rel:
        return None
    try:
        base = Path(root).resolve()
        candidate = Path(rel)
        target = (candidate if candidate.is_absolute() else base / candidate).resolve()
        target.relative_to(base)
    except (OSError, ValueError, RuntimeError):
        return None
    return target


def to_rel(root: Path, path: Path) -> str:
    """Путь относительно корня с прямыми слэшами — как в JSON и в интерфейсе."""
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()
