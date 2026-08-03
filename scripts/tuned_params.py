"""Чтение подобранных параметров из артефактов предыдущих шагов (Этапы 8-9).

Смысл модуля — чтобы найденные оптимумы действительно применялись, а не
оставались в таблице. Раньше сеточный поиск честно находил лучший порог,
а сравнение режимов запускалось с зашитым 0.5; сравнение с ненастроенным
ByteTrack не защищается на слушании.

Симметрия настройки. Порог τ для standard подбирается на val сеточным
поиском, поэтому и коэффициент k для LQA-режима подбирается там же
(scripts/sweep_lqa_k.py). Иначе сравнение было бы перекошено: одному
методу дали лучшие параметры, другому — значение по умолчанию.

Все функции возвращают None, если артефакта ещё нет: вызывающая сторона
сама решает, падать или брать значение по умолчанию.
"""
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / 'experiments' / 'results'
NOTES = PROJECT_ROOT / 'notes'

DEFAULT_TAU = 0.5          # план, п. 9.4 — если сетка ещё не считана
DEFAULT_BUFFER = 30        # значение ByteTrack по умолчанию
DEFAULT_K = 0.2            # план, п. 9.3


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_tracker_params(subset: str,
                        root: Path = PROJECT_ROOT) -> tuple[float, int] | None:
    """(τ, buffer) с лучшим HOTA для поднабора — из сеточного поиска Этапа 8."""
    rows = [r for r in _rows(root / 'experiments' / 'results' /
                             'tracking_param_search.csv')
            if r.get('subset') == subset and _f(r.get('HOTA')) is not None]
    if not rows:
        return None
    best = max(rows, key=lambda r: _f(r['HOTA']))
    tau, buf = _f(best.get('track_high_thresh')), _f(best.get('track_buffer'))
    if tau is None or buf is None:
        return None
    return tau, int(buf)


def brightness_coefficients(root: Path = PROJECT_ROOT) -> tuple[float, float] | None:
    """(a, b) формулы τ = a·brightness + b — из регрессии Этапа 8."""
    rows = _rows(root / 'experiments' / 'results' / 'brightness_coefficients.csv')
    if rows:
        a, b = _f(rows[-1].get('a')), _f(rows[-1].get('b'))
        if a is not None and b is not None:
            return a, b
    # запасной путь: разобрать заметку, если CSV писался старой версией скрипта
    md = root / 'notes' / 'brightness_to_threshold.md'
    if md.exists():
        m = re.search(r'tau_base\(видео\)\s*=\s*(-?[\d.]+)\s*\*\s*brightness\s*\+\s*(-?[\d.]+)',
                      md.read_text(encoding='utf-8', errors='replace'))
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def best_lqa_k(root: Path = PROJECT_ROOT) -> float | None:
    """k с лучшим средним HOTA по поднаборам — из свипа scripts/sweep_lqa_k.py."""
    rows = _rows(root / 'experiments' / 'results' / 'lqa_k_sweep.csv')
    by_k: dict[float, list[float]] = {}
    for r in rows:
        k, hota = _f(r.get('k')), _f(r.get('HOTA'))
        if k is not None and hota is not None:
            by_k.setdefault(k, []).append(hota)
    if not by_k:
        return None
    return max(by_k, key=lambda k: sum(by_k[k]) / len(by_k[k]))


def resolve_tau(subset: str, override: float | None = None,
                root: Path = PROJECT_ROOT) -> tuple[float, int, str]:
    """(τ, buffer, откуда) — явное значение важнее подобранного, то важнее умолчания."""
    if override is not None:
        return override, DEFAULT_BUFFER, 'задан вручную'
    found = best_tracker_params(subset, root)
    if found is not None:
        return found[0], found[1], 'оптимум сетки Этапа 8'
    return DEFAULT_TAU, DEFAULT_BUFFER, 'умолчание (сетка не считана)'


def resolve_k(override: float | None = None,
              root: Path = PROJECT_ROOT) -> tuple[float, str]:
    """(k, откуда) для LQA-адаптации."""
    if override is not None:
        return override, 'задан вручную'
    found = best_lqa_k(root)
    if found is not None:
        return found, 'оптимум свипа k'
    return DEFAULT_K, 'умолчание (свип не считан)'
