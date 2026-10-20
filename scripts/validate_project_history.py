"""Проверка каталога истории проекта и словаря метрик.

Запуск: python scripts/validate_project_history.py [--strict]

Печатает ошибки (нарушают правила AGENTS.md — карточка не принимается) и
предупреждения (источник не найден, число не совпало с артефактом).
Код возврата 1 при ошибках; с --strict — и при предупреждениях.
Ничего не меняет: только чтение каталога, словаря, плана и Git.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.project_history.index import build_index  # noqa: E402
from scripts.obsidian_confirmed_results import collect  # noqa: E402
import json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--strict', action='store_true', help='предупреждения тоже считать ошибкой')
    args = ap.parse_args()

    idx = build_index(PROJECT_ROOT, cache_dir=None, write_cache=False)
    confirmed_error=None
    try:
        confirmed=collect(PROJECT_ROOT,json.loads((PROJECT_ROOT/'experiments/project_history.json').read_text(encoding='utf-8')),
                          json.loads((PROJECT_ROOT/'experiments/metrics_glossary.json').read_text(encoding='utf-8')))
        print(f'подтверждённых сравнений в подборке: {len(confirmed)}')
    except (ValueError,KeyError,TypeError,OSError) as exc:
        confirmed_error=str(exc)
        print('ОШИБКА подборки подтверждённых результатов:',confirmed_error)
    print(f'направлений {len(idx.areas)}, экспериментов {len(idx.experiments)}, '
          f'коммитов {len(idx.commits)}, событий {len(idx.timeline)}, '
          f'метрик в словаре {len(idx.glossary.get("metrics", {}))}')
    for e in idx.experiments:
        n_unc = sum(1 for f in ('expected', 'done', 'obtained', 'decision') if not getattr(e, f))
        n_ev = sum(1 for v in e.evidence if v.exists)
        n_missing = sum(1 for v in e.evidence if v.exists is False)
        n_nodesc = sum(1 for m in e.metrics if not m.described)
        flags = []
        if n_unc:
            flags.append(f'не подтверждено полей: {n_unc}')
        if n_missing:
            flags.append(f'источников не найдено: {n_missing}')
        if n_nodesc:
            flags.append(f'метрик без описания: {n_nodesc}')
        if e.missing_commits:
            flags.append(f'коммитов нет в истории: {", ".join(e.missing_commits)}')
        print(f'  [{e.state:19}] {e.id:28} метрик {len(e.metrics):2}  доказательств {n_ev:2}'
              + (f'  · {"; ".join(flags)}' if flags else ''))
    if idx.errors:
        print(f'\nОШИБКИ ({len(idx.errors)}):')
        for x in idx.errors:
            print('  ✗', x)
    if idx.warnings:
        print(f'\nПРЕДУПРЕЖДЕНИЯ ({len(idx.warnings)}):')
        for x in idx.warnings:
            print('  !', x)
    bad = bool(idx.errors) or confirmed_error is not None or (args.strict and bool(idx.warnings))
    print('\n' + ('FAIL: каталог не принят' if bad else 'OK: каталог согласован с источниками'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
