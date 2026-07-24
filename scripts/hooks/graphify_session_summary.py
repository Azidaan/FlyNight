"""SessionStart-хук: компактная сводка графа graphify в контекст сессии.

Выводит JSON с additionalContext (~200 токенов): размер графа, свежесть
относительно .py-файлов, топ-узлы по степени и напоминание об инструменте.
Полный graph.json (сотни КБ) в контекст НЕ загружается — вся экономия
именно в том, что ориентировка идёт через `graphify query`, а не чтение.
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r'E:\Dissertation')
GRAPH = PROJECT_ROOT / 'graphify-out' / 'graph.json'
CODE_DIRS = ('datasets', 'training', 'evaluation', 'tracking', 'models',
             'scripts', 'tests', 'prototype', 'visualization', 'utils')


def emit(text: str) -> None:
    # Windows-консоль по умолчанию cp866/cp1251 — харнесс ждёт UTF-8.
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps({
        'hookSpecificOutput': {
            'hookEventName': 'SessionStart',
            'additionalContext': text,
        }
    }, ensure_ascii=False))


def main() -> int:
    if not GRAPH.exists():
        emit('[graphify] Граф кода не построен. Построить: '
             r'E:\Dissertation\tools\graphify\.venv\Scripts\graphify.exe . --no-label')
        return 0

    graph_mtime = GRAPH.stat().st_mtime
    try:
        data = json.loads(GRAPH.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        emit('[graphify] graph.json повреждён — пересобрать: graphify . --no-label')
        return 0

    nodes = data.get('nodes', [])
    edges = data.get('edges', data.get('links', []))

    # Степени узлов — топ-5 «god nodes».
    deg: dict[str, int] = {}
    for e in edges:
        for k in ('source', 'target'):
            v = e.get(k)
            if isinstance(v, (str, int)):
                deg[str(v)] = deg.get(str(v), 0) + 1
    id2label = {str(n.get('id')): (n.get('label') or n.get('id')) for n in nodes}
    top = sorted(deg.items(), key=lambda x: -x[1])[:5]
    top_str = ', '.join(f'{id2label.get(i, i)}({d})' for i, d in top)

    # Свежесть: есть ли .py новее графа.
    stale = 0
    for d in CODE_DIRS:
        p = PROJECT_ROOT / d
        if not p.is_dir():
            continue
        for f in p.rglob('*.py'):
            try:
                if f.stat().st_mtime > graph_mtime:
                    stale += 1
            except OSError:
                pass
    age_h = (time.time() - graph_mtime) / 3600
    stale_note = (f' ВНИМАНИЕ: {stale} .py-файлов новее графа — обнови '
                  f'(graphify . --update --no-label --no-viz).' if stale else ' Граф актуален.')

    emit(f'[graphify] Граф кода: {len(nodes)} узлов, {len(edges)} рёбер '
         f'(обновлён {age_h:.1f} ч назад).{stale_note} '
         f'Центральные узлы: {top_str}. '
         f'Для разведки структуры СНАЧАЛА: '
         r'E:\Dissertation\tools\graphify\.venv\Scripts\graphify.exe'
         f' query "..." --budget 800 (или path/explain); '
         f'факты для правок кода перепроверять Grep/Read.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
