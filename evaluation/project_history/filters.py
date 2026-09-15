"""Фильтрация записей истории: все условия соединяются через И.

Работает со словарями (результат to_dict), чтобы одна и та же функция
годилась для экспериментов, коммитов и событий. Значения для чипов
строятся из данных (facets), а не из зашитого списка.
"""
from __future__ import annotations

from typing import Any

TEXT_FIELDS = ('id', 'title', 'summary', 'question', 'hypothesis', 'expected',
               'done', 'obtained', 'interpretation', 'decision', 'subject', 'body',
               'text', 'author', 'hash', 'short', 'detector', 'tracker', 'dataset',
               'split', 'baseline', 'variant', 'limitations', 'next_step', 'area',
               'state', 'state_label', 'kind', 'source', 'raw', 'result_label')
LIST_FIELDS = ('tags', 'conditions', 'commits', 'files', 'runs', 'reports',
               'experiments', 'source_kinds', 'seeds')


def _text(rec: dict) -> str:
    parts: list[str] = []
    for k in TEXT_FIELDS:
        v = rec.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    for k in LIST_FIELDS:
        v = rec.get(k)
        if isinstance(v, list):
            parts += [str(x) for x in v]
    for m in rec.get('metrics', []) or []:
        if isinstance(m, dict):
            parts += [str(m.get('key', '')), str(m.get('value', '')), str(m.get('subset', ''))]
    return ' '.join(parts).lower()


def date_span(rec: dict) -> tuple[str, str]:
    """(начало, конец) записи в ISO или ('', '') — если дат нет."""
    start = rec.get('started') or rec.get('when') or rec.get('date') or ''
    end = rec.get('finished') or rec.get('when_end') or start
    return (str(start)[:10], str(end)[:10])


def _as_list(v: Any) -> list[str]:
    if v is None or v == '' or v == []:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if str(x) != '']
    return [str(v)]


def _match_any(values: list[str], wanted: list[str]) -> bool:
    return any(v in wanted for v in values)


def apply(records: list[dict], f: dict | None) -> list[dict]:
    f = f or {}
    q = str(f.get('q', '') or '').strip().lower()
    words = q.split()
    area = _as_list(f.get('area'))
    state = _as_list(f.get('state'))
    result = _as_list(f.get('result'))
    kind = _as_list(f.get('kind'))
    source = _as_list(f.get('source_kind'))
    tag = _as_list(f.get('tag'))
    seed = _as_list(f.get('seed'))
    detector = _as_list(f.get('detector'))
    tracker = _as_list(f.get('tracker'))
    dataset = _as_list(f.get('dataset'))
    condition = _as_list(f.get('condition'))
    date_from = str(f.get('date_from') or '')[:10]
    date_to = str(f.get('date_to') or '')[:10]
    tri = {k: f.get(k) for k in ('has_commit', 'has_metrics', 'has_artifacts', 'has_artifact')}

    out = []
    for r in records:
        if words:
            t = _text(r)
            if not all(w in t for w in words):
                continue
        if area and r.get('area') not in area:
            continue
        if state and r.get('state') not in state:
            continue
        if result and r.get('result') not in result:
            continue
        if kind and r.get('kind') not in kind:
            continue
        if source and not _match_any(_as_list(r.get('source_kinds')), source):
            continue
        if tag and not _match_any(_as_list(r.get('tags')), tag):
            continue
        if seed and not _match_any(_as_list(r.get('seeds')), seed):
            continue
        if detector and r.get('detector') not in detector:
            continue
        if tracker and r.get('tracker') not in tracker:
            continue
        if dataset and r.get('dataset') not in dataset and r.get('split') not in dataset:
            continue
        if condition and not _match_any(_as_list(r.get('conditions')), condition):
            continue
        ok = True
        for k, want in tri.items():
            if want is None or want == '' or k not in r:
                continue
            if bool(r.get(k)) != bool(want):
                ok = False
                break
        if not ok:
            continue
        if date_from or date_to:
            s, e = date_span(r)
            if not s:
                continue
            if date_from and e < date_from:
                continue
            if date_to and s > date_to:
                continue
        out.append(r)
    return out


def facets(records: list[dict]) -> dict[str, list[str]]:
    keys = {
        'area': lambda r: _as_list(r.get('area')),
        'state': lambda r: _as_list(r.get('state')),
        'result': lambda r: _as_list(r.get('result')),
        'kind': lambda r: _as_list(r.get('kind')),
        'source_kind': lambda r: _as_list(r.get('source_kinds')),
        'tag': lambda r: _as_list(r.get('tags')),
        'seed': lambda r: _as_list(r.get('seeds')),
        'detector': lambda r: _as_list(r.get('detector')),
        'tracker': lambda r: _as_list(r.get('tracker')),
        'dataset': lambda r: _as_list(r.get('dataset')) + _as_list(r.get('split')),
        'condition': lambda r: _as_list(r.get('conditions')),
        'author': lambda r: _as_list(r.get('author')),
    }
    out: dict[str, list[str]] = {}
    for name, fn in keys.items():
        vals: set[str] = set()
        for r in records:
            vals.update(fn(r))
        out[name] = sorted(vals)
    return out
