"""Кураторский каталог и словарь метрик: загрузка и проверка.

Каталог (experiments/project_history.json) хранит только связи и короткие
проверяемые резюме — «ожидалось», «получено», «решение». Большие тексты и
таблицы не дублируются: на них ссылаются evidence и metrics[].source.

Словарь (experiments/metrics_glossary.json) описывает каждую метрику: что
это, куда лучше, чем считалась. Метрика, которой нет в словаре, — ошибка
каталога: карточка рисуется, но с пометкой «нет описания».

Проверки не роняют загрузку: ошибки и предупреждения собираются в списки,
чтобы окно показало их рядом с данными (10.10).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import (
    DATE_PRECISIONS, EVENT_KINDS, EVIDENCE_KINDS, RESULT_OF_STATE, STATES,
    EvidenceReference, ExperimentRecord, MetricValue, ProjectArea)
from .safe_paths import looks_like_hash, resolve_inside

SCHEMA_VERSION = 1
REQUIRED_EXPERIMENT = ('id', 'title', 'area', 'state')
REQUIRED_METRIC_DESC = ('ru', 'better')
BETTER_VALUES = ('higher', 'lower', 'none')
EMPTY_CATALOG: dict[str, Any] = {
    'schema_version': SCHEMA_VERSION, 'areas': [], 'experiments': [],
    'manual_events': [], 'commit_links': {}}
EMPTY_GLOSSARY: dict[str, Any] = {
    'schema_version': SCHEMA_VERSION, 'metrics': {}, 'aliases': {}, 'terms': {}}


@dataclass
class Glossary:
    metrics: dict[str, dict] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    terms: dict[str, dict] = field(default_factory=dict)

    def resolve(self, key: str) -> str:
        """Каноническое имя метрики по ключу или алиасу (регистр важен для
        канонических имён, алиасы можно писать как угодно)."""
        if key in self.metrics:
            return key
        if key in self.aliases:
            return self.aliases[key]
        low = key.lower()
        for a, canon in self.aliases.items():
            if a.lower() == low:
                return canon
        for canon in self.metrics:
            if canon.lower() == low:
                return canon
        return key

    def describe(self, key: str) -> dict | None:
        return self.metrics.get(self.resolve(key))


@dataclass
class Catalog:
    areas: list[ProjectArea] = field(default_factory=list)
    experiments: list[ExperimentRecord] = field(default_factory=list)
    manual_events: list[dict] = field(default_factory=list)
    commit_links: dict[str, dict] = field(default_factory=dict)
    glossary: Glossary = field(default_factory=Glossary)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    catalog_path: str = ''
    glossary_path: str = ''

    def area_ids(self) -> set[str]:
        return {a.id for a in self.areas}


def _read_json(path: Path, empty: dict, warnings: list[str], what: str) -> dict:
    if not path.exists():
        warnings.append(f'{what}: файл {path.name} не найден — каталог пуст')
        return json.loads(json.dumps(empty))
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        warnings.append(f'{what}: не прочитан ({e})')
        return json.loads(json.dumps(empty))
    if not isinstance(data, dict):
        warnings.append(f'{what}: корень не объект')
        return json.loads(json.dumps(empty))
    v = data.get('schema_version')
    if v != SCHEMA_VERSION:
        warnings.append(f'{what}: schema_version {v!r}, ожидается {SCHEMA_VERSION}')
    return data


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(',', '.').replace('−', '-'))
        except ValueError:
            return None
    return None


def _delta_pct(value: Any, baseline: Any) -> float | None:
    v, b = _num(value), _num(baseline)
    if v is None or b is None or b == 0:
        return None
    return (v - b) / abs(b) * 100.0


def _check_paths(paths: list[str], root: Path, where: str,
                 errors: list[str], warnings: list[str]) -> list[str]:
    out = []
    for p in paths:
        if not isinstance(p, str) or not p:
            errors.append(f'{where}: пустой путь')
            continue
        resolved = resolve_inside(root, p)
        if resolved is None:
            errors.append(f'{where}: путь {p!r} выходит за пределы проекта')
            continue
        if not resolved.exists():
            warnings.append(f'{where}: файл {p} не найден на диске')
        out.append(p)
    return out


def _metric(raw: Any, glossary: Glossary, where: str, errors: list[str],
            warnings: list[str], root: Path) -> MetricValue | None:
    if isinstance(raw, str):
        raw = {'key': raw}
    if not isinstance(raw, dict) or not raw.get('key'):
        errors.append(f'{where}: метрика без ключа: {raw!r}')
        return None
    key = glossary.resolve(str(raw['key']))
    desc = glossary.metrics.get(key)
    m = MetricValue(
        key=key, value=raw.get('value'), baseline=raw.get('baseline'),
        subset=str(raw.get('subset', '')), source=str(raw.get('source', '')),
        locator=str(raw.get('locator', '')), note=str(raw.get('note', '')))
    if desc:
        m.described = True
        m.title = str(desc.get('title', key))
        m.better = str(desc.get('better', ''))
    else:
        errors.append(f'{where}: метрика {raw["key"]!r} не описана в словаре '
                      f'metrics_glossary.json')
        m.title = key
    m.delta_pct = _delta_pct(m.value, m.baseline)
    if m.source:
        r = resolve_inside(root, m.source)
        if r is None:
            errors.append(f'{where}: источник метрики {m.source!r} вне проекта')
            m.source_exists = False
        else:
            m.source_exists = r.exists()
            if not m.source_exists:
                warnings.append(f'{where}: источник метрики {m.source} не найден')
    return m


def _evidence(raw: Any, where: str, errors: list[str], warnings: list[str],
              root: Path) -> EvidenceReference | None:
    if not isinstance(raw, dict) or not raw.get('ref'):
        errors.append(f'{where}: доказательство без ref: {raw!r}')
        return None
    kind = str(raw.get('kind', 'file'))
    if kind not in EVIDENCE_KINDS:
        errors.append(f'{where}: неизвестный тип доказательства {kind!r}')
        kind = 'file'
    ev = EvidenceReference(kind=kind, ref=str(raw['ref']),
                           locator=str(raw.get('locator', '')),
                           claim=str(raw.get('claim', '')),
                           level=str(raw.get('level', 'unconfirmed')))
    if kind == 'commit':
        if not looks_like_hash(ev.ref):
            errors.append(f'{where}: доказательство-коммит с плохим hash {ev.ref!r}')
        ev.exists = None                      # проверит индексатор через git
    else:
        r = resolve_inside(root, ev.ref)
        if r is None:
            errors.append(f'{where}: доказательство {ev.ref!r} вне проекта')
            ev.exists = False
        else:
            ev.exists = r.exists()
            if not ev.exists:
                warnings.append(f'{where}: источник {ev.ref} не найден')
        if ev.exists:
            ev.level = ev.level if ev.level in ('confirmed', 'derived') else 'derived'
        else:
            ev.level = 'unconfirmed'
    return ev


def _experiment(raw: dict, cat: Catalog, root: Path) -> ExperimentRecord | None:
    errors, warnings = cat.errors, cat.warnings
    where = f'эксперимент {raw.get("id", "?")!r}'
    for f in REQUIRED_EXPERIMENT:
        if not raw.get(f):
            errors.append(f'{where}: нет обязательного поля {f!r}')
    if not raw.get('id'):
        return None
    state = str(raw.get('state', 'unknown'))
    if state not in STATES:
        errors.append(f'{where}: недопустимое состояние {state!r}')
        state = 'unknown'
    area = str(raw.get('area', ''))
    if area and area not in cat.area_ids():
        errors.append(f'{where}: направление {area!r} не объявлено в areas')
    e = ExperimentRecord(id=str(raw['id']), title=str(raw.get('title', '')),
                         area=area, state=state)
    for f in ('summary', 'question', 'hypothesis', 'expected', 'done', 'obtained',
              'interpretation', 'decision', 'started', 'finished', 'detector',
              'tracker', 'dataset', 'split', 'checkpoint', 'baseline', 'variant',
              'limitations', 'cannot_claim', 'next_step'):
        v = raw.get(f, '')
        setattr(e, f, '' if v is None else str(v))
    for f in ('tags', 'conditions'):
        v = raw.get(f, [])
        setattr(e, f, [str(x) for x in v] if isinstance(v, list) else [])
    seeds = raw.get('seeds', [])
    e.seeds = [int(s) for s in seeds if isinstance(s, (int, str)) and str(s).isdigit()] \
        if isinstance(seeds, list) else []
    e.commits = []
    for h in raw.get('commits', []) or []:
        if looks_like_hash(str(h)):
            e.commits.append(str(h))
        else:
            errors.append(f'{where}: плохой hash коммита {h!r}')
    for f in ('files', 'runs', 'reports'):
        v = raw.get(f, []) or []
        setattr(e, f, _check_paths(list(v), root, where, errors, warnings))
    e.metrics = [m for m in (_metric(x, cat.glossary, where, errors, warnings, root)
                             for x in raw.get('metrics', []) or []) if m]
    e.evidence = [x for x in (_evidence(v, where, errors, warnings, root)
                              for v in raw.get('evidence', []) or []) if x]
    acc = raw.get('acceptance', []) or []
    e.acceptance = [a for a in acc if isinstance(a, dict) and a.get('criterion')]
    e.confirmed = bool(raw.get('confirmed', False)) and all(
        ev.exists is not False for ev in e.evidence) and bool(e.evidence)
    e.result = RESULT_OF_STATE[state]
    for d in (e.started, e.finished):
        if d and not (len(d) >= 7 and d[:4].isdigit() and d[4] == '-'):
            errors.append(f'{where}: дата {d!r} не в формате ГГГГ-ММ[-ДД]')
    return e


def _manual_event(raw: dict, cat: Catalog, root: Path, n: int) -> dict | None:
    where = f'manual_events[{n}]'
    if not isinstance(raw, dict) or not raw.get('when') or not raw.get('title'):
        cat.errors.append(f'{where}: нужны when и title')
        return None
    ev = dict(raw)
    ev.setdefault('kind', 'manual')
    if ev['kind'] not in EVENT_KINDS:
        cat.errors.append(f'{where}: тип события {ev["kind"]!r}')
        ev['kind'] = 'manual'
    ev.setdefault('state', 'unknown')
    if ev['state'] not in STATES:
        cat.errors.append(f'{where}: состояние {ev["state"]!r}')
        ev['state'] = 'unknown'
    ev.setdefault('date_precision', 'day')
    if ev['date_precision'] not in DATE_PRECISIONS:
        ev['date_precision'] = 'derived'
    if ev.get('area') and ev['area'] not in cat.area_ids():
        cat.errors.append(f'{where}: направление {ev["area"]!r} не объявлено')
    ev['commits'] = [h for h in ev.get('commits', []) or [] if looks_like_hash(str(h))]
    ev['files'] = _check_paths(list(ev.get('files', []) or []), root, where,
                               cat.errors, cat.warnings)
    return ev


def load_glossary(path: Path, errors: list[str], warnings: list[str]) -> Glossary:
    data = _read_json(path, EMPTY_GLOSSARY, warnings, 'словарь метрик')
    g = Glossary()
    metrics = data.get('metrics', {})
    if not isinstance(metrics, dict):
        errors.append('словарь метрик: metrics должен быть объектом')
        metrics = {}
    for key, desc in metrics.items():
        if not isinstance(desc, dict):
            errors.append(f'словарь метрик: {key!r} — описание не объект')
            continue
        for f in REQUIRED_METRIC_DESC:
            if not desc.get(f):
                errors.append(f'словарь метрик: у {key!r} нет поля {f!r}')
        if desc.get('better') and desc['better'] not in BETTER_VALUES:
            errors.append(f'словарь метрик: у {key!r} better={desc["better"]!r}, '
                          f'допустимо {"/".join(BETTER_VALUES)}')
        g.metrics[key] = desc
    aliases = data.get('aliases', {})
    if isinstance(aliases, dict):
        for a, canon in aliases.items():
            if canon not in g.metrics:
                errors.append(f'словарь метрик: алиас {a!r} ведёт на неизвестную {canon!r}')
            else:
                g.aliases[str(a)] = str(canon)
    terms = data.get('terms', {})
    if isinstance(terms, dict):
        g.terms = {str(k): v for k, v in terms.items() if isinstance(v, dict)}
    return g


def load(catalog_path: Path, glossary_path: Path, project_root: Path) -> Catalog:
    cat = Catalog(catalog_path=str(catalog_path), glossary_path=str(glossary_path))
    root = Path(project_root)
    cat.glossary = load_glossary(Path(glossary_path), cat.errors, cat.warnings)
    data = _read_json(Path(catalog_path), EMPTY_CATALOG, cat.warnings, 'каталог')

    seen: set[str] = set()
    for raw in data.get('areas', []) or []:
        if not isinstance(raw, dict) or not raw.get('id'):
            cat.errors.append(f'направление без id: {raw!r}')
            continue
        aid = str(raw['id'])
        if aid in seen:
            cat.errors.append(f'направление {aid!r}: повтор id')
            continue
        seen.add(aid)
        cat.areas.append(ProjectArea(
            id=aid, title=str(raw.get('title', aid)),
            description=str(raw.get('description', '')),
            tags=[str(t) for t in raw.get('tags', []) or []],
            parent_id=str(raw.get('parent_id', '')),
            order=int(raw.get('order', 0) or 0)))
    cat.areas.sort(key=lambda a: (a.order, a.title))

    seen = set()
    for raw in data.get('experiments', []) or []:
        if not isinstance(raw, dict):
            cat.errors.append(f'эксперимент не объект: {raw!r}')
            continue
        eid = str(raw.get('id', ''))
        if eid and eid in seen:
            cat.errors.append(f'эксперимент {eid!r}: повтор id')
            continue
        e = _experiment(raw, cat, root)
        if e is not None:
            seen.add(e.id)
            cat.experiments.append(e)

    for n, raw in enumerate(data.get('manual_events', []) or []):
        ev = _manual_event(raw, cat, root, n)
        if ev:
            cat.manual_events.append(ev)

    links = data.get('commit_links', {}) or {}
    if not isinstance(links, dict):
        cat.errors.append('commit_links должен быть объектом {hash: {...}}')
        links = {}
    exp_ids = {e.id for e in cat.experiments}
    for h, link in links.items():
        if not looks_like_hash(str(h)):
            cat.errors.append(f'commit_links: плохой hash {h!r}')
            continue
        if not isinstance(link, dict):
            link = {'experiments': link if isinstance(link, list) else []}
        for eid in link.get('experiments', []) or []:
            if eid not in exp_ids:
                cat.errors.append(f'commit_links[{h}]: эксперимент {eid!r} не найден')
        cat.commit_links[str(h)] = link
    return cat
