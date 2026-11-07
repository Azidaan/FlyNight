"""Сборка индекса истории: Git + план + каталог + артефакты → один объект.

Роли источников не смешиваются: коммиты берутся из Git, хронология и
интерпретация — из плана, связи и «ожидалось / получено» — из каталога.
Индексатор только связывает и помечает: у события есть коммит или нет,
файл существует или нет, число совпадает с источником или нет.

Кэш (.cache/project_history/index.json) — производный файл. Его отпечаток:
HEAD, mtime/size плана, каталога, словаря и отслеживаемых отчётов. Состояние
рабочей директории в кэш не пишется — оно читается живьём при каждом
открытии, поэтому устареть не может.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import artifacts as art
from . import catalog as cat_mod
from . import plan_parser
from .git_reader import GitError, GitReader
from .model import (
    CommitRecord, EvidenceReference, ExperimentRecord, ProjectArea,
    TimelineEvent)
from .safe_paths import resolve_inside

PLAN_REL = 'dissertation/ЕДИНЫЙ_ПЛАН.md'
CATALOG_REL = 'experiments/project_history.json'
GLOSSARY_REL = 'experiments/metrics_glossary.json'
CACHE_REL = '.cache/project_history'
INDEX_VERSION = 3
# отчёты, изменение которых устаревает кэш (кроме плана и каталога)
TRACKED_REPORT_GLOBS = ('*.md', 'experiments/results/*.csv', 'experiments/results/*.json')


@dataclass
class Index:
    built_at: str = ''
    fingerprint: dict = field(default_factory=dict)
    from_cache: bool = False
    version: int = INDEX_VERSION
    areas: list[ProjectArea] = field(default_factory=list)
    experiments: list[ExperimentRecord] = field(default_factory=list)
    commits: list[CommitRecord] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    glossary: dict = field(default_factory=dict)
    working_tree: dict = field(default_factory=dict)
    remote_refs: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self, with_working_tree: bool = True) -> dict[str, Any]:
        d = {
            'built_at': self.built_at, 'fingerprint': self.fingerprint,
            'from_cache': self.from_cache, 'version': self.version,
            'areas': [a.to_dict() for a in self.areas],
            'experiments': [e.to_dict() for e in self.experiments],
            'commits': [c.to_dict() for c in self.commits],
            'timeline': [t.to_dict() for t in self.timeline
                         if with_working_tree or t.kind != 'uncommitted'],
            'plan': self.plan, 'artifacts': self.artifacts,
            'glossary': self.glossary, 'remote_refs': self.remote_refs,
            'warnings': self.warnings, 'errors': self.errors, 'stats': self.stats,
        }
        if with_working_tree:
            d['working_tree'] = self.working_tree
        else:
            d['stats'] = dict(self.stats)
            if 'events' in d['stats']:
                d['stats']['events'] = len(d['timeline'])
            if 'by_kind' in d['stats']:
                d['stats']['by_kind'] = {kind: count for kind, count in d['stats']['by_kind'].items()
                                       if kind != 'uncommitted'}
        return d


# ——— отпечаток и кэш ———

def _stamp(p: Path) -> str:
    try:
        st = p.stat()
        return f'{int(st.st_mtime)}:{st.st_size}'
    except OSError:
        return 'missing'


def fingerprint(root: Path) -> dict:
    root = Path(root)
    git = GitReader(root)
    try:
        head = git.head() if git.available() else 'no-git'
    except GitError:
        head = 'no-git'
    reports = []
    for pattern in TRACKED_REPORT_GLOBS:
        for p in sorted(root.glob(pattern)):
            reports.append(f'{p.relative_to(root).as_posix()}={_stamp(p)}')
    digest = hashlib.sha1('\n'.join(reports).encode('utf-8')).hexdigest()[:16]
    return {'version': INDEX_VERSION, 'head': head,
            'plan': _stamp(root / PLAN_REL), 'catalog': _stamp(root / CATALOG_REL),
            'glossary': _stamp(root / GLOSSARY_REL), 'reports': digest,
            'n_reports': len(reports)}


def _cache_file(root: Path, cache_dir: Path | None) -> Path:
    return (Path(cache_dir) if cache_dir else Path(root) / CACHE_REL) / 'index.json'


# ——— сборка ———

def _commit_stats(git: GitReader, commits: list[CommitRecord], warnings: list[str]) -> None:
    """Число файлов и строк на каждый коммит одним чтением (--shortstat)."""
    try:
        out = git._run('log', '--format=%x1e%H', '--shortstat', 'HEAD')
    except GitError as e:
        warnings.append(f'git shortstat: {e}')
        return
    stats: dict[str, tuple[int, int, int]] = {}
    for block in out.split('\x1e'):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        h, files = lines[0], 0
        add = dele = 0
        for ln in lines[1:]:
            if 'changed' in ln:
                for part in ln.split(','):
                    part = part.strip()
                    n = int(part.split()[0]) if part and part.split()[0].isdigit() else 0
                    if 'file' in part:
                        files = n
                    elif 'insertion' in part:
                        add = n
                    elif 'deletion' in part:
                        dele = n
        stats[h] = (files, add, dele)
    for c in commits:
        f, a, d = stats.get(c.hash, (0, 0, 0))
        c.n_files, c.added, c.deleted = f, a, d
        c.is_empty = (f == 0) and not c.is_merge


def _plan_summary(doc: plan_parser.PlanDocument) -> dict:
    return {
        'path': PLAN_REL,
        'n_lines': len(doc.lines),
        'default_year': doc.default_year,
        'headings': [{'level': h.level, 'title': h.title, 'line': h.line,
                      'line_to': h.line_to} for h in doc.headings],
        'tables': [{'section': t.section, 'line': t.line, 'header': t.header,
                    'rows': t.rows} for t in doc.tables],
        'changelog': [{'when': e.when, 'date': e.date, 'time': e.time, 'title': e.title,
                       'line_from': e.line_from, 'line_to': e.line_to,
                       'hashes': e.all_hashes(), 'n_entries': len(e.entries)}
                      for e in doc.changelog],
    }


def _closed_table_events(doc: plan_parser.PlanDocument) -> list[TimelineEvent]:
    """Строки таблицы «Что закрыто» → события закрытия (даты — если есть в тексте)."""
    out = []
    for t in doc.tables:
        if not t.section.lower().startswith('4.') or 'закрыт' not in t.section.lower():
            continue
        for k, row in enumerate(t.rows):
            if len(row) < 2:
                continue
            name = row[0].replace('**', '').strip()
            m = plan_parser.re.search(r'\((\d{1,2})\.(\d{1,2})\)', name)
            when, prec = '', 'derived'
            if m and plan_parser._valid(int(m.group(2)), int(m.group(1))):
                when = plan_parser._iso(doc.default_year, int(m.group(2)), int(m.group(1)))
                prec = 'day'
            out.append(TimelineEvent(
                id=f'plan_closed:{t.line}:{k}', when=when, date_precision=prec,
                title=name, text=' · '.join(c.replace('**', '') for c in row[1:]),
                kind='plan_closed', state='closed',
                plan_lines=[t.line + 1 + k, t.line + 1 + k],
                source=f'{PLAN_REL}:{t.line + 1 + k}', raw=' | '.join(row)))
    return out


def build_index(root: Path, cache_dir: Path | None = None,
                write_cache: bool = False) -> Index:
    root = Path(root)
    idx = Index(built_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
    idx.fingerprint = fingerprint(root)
    warnings, errors = idx.warnings, idx.errors

    # 1. Git
    git = GitReader(root)
    commits: list[CommitRecord] = []
    if git.available():
        try:
            commits = git.log()
            _commit_stats(git, commits, warnings)
            branch = git.branch()
            for c in commits:
                c.branch = branch
            idx.remote_refs = git.remote_refs()
        except GitError as e:
            errors.append(f'Git: {e}')
    else:
        errors.append('Git недоступен: история коммитов не прочитана')
    idx.commits = commits
    by_full = {c.hash: c for c in commits}

    def find_commit(h: str) -> CommitRecord | None:
        if h in by_full:
            return by_full[h]
        matches = [c for c in commits if c.hash.startswith(h)]
        return matches[0] if len(matches) == 1 else None

    # 2. План
    doc: plan_parser.PlanDocument | None = None
    plan_path = root / PLAN_REL
    if plan_path.exists():
        try:
            doc = plan_parser.parse_plan(plan_path.read_text(encoding='utf-8'))
            idx.plan = _plan_summary(doc)
        except (OSError, ValueError) as e:
            errors.append(f'План: {e}')
    else:
        errors.append(f'План {PLAN_REL} не найден')

    # 3. Каталог и словарь
    cat = cat_mod.load(root / CATALOG_REL, root / GLOSSARY_REL, root)
    errors += [f'Каталог: {e}' for e in cat.errors]
    warnings += [f'Каталог: {w}' for w in cat.warnings]
    idx.areas = cat.areas
    idx.experiments = cat.experiments
    idx.glossary = {'metrics': cat.glossary.metrics, 'aliases': cat.glossary.aliases,
                    'terms': cat.glossary.terms}

    # 4. Артефакты
    try:
        idx.artifacts = art.scan(root)
    except OSError as e:
        warnings.append(f'Артефакты: {e}')
    artifact_paths = {a['path'] for a in idx.artifacts}

    # 5. Связывание эксперимент ↔ коммит ↔ артефакт
    exp_by_id = {e.id: e for e in cat.experiments}
    for e in cat.experiments:
        for h in e.commits:
            c = find_commit(h)
            if c is None:
                e.missing_commits.append(h)
                warnings.append(f'Каталог: эксперимент {e.id!r} ссылается на '
                                f'коммит {h}, которого нет в истории')
            else:
                if e.id not in c.experiments:
                    c.experiments.append(e.id)
        e.has_commit = bool(e.commits) and len(e.missing_commits) < len(e.commits)
        e.has_metrics = bool(e.metrics)
        kinds: set[str] = set()
        if e.has_commit:
            kinds.add('git')
        for m in e.metrics:
            if m.source:
                kinds.add(Path(m.source).suffix.lower().lstrip('.') or 'file')
                if m.locator and m.value is not None:
                    ok, note = art.check_metric_source(root, m.source, m.locator, m.value)
                    if ok is False:
                        m.note = (m.note + ' · ' if m.note else '') + note
                        warnings.append(f'Каталог: {e.id}/{m.key}: {note}')
                    elif ok is True and not m.note:
                        m.note = note
        for ev in e.evidence:
            if ev.kind == 'commit':
                c = find_commit(ev.ref)
                ev.exists = c is not None
                ev.level = 'confirmed' if c else 'unconfirmed'
                kinds.add('git')
            elif ev.kind == 'plan_section':
                kinds.add('plan')
                if doc and ev.locator:
                    loc = ev.locator.lstrip('§').strip()
                    hit = next((h for h in doc.headings
                                if h.title.startswith(loc + '.') or h.title.startswith(loc + ' ')
                                or h.title == loc), None)
                    if hit:
                        ev.note = f'строки {hit.line}–{hit.line_to}'
                        ev.level = 'confirmed'
                    else:
                        ev.note = 'раздел не найден по номеру'
            else:
                kinds.add(ev.kind)
        arts = [p for p in e.files + e.runs + e.reports if resolve_inside(root, p)
                and resolve_inside(root, p).exists()]
        e.has_artifacts = bool(arts) or any(m.source_exists for m in e.metrics)
        if any(p in artifact_paths for p in e.reports):
            kinds.add('markdown')
        if e.runs:
            kinds.add('runs')
        e.source_kinds = sorted(kinds)

    for h, link in cat.commit_links.items():
        c = find_commit(h)
        if c is None:
            warnings.append(f'Каталог: commit_links ссылается на неизвестный коммит {h}')
            continue
        for eid in link.get('experiments', []) or []:
            if eid in exp_by_id and eid not in c.experiments:
                c.experiments.append(eid)
                if h not in exp_by_id[eid].commits:
                    exp_by_id[eid].commits.append(h)
                    exp_by_id[eid].has_commit = True
        if link.get('result'):
            c.result = str(link['result'])
            c.result_source = str(link.get('source', 'каталог'))

    # 6. Хронология
    timeline: list[TimelineEvent] = []
    if doc is not None:
        for e in doc.changelog:
            ev_ids = []
            block_hashes = [h for h in e.hashes]
            items = e.entries or []
            if not items:
                tid = f'plan_changelog:{e.line_from}'
                timeline.append(TimelineEvent(
                    id=tid, when=e.when, date_precision='exact' if e.time else 'day',
                    title=e.title, text=e.text, kind='plan_changelog',
                    commits=list(e.all_hashes()), plan_lines=[e.line_from, e.line_to],
                    source=f'{PLAN_REL}:{e.line_from}'))
                continue
            # заголовок блока — как событие-обёртка (без hash подзаписей)
            timeline.append(TimelineEvent(
                id=f'plan_changelog:{e.line_from}', when=e.when,
                date_precision='exact' if e.time else 'day', title=e.title,
                text=e.text, kind='plan_changelog', commits=block_hashes,
                plan_lines=[e.line_from, e.line_from], source=f'{PLAN_REL}:{e.line_from}'))
            for s in items:
                timeline.append(TimelineEvent(
                    id=f'plan_changelog:{s.line_from}', when=s.when,
                    date_precision=s.precision, title=s.title, text=s.text,
                    kind='plan_changelog', commits=list(s.hashes),
                    plan_lines=[s.line_from, s.line_to],
                    source=f'{PLAN_REL}:{s.line_from}'))
                ev_ids.append(s.line_from)
        for c in doc.chronology:
            timeline.append(TimelineEvent(
                id=f'plan_chronology:{c.line}', when=c.when, when_end=c.when_end,
                date_precision=c.precision, title=c.title,
                text=c.text if c.kind == 'bullet' else c.text[:600],
                kind='plan_chronology', commits=list(c.hashes),
                plan_lines=[c.line, c.line_to], source=f'{PLAN_REL}:{c.line}',
                raw=c.text if c.precision == 'derived' else ''))
        timeline += _closed_table_events(doc)

    for n, m in enumerate(cat.manual_events):
        timeline.append(TimelineEvent(
            id=f'manual:{n}', when=str(m['when']), when_end=str(m.get('when_end', '')),
            date_precision=m['date_precision'], title=str(m['title']),
            text=str(m.get('text', '')), kind=m['kind'], area=str(m.get('area', '')),
            state=m['state'], experiment_id=str(m.get('experiment', '')),
            commits=list(m['commits']), files=list(m['files']),
            source=str(m.get('source', CATALOG_REL))))

    for c in commits:
        timeline.append(TimelineEvent(
            id=f'commit:{c.hash}', when=c.date, date_precision='exact',
            title=c.subject, text=c.body, kind='commit', commits=[c.hash],
            experiment_id=c.experiments[0] if c.experiments else '',
            source=f'git:{c.short}', has_commit=True))

    # пометки: есть ли коммит, артефакт, направление/состояние через эксперимент
    exp_by_commit: dict[str, list[str]] = {}
    for e in cat.experiments:
        for h in e.commits:
            c = find_commit(h)
            if c:
                exp_by_commit.setdefault(c.hash, []).append(e.id)
    for ev in timeline:
        resolved = []
        for h in ev.commits:
            c = find_commit(h)
            if c is not None:
                resolved.append(c.hash)
                if ev.kind != 'commit' and ev.id not in c.plan_events:
                    c.plan_events.append(ev.id)
        ev.commits = [h[:7] for h in resolved] if ev.kind != 'commit' else [resolved[0][:7]] if resolved else []
        ev.has_commit = bool(resolved)
        if not ev.experiment_id:
            for h in resolved:
                if exp_by_commit.get(h):
                    ev.experiment_id = exp_by_commit[h][0]
                    break
        if ev.experiment_id and ev.experiment_id in exp_by_id:
            e = exp_by_id[ev.experiment_id]
            ev.area = ev.area or e.area
            if ev.state == 'unknown' and ev.kind != 'plan_closed':
                ev.state = e.state
        ev.has_artifact = any(resolve_inside(root, f) and resolve_inside(root, f).exists()
                              for f in ev.files)

    # 7. Незакоммиченное состояние — всегда живьём
    _attach_working_tree(idx, git, timeline)
    idx.timeline = timeline
    _sort_timeline(idx.timeline)
    idx.stats = _stats(idx)
    if write_cache:
        _write_cache(idx, _cache_file(root, cache_dir))
    return idx


def _attach_working_tree(idx: Index, git: GitReader, timeline: list[TimelineEvent]) -> None:
    timeline[:] = [t for t in timeline if t.kind != 'uncommitted']
    wt: dict = {}
    if git.available():
        try:
            wt = git.working_tree()
        except GitError as e:
            idx.warnings.append(f'git status: {e}')
    idx.working_tree = wt
    unlinked = [t.id for t in timeline
                if t.kind in ('plan_changelog', 'plan_chronology') and not t.has_commit]
    files = [m['path'] for m in wt.get('modified', [])] + list(wt.get('untracked', []))
    text = ('Ветка {b}, HEAD {h}. Изменённых файлов: {m}, новых неотслеживаемых: {u}. '
            'Это не коммит: изменения ещё не зафиксированы в истории.').format(
        b=wt.get('branch', '?'), h=wt.get('short', '?'),
        m=len(wt.get('modified', [])), u=len(wt.get('untracked', [])))
    timeline.append(TimelineEvent(
        id='uncommitted', when=time.strftime('%Y-%m-%dT%H:%M'), date_precision='exact',
        title='Текущее незакоммиченное состояние', text=text, kind='uncommitted',
        state='unknown', files=files, source='git status --porcelain',
        raw=json.dumps({'unlinked_plan_events': unlinked}, ensure_ascii=False)))


def _sort_timeline(timeline: list[TimelineEvent]) -> None:
    def key(t: TimelineEvent):
        w = t.when or '0000'          # без даты — в самый низ
        return (w[:16], t.kind != 'uncommitted', t.plan_lines[0] if t.plan_lines else 0)
    timeline.sort(key=key, reverse=True)


def _stats(idx: Index) -> dict:
    by_state: dict[str, int] = {}
    for e in idx.experiments:
        by_state[e.state] = by_state.get(e.state, 0) + 1
    by_kind: dict[str, int] = {}
    for t in idx.timeline:
        by_kind[t.kind] = by_kind.get(t.kind, 0) + 1
    return {'experiments': len(idx.experiments), 'by_state': by_state,
            'commits': len(idx.commits),
            'empty_commits': sum(1 for c in idx.commits if c.is_empty),
            'events': len(idx.timeline), 'by_kind': by_kind,
            'events_without_commit': sum(
                1 for t in idx.timeline
                if t.kind in ('plan_changelog', 'plan_chronology') and not t.has_commit),
            'artifacts': len(idx.artifacts), 'areas': len(idx.areas)}


# ——— кэш ———

def _write_cache(idx: Index, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(idx.to_dict(with_working_tree=False),
                                  ensure_ascii=False), encoding='utf-8')
        tmp.replace(path)
    except OSError as e:
        idx.warnings.append(f'Кэш не записан: {e}')


def _from_dict(d: dict) -> Index:
    from .model import CommitFile, MetricValue
    idx = Index(built_at=d.get('built_at', ''), fingerprint=d.get('fingerprint', {}),
                version=d.get('version', 0))
    idx.areas = [ProjectArea(**a) for a in d.get('areas', [])]
    for e in d.get('experiments', []):
        e = dict(e)
        e.pop('state_label', None)
        e.pop('result_label', None)
        e['metrics'] = [MetricValue(**m) for m in e.get('metrics', [])]
        e['evidence'] = [EvidenceReference(**v) for v in e.get('evidence', [])]
        idx.experiments.append(ExperimentRecord(**e))
    for c in d.get('commits', []):
        c = dict(c)
        c['files'] = [CommitFile(**f) for f in c.get('files', [])]
        idx.commits.append(CommitRecord(**c))
    for t in d.get('timeline', []):
        t = dict(t)
        t.pop('state_label', None)
        t.pop('kind_label', None)
        idx.timeline.append(TimelineEvent(**t))
    idx.plan = d.get('plan', {})
    idx.artifacts = d.get('artifacts', [])
    idx.glossary = d.get('glossary', {})
    idx.remote_refs = d.get('remote_refs', [])
    idx.warnings = list(d.get('warnings', []))
    idx.errors = list(d.get('errors', []))
    idx.stats = d.get('stats', {})
    return idx


def load_index(root: Path, cache_dir: Path | None = None, force: bool = False) -> Index:
    """Индекс из кэша, если отпечаток совпал, иначе пересборка с записью кэша."""
    root = Path(root)
    path = _cache_file(root, cache_dir)
    if not force and path.exists():
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            if raw.get('fingerprint') == fingerprint(root) and raw.get('version') == INDEX_VERSION:
                idx = _from_dict(raw)
                idx.from_cache = True
                _attach_working_tree(idx, GitReader(root), idx.timeline)
                _sort_timeline(idx.timeline)
                idx.stats = _stats(idx)
                return idx
        except (OSError, ValueError, TypeError) as e:
            pass  # кэш повреждён — пересобираем
    idx = build_index(root, cache_dir=cache_dir, write_cache=True)
    idx.from_cache = False
    return idx
