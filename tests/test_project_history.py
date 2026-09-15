"""Тесты окна «История проекта» — без GUI.

Нумерация в названиях (r16_x) — пункты требований: 16.1 разбор журнала
плана, 16.3 коммиты, 16.7 пути и так далее.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.project_history import model, safe_paths  # noqa: E402


# ————— модель и состояния (16.10) —————

def test_r16_10_states_are_the_ten_agreed():
    assert set(model.STATES) == {
        'planned', 'ready', 'running', 'completed_positive',
        'completed_negative', 'inconclusive', 'closed', 'infrastructure',
        'superseded', 'unknown'}
    # у каждого состояния есть русская подпись и класс результата
    for s in model.STATES:
        assert model.STATE_LABELS[s]
        assert model.RESULT_OF_STATE[s] in model.RESULT_KINDS


def test_r16_10_invalid_state_rejected():
    with pytest.raises(ValueError):
        model.check_state('done')
    assert model.check_state('closed') == 'closed'


# ————— безопасные пути и hash (16.6, 16.7) —————

def test_r16_7_path_inside_project_resolves(tmp_path):
    root = tmp_path / 'proj'
    (root / 'a').mkdir(parents=True)
    (root / 'a' / 'f.txt').write_text('x', encoding='utf-8')
    p = safe_paths.resolve_inside(root, 'a/f.txt')
    assert p == (root / 'a' / 'f.txt').resolve()


@pytest.mark.parametrize('rel', ['../x', 'a/../../x', '..\\x', 'C:/Windows'])
def test_r16_7_path_escape_rejected(tmp_path, rel):
    root = tmp_path / 'proj'
    root.mkdir()
    assert safe_paths.resolve_inside(root, rel) is None


def test_r16_7_sibling_with_same_prefix_rejected(tmp_path):
    """E:\\Dissertation_fake не должен проходить как «внутри» E:\\Dissertation."""
    root = tmp_path / 'proj'
    fake = tmp_path / 'proj_fake'
    root.mkdir()
    fake.mkdir()
    (fake / 'f.txt').write_text('x', encoding='utf-8')
    assert safe_paths.resolve_inside(root, str(fake / 'f.txt')) is None
    assert safe_paths.resolve_inside(root, '../proj_fake/f.txt') is None


@pytest.mark.parametrize('h,ok', [
    ('3472a05', True), ('3472a05d061f1c699624465e69b997ee8d61484a', True),
    ('HEAD', False), ('3472a05; rm', False), ('', False), ('xyz1234', False),
    ('abc', False), ('--all', False),
])
def test_r16_6_hash_syntax(h, ok):
    assert safe_paths.looks_like_hash(h) is ok


# ————— Git (16.3, 16.4, 16.5, 16.6, 16.15) —————

from evaluation.project_history import git_reader  # noqa: E402

CONTROL_COMMITS = ['3472a05', '4690dad', '2dbdce0', '889913f', '7a989be']


@pytest.fixture(scope='module')
def git():
    g = git_reader.GitReader(PROJECT_ROOT)
    if not g.available():
        pytest.skip('git недоступен')
    return g


def test_r16_15_control_commits_exist(git):
    for h in CONTROL_COMMITS:
        assert git.commit_exists(h), h


def test_r16_3_log_returns_commits_with_fields(git):
    commits = git.log(limit=20)
    assert len(commits) == 20
    c = commits[0]
    assert safe_paths.looks_like_hash(c.hash) and len(c.hash) == 40
    assert c.short and c.date and c.author and c.subject
    assert c.date[:4].isdigit()
    # порядок — от новых к старым
    assert commits[0].date >= commits[-1].date


def test_r16_4_commit_files_and_stats(git):
    d = git.commit_detail('4690dad')
    assert d.files_loaded and d.n_files == 16
    paths = {f.path for f in d.files}
    assert 'tracking/flyvis_tracker.py' in paths
    f = next(x for x in d.files if x.path == 'tracking/flyvis_tracker.py')
    assert f.status == 'A' and f.added == 474 and f.deleted == 0
    assert d.added == 3255 and d.deleted == 2


def test_r16_4_empty_commit_is_marked(git):
    d = git.commit_detail('3472a05')
    assert d.is_empty and d.n_files == 0 and d.subject == 'Tracker 1,0'


def test_r16_5_diff_is_capped(git):
    text, truncated = git.diff_preview('4690dad', max_bytes=2000)
    assert truncated is True
    assert len(text.encode('utf-8')) <= 2000 + 200   # + подпись об обрезке
    full, trunc2 = git.diff_preview('2dbdce0', max_bytes=10_000_000)
    assert trunc2 is False and 'video_alignment' in full


@pytest.mark.parametrize('bad', ['HEAD', '3472a05;', 'deadbeef', '--all', 'main'])
def test_r16_6_unknown_or_malformed_hash(git, bad):
    assert git.commit_exists(bad) is False
    with pytest.raises(git_reader.GitError):
        git.commit_detail(bad)
    with pytest.raises(git_reader.GitError):
        git.diff_preview(bad)


def test_r16_6_reader_never_uses_shell_or_mutating_commands():
    import inspect
    src = inspect.getsource(git_reader)
    assert 'shell=True' not in src
    for forbidden in ("'reset'", "'checkout'", "'switch'", "'clean'", "'pull'",
                      "'fetch'", "'push'", "'rebase'", "'merge'", "'stash'"):
        assert forbidden not in src, forbidden


def test_working_tree_state_has_expected_shape(git):
    st = git.working_tree()
    assert st['branch']
    assert len(st['head']) == 40
    assert isinstance(st['modified'], list) and isinstance(st['untracked'], list)
    assert isinstance(st['is_clean'], bool)


# ————— разбор единого плана (16.1, 16.2) —————

from evaluation.project_history import plan_parser  # noqa: E402

PLAN_PATH = PROJECT_ROOT / 'dissertation' / 'ЕДИНЫЙ_ПЛАН.md'

SAMPLE_PLAN = """# Единый план работы

Составлен 07.09.2026. Текст.

## 2. Хронология

### 2.1. Июнь — июль: каркас

Собраны данные.

### 2.2. 18–25.08: DRB, версии 1 и 2

- **19.08.** Сравнение оказалось нечестным.
- **22.08.** Замер шума: **шесть выводов не пережили**.

### 2.3. 31.08: фиксированный фильтр

Проверена рекомендация.

## 3. Числа

| что | значение | источник |
|---|---|---|
| разброс, 12 эпох | 0.0144 | `probe_subsets.csv` |
| ночная mAP50 | 0.3772 ± 0.0031 | `target_full_night.csv` |

## Журнал изменений

**13.09.2026, 19:19 MSK** — FlyVis и раздельная уверенность.

*02:17, самостоятельный трекер.* Коммит `4690dad` добавил трекер.

*Стресс-проверка.* Без коммита, только отчёт <b>x</b>.

*14.09, 21:35, аудит.* Подтверждено, что набор пар строится из `deadbee`.

**08.09.2026** — внесён раздел 2.8.
"""


def test_r16_2_headings_with_lines_and_levels():
    doc = plan_parser.parse_plan(SAMPLE_PLAN)
    got = [(h.level, h.title, h.line) for h in doc.headings]
    assert got[0] == (1, 'Единый план работы', 1)
    assert (3, '2.2. 18–25.08: DRB, версии 1 и 2', 11) in got
    assert (2, 'Журнал изменений', 27) in got
    # у раздела есть диапазон строк
    sec = doc.section_by_title('2.2. 18–25.08: DRB, версии 1 и 2')
    assert sec.line_from == 11 and sec.line_to == 15


def test_r16_2_tables_parsed_with_context():
    doc = plan_parser.parse_plan(SAMPLE_PLAN)
    assert len(doc.tables) == 1
    t = doc.tables[0]
    assert t.header == ['что', 'значение', 'источник']
    assert t.rows[1] == ['ночная mAP50', '0.3772 ± 0.0031', '`target_full_night.csv`']
    assert t.section == '3. Числа' and t.line == 22


def test_r16_1_changelog_entries_and_subentries():
    doc = plan_parser.parse_plan(SAMPLE_PLAN)
    assert [e.date for e in doc.changelog] == ['2026-09-13', '2026-09-08']
    e = doc.changelog[0]
    assert e.time == '19:19' and e.title == 'FlyVis и раздельная уверенность.'
    assert e.line_from == 29
    subs = e.entries
    assert [s.title for s in subs] == [
        'самостоятельный трекер', 'Стресс-проверка', 'аудит']
    assert subs[0].when == '2026-09-13T02:17' and subs[0].hashes == ['4690dad']
    assert subs[1].when == '2026-09-13' and subs[1].hashes == []
    # дата внутри подзаписи важнее даты блока
    assert subs[2].when == '2026-09-14T21:35' and subs[2].hashes == ['deadbee']
    assert '<b>x</b>' in subs[1].text          # текст хранится как есть, экранирование — при показе


def test_r16_1_chronology_dates_without_year_get_year_from_context():
    doc = plan_parser.parse_plan(SAMPLE_PLAN)
    items = doc.chronology
    by_title = {i.title: i for i in items}
    sec = by_title['18–25.08: DRB, версии 1 и 2']
    assert (sec.when, sec.when_end, sec.precision) == ('2026-08-18', '2026-08-25', 'range')
    b = by_title['Сравнение оказалось нечестным.']
    assert (b.when, b.precision, b.line) == ('2026-08-19', 'day', 13)
    m = by_title['Июнь — июль: каркас']
    assert m.precision == 'month' and m.when == '2026-06-01' and m.when_end == '2026-07-31'
    single = by_title['31.08: фиксированный фильтр']
    assert single.when == '2026-08-31' and single.precision == 'day'


def test_r16_1_real_plan_changelog_links_control_commits():
    doc = plan_parser.parse_plan(PLAN_PATH.read_text(encoding='utf-8'))
    assert len(doc.changelog) >= 4
    all_hashes = {h for e in doc.changelog for h in e.all_hashes()}
    assert {'4690dad', '2dbdce0', '889913f'} <= all_hashes
    assert any(h.title.startswith('Журнал изменений') for h in doc.headings)
    assert len(doc.tables) >= 10
    assert doc.default_year == 2026


def test_r16_1_real_plan_dated_paragraphs_in_chronology():
    doc = plan_parser.parse_plan(PLAN_PATH.read_text(encoding='utf-8'))
    bullets = [c for c in doc.chronology if c.kind == 'bullet']
    whens = {(c.when, c.when_end) for c in bullets}
    assert ('2026-09-09', '') in whens
    assert ('2026-09-10', '2026-09-11') in whens
    for c in doc.chronology:
        assert c.precision in ('day', 'range', 'month', 'derived')
        if c.precision != 'derived':
            assert c.when.startswith('2026-')


# ————— каталог и словарь метрик (16.9, 16.10, 16.11) —————

from evaluation.project_history import catalog  # noqa: E402


def _glossary() -> dict:
    return {'schema_version': 1,
            'metrics': {'HOTA': {'title': 'HOTA', 'ru': 'совместная оценка', 'better': 'higher',
                                 'tool': 'TrackEval'},
                        'IDSw': {'title': 'IDSw', 'ru': 'переключения ID', 'better': 'lower'}},
            'aliases': {'IDSW': 'IDSw', 'idsw': 'IDSw'},
            'terms': {'ReID': {'title': 'ReID', 'ru': 'повторная идентификация по внешнему виду'}}}


def _catalog(**over) -> dict:
    base = {'schema_version': 1,
            'areas': [{'id': 'flyvis', 'title': 'FlyVis'}],
            'experiments': [{
                'id': 'e1', 'title': 'Эксперимент', 'area': 'flyvis', 'state': 'completed_negative',
                'commits': ['4690dad'],
                'metrics': [{'key': 'HOTA', 'value': 29.113, 'baseline': 52.437,
                             'source': 'dissertation/ЕДИНЫЙ_ПЛАН.md'},
                            {'key': 'IDSW', 'value': 3384}],
                'files': ['tracking/flyvis_tracker.py'],
                'evidence': [{'kind': 'plan_section', 'ref': 'dissertation/ЕДИНЫЙ_ПЛАН.md',
                              'locator': '§0'}]}],
            'manual_events': [],
            'commit_links': {}}
    base.update(over)
    return base


def _write(tmp_path, cat, glos):
    c = tmp_path / 'project_history.json'
    g = tmp_path / 'metrics_glossary.json'
    c.write_text(json.dumps(cat, ensure_ascii=False), encoding='utf-8')
    g.write_text(json.dumps(glos, ensure_ascii=False), encoding='utf-8')
    return c, g


def test_catalog_valid_loads_without_errors(tmp_path):
    c, g = _write(tmp_path, _catalog(), _glossary())
    loaded = catalog.load(c, g, project_root=PROJECT_ROOT)
    assert loaded.errors == []
    e = loaded.experiments[0]
    assert e.state == 'completed_negative'
    # алиас метрики разрешён и описание подцеплено
    m = {x.key: x for x in e.metrics}
    assert 'IDSw' in m and m['IDSw'].described and m['IDSw'].better == 'lower'
    assert m['HOTA'].title == 'HOTA'
    # процент к baseline: ниже — хуже для higher
    assert m['HOTA'].delta_pct == pytest.approx((29.113 - 52.437) / 52.437 * 100, abs=0.01)


def test_r16_9_duplicate_ids_rejected(tmp_path):
    cat = _catalog()
    cat['experiments'].append(dict(cat['experiments'][0]))
    cat['areas'].append({'id': 'flyvis', 'title': 'дубль'})
    c, g = _write(tmp_path, cat, _glossary())
    loaded = catalog.load(c, g, project_root=PROJECT_ROOT)
    assert any('e1' in err and 'повтор' in err for err in loaded.errors)
    assert any('flyvis' in err and 'повтор' in err for err in loaded.errors)


def test_r16_10_bad_state_and_unknown_area_reported(tmp_path):
    cat = _catalog()
    cat['experiments'][0]['state'] = 'done'
    cat['experiments'][0]['area'] = 'nope'
    c, g = _write(tmp_path, cat, _glossary())
    loaded = catalog.load(c, g, project_root=PROJECT_ROOT)
    assert any('состояние' in err for err in loaded.errors)
    assert any('направление' in err for err in loaded.errors)


def test_r16_11_missing_source_file_detected(tmp_path):
    cat = _catalog()
    cat['experiments'][0]['files'] = ['tracking/does_not_exist.py']
    cat['experiments'][0]['evidence'][0]['ref'] = 'reports/nothing.md'
    c, g = _write(tmp_path, cat, _glossary())
    loaded = catalog.load(c, g, project_root=PROJECT_ROOT)
    assert any('does_not_exist' in w for w in loaded.warnings)
    ev = loaded.experiments[0].evidence[0]
    assert ev.exists is False and ev.level == 'unconfirmed'


def test_metric_without_glossary_entry_is_flagged(tmp_path):
    cat = _catalog()
    cat['experiments'][0]['metrics'].append({'key': 'mystery', 'value': 1})
    c, g = _write(tmp_path, cat, _glossary())
    loaded = catalog.load(c, g, project_root=PROJECT_ROOT)
    m = next(x for x in loaded.experiments[0].metrics if x.key == 'mystery')
    assert m.described is False
    assert any('mystery' in err for err in loaded.errors)


def test_glossary_requires_ru_and_better(tmp_path):
    glos = _glossary()
    glos['metrics']['HOTA'] = {'title': 'HOTA'}
    c, g = _write(tmp_path, _catalog(), glos)
    loaded = catalog.load(c, g, project_root=PROJECT_ROOT)
    assert any('HOTA' in err and ('ru' in err or 'better' in err) for err in loaded.errors)


def test_bad_hash_and_outside_path_rejected(tmp_path):
    cat = _catalog()
    cat['experiments'][0]['commits'] = ['not a hash']
    cat['experiments'][0]['files'] = ['../outside.py']
    c, g = _write(tmp_path, cat, _glossary())
    loaded = catalog.load(c, g, project_root=PROJECT_ROOT)
    assert any('hash' in err for err in loaded.errors)
    assert any('outside' in err for err in loaded.errors)


def test_missing_catalog_file_gives_empty_catalog_with_warning(tmp_path):
    loaded = catalog.load(tmp_path / 'no.json', tmp_path / 'no2.json', project_root=PROJECT_ROOT)
    assert loaded.experiments == [] and loaded.warnings


# ————— артефакты, индекс, кэш (16.13, 16.14) —————

from evaluation.project_history import artifacts, index as hindex  # noqa: E402


def test_artifact_scan_respects_limits_and_extensions(tmp_path):
    root = tmp_path
    (root / 'experiments' / 'results').mkdir(parents=True)
    (root / 'runs' / 'x' / 'combined').mkdir(parents=True)
    (root / 'runs' / 'x' / 'combined' / 'deep' / 'deeper').mkdir(parents=True)
    (root / 'experiments' / 'results' / 'a.csv').write_text('m,v\n1,2\n', encoding='utf-8')
    (root / 'experiments' / 'results' / 'b.png').write_bytes(b'\x89PNG')
    (root / 'runs' / 'x' / 'combined' / 'summary.json').write_text('{"HOTA": 1}', encoding='utf-8')
    (root / 'runs' / 'x' / 'combined' / 'deep' / 'deeper' / 'far.json').write_text('{}', encoding='utf-8')
    (root / 'REPORT.md').write_text('# r', encoding='utf-8')
    big = root / 'experiments' / 'results' / 'big.csv'
    big.write_text('x' * 10, encoding='utf-8')
    found = artifacts.scan(root, max_files=100, max_bytes=5)
    paths = {a['path'] for a in found}
    assert 'experiments/results/a.csv' in paths
    assert 'runs/x/combined/summary.json' in paths
    assert 'REPORT.md' in paths
    assert 'experiments/results/b.png' not in paths          # не тот тип
    assert 'runs/x/combined/deep/deeper/far.json' not in paths  # глубже лимита
    big_rec = next(a for a in found if a['path'].endswith('big.csv'))
    assert big_rec['too_big'] is True
    assert len(artifacts.scan(root, max_files=2)) == 2


def test_artifact_json_value_lookup(tmp_path):
    f = tmp_path / 's.json'
    f.write_text('{"tiled": {"HOTA": 29.113, "n": [1, 2]}}', encoding='utf-8')
    assert artifacts.lookup_value(f, 'tiled.HOTA') == 29.113
    assert artifacts.lookup_value(f, 'tiled.n.1') == 2
    assert artifacts.lookup_value(f, 'nope.x') is None
    c = tmp_path / 't.csv'
    c.write_text('модель,метрика,значение\nbase,HOTA,52.4\n', encoding='utf-8')
    assert artifacts.lookup_value(c, 'row=1;col=значение') == 52.4
    assert artifacts.lookup_value(c, 'метрика=HOTA;col=значение') == 52.4


@pytest.fixture(scope='module')
def built():
    return hindex.build_index(PROJECT_ROOT, cache_dir=None)


def test_index_contains_all_sources(built):
    assert len(built.commits) >= 160
    assert built.working_tree['head']
    assert built.plan['headings']
    assert any(e.kind == 'uncommitted' for e in built.timeline)


def test_r16_13_events_with_and_without_commit(built):
    with_commit = [e for e in built.timeline if e.kind == 'plan_changelog' and e.has_commit]
    without = [e for e in built.timeline if e.kind == 'plan_changelog' and not e.has_commit]
    assert with_commit and without
    hashes = {h for e in with_commit for h in e.commits}
    assert {'4690dad', '2dbdce0', '889913f'} <= hashes
    # у события без коммита пометка не «не выполнено», а отсутствие связи
    assert all(e.commits == [] for e in without)
    # каждый связанный hash существует в истории
    by_short = {c.short: c for c in built.commits}
    for h in hashes:
        assert any(c.hash.startswith(h) for c in built.commits), h


def test_empty_commit_flagged_in_list(built):
    c = next(c for c in built.commits if c.hash.startswith('3472a05'))
    assert c.is_empty is True and c.n_files == 0


def test_index_serialises_to_json(built):
    d = built.to_dict()
    s = json.dumps(d, ensure_ascii=False)
    assert 'timeline' in d and 'experiments' in d and len(s) > 1000


def test_r16_14_cache_invalidated_by_fingerprint(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    first = hindex.load_index(PROJECT_ROOT, cache_dir=cache)
    assert (cache / 'index.json').exists()
    second = hindex.load_index(PROJECT_ROOT, cache_dir=cache)
    assert second.from_cache is True and second.fingerprint == first.fingerprint
    # изменился отпечаток (как если бы сдвинулся HEAD или план) — пересборка
    monkeypatch.setattr(hindex, 'fingerprint',
                        lambda root: dict(first.fingerprint, head='0' * 40))
    third = hindex.load_index(PROJECT_ROOT, cache_dir=cache)
    assert third.from_cache is False
    # состояние рабочей директории в кэше не хранится — всегда живое
    raw = json.loads((cache / 'index.json').read_text(encoding='utf-8'))
    assert 'working_tree' not in raw


# ————— фильтры и экранирование (16.8, 16.12) —————

from evaluation.project_history import filters, render  # noqa: E402


def _recs():
    return [
        {'id': 'a', 'title': 'DRB v2', 'area': 'drb', 'state': 'closed', 'result': 'negative',
         'started': '2026-08-18', 'finished': '2026-08-31', 'has_commit': True,
         'has_metrics': True, 'has_artifacts': True, 'source_kinds': ['git', 'plan'],
         'seeds': [42, 123], 'detector': 'YOLO26s', 'tracker': '', 'tags': ['fga'],
         'conditions': ['night'], 'summary': 'частотная маска'},
        {'id': 'b', 'title': 'FlyVis stress', 'area': 'flyvis', 'state': 'completed_negative',
         'result': 'negative', 'started': '2026-09-13', 'finished': '2026-09-13',
         'has_commit': True, 'has_metrics': True, 'has_artifacts': True,
         'source_kinds': ['git', 'json', 'markdown'], 'seeds': [], 'detector': 'YOLO26s',
         'tracker': 'FlyVisTracker', 'tags': [], 'conditions': ['night', 'overexposed'],
         'summary': 'HOTA 29 против 52'},
        {'id': 'c', 'title': 'q_match 30 Гц', 'area': 'flyvis', 'state': 'planned',
         'result': 'none', 'started': '', 'finished': '', 'has_commit': False,
         'has_metrics': False, 'has_artifacts': False, 'source_kinds': ['plan'],
         'seeds': [], 'detector': '', 'tracker': 'XYWHByteTrack', 'tags': ['q_match'],
         'conditions': [], 'summary': 'ещё не запускался'},
    ]


def ids(rs):
    return [r['id'] for r in rs]


def test_r16_12_filters_combine_with_and():
    rs = _recs()
    assert ids(filters.apply(rs, {})) == ['a', 'b', 'c']
    assert ids(filters.apply(rs, {'area': ['flyvis']})) == ['b', 'c']
    assert ids(filters.apply(rs, {'area': ['flyvis'], 'has_commit': True})) == ['b']
    assert ids(filters.apply(rs, {'result': ['negative'], 'q': 'hota'})) == ['b']
    assert ids(filters.apply(rs, {'state': ['planned', 'closed']})) == ['a', 'c']
    assert ids(filters.apply(rs, {'date_from': '2026-09-01'})) == ['b']
    assert ids(filters.apply(rs, {'date_to': '2026-08-31'})) == ['a']
    assert ids(filters.apply(rs, {'has_metrics': False})) == ['c']
    assert ids(filters.apply(rs, {'source_kind': ['json']})) == ['b']
    assert ids(filters.apply(rs, {'seed': '123'})) == ['a']
    assert ids(filters.apply(rs, {'tracker': ['XYWHByteTrack']})) == ['c']
    assert ids(filters.apply(rs, {'condition': ['overexposed']})) == ['b']
    assert ids(filters.apply(rs, {'tag': ['q_match'], 'area': ['drb']})) == []
    # записи без даты не проходят фильтр по дате, но проходят без него
    assert 'c' not in ids(filters.apply(rs, {'date_from': '2020-01-01'}))


def test_filter_facets_are_built_from_data():
    f = filters.facets(_recs())
    assert f['area'] == ['drb', 'flyvis']
    assert f['tracker'] == ['FlyVisTracker', 'XYWHByteTrack']
    assert f['seed'] == ['123', '42']
    assert f['condition'] == ['night', 'overexposed']
    assert 'planned' in f['state']


def test_r16_8_html_escaping():
    s = '<script>alert(1)</script> & "q" \'x\' `код`'
    out = render.escape(s)
    assert '<script>' not in out and '&lt;script&gt;' in out
    assert '&amp;' in out and '&quot;' in out and '&#x27;' in out


def test_r16_8_plan_render_escapes_and_anchors():
    md = ('# Заголовок <b>x</b>\n\nАбзац с `code` и **жирным** <img src=x onerror=alert(1)>\n\n'
          '| a | b |\n|---|---|\n| 1 | <i>2</i> |\n\n- пункт\n- ещё\n')
    doc = plan_parser.parse_plan(md)
    html = render.plan_html(doc)
    assert 'onerror' not in html or '&lt;img' in html
    assert '<img' not in html and '<i>' not in html
    assert 'id="L1"' in html and '<h1' in html
    assert '<code>code</code>' in html and '<b>жирным</b>' in html
    assert '<table' in html and '<td>1</td>' in html and '&lt;i&gt;2&lt;/i&gt;' in html
    assert '<li' in html


# ————— интеграция с панелью (16.16) —————

def test_r16_16_pipeline_and_probes_untouched_by_history():
    import importlib.util
    spec = importlib.util.spec_from_file_location('launcher', PROJECT_ROOT / 'Запусти меня.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    before = (json.dumps([s['id'] for s in mod.PIPELINE]), json.dumps([p['id'] for p in mod.PROBES]),
              json.dumps(sorted(mod.TASKS)))
    assert hasattr(mod.Api, 'open_history')
    # история строится и фильтруется — панель при этом не меняется
    idx = hindex.build_index(PROJECT_ROOT, cache_dir=None)
    filters.apply([e.to_dict() for e in idx.experiments], {'q': 'x'})
    after = (json.dumps([s['id'] for s in mod.PIPELINE]), json.dumps([p['id'] for p in mod.PROBES]),
             json.dumps(sorted(mod.TASKS)))
    assert before == after
    # модуль истории не зависит от панели и не умеет ничего запускать
    import evaluation.project_history as ph
    src = ''.join((Path(ph.__file__).parent / f).read_text(encoding='utf-8')
                  for f in ('index.py', 'catalog.py', 'artifacts.py', 'filters.py', 'render.py'))
    assert 'Запусти меня' not in src and 'subprocess' not in src


def test_launcher_html_has_history_button():
    html = (PROJECT_ROOT / 'ui' / 'launcher.html').read_text(encoding='utf-8')
    assert 'id="histbtn"' in html and 'open_history' in html


# ————— настоящий каталог проекта —————

def test_real_catalog_is_valid_and_links_control_commits(built):
    assert built.errors == [], built.errors
    assert not [w for w in built.warnings if 'не совпадает' in w or 'в источнике' in w], built.warnings
    by_commit = {}
    for e in built.experiments:
        for h in e.commits:
            by_commit.setdefault(h, []).append(e.id)
    for h in ['3472a05', '4690dad', '2dbdce0', '889913f', '7a989be']:
        assert h in by_commit, h
    assert all(m.described for e in built.experiments for m in e.metrics)
    assert len(built.experiments) >= 17
    states = {e.state for e in built.experiments}
    assert {'closed', 'completed_negative', 'completed_positive', 'infrastructure', 'inconclusive'} <= states


def test_r16_8_plan_render_does_not_duplicate_last_table_row():
    md = '# T\n\nтекст\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nпосле\n'
    html = render.plan_html(plan_parser.parse_plan(md))
    assert html.count('<td>3</td>') == 1 and '| 3 | 4 |' not in html
    assert '<p id="L10"' in html and 'после' in html


def test_ui_plan_tab_keeps_filter_rail_markup():
    html = (PROJECT_ROOT / 'ui' / 'project_history.html').read_text(encoding='utf-8')
    assert 'id="tocbody"' in html and "$('tocbody').innerHTML" in html
    assert "$('railbody').innerHTML" not in html
