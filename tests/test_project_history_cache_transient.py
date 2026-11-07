"""Disk cache must not retain transient untracked paths; live state is rebuilt."""
import json
from copy import deepcopy

from evaluation.project_history import index as module
from evaluation.project_history.model import TimelineEvent


def fixture():
    live = TimelineEvent(id='uncommitted', when='2026-11-07', kind='uncommitted', title='live',
                         files=['runs/temporary/%08d.dat' % n for n in range(10000)])
    historical = TimelineEvent(id='past', when='2026-11-06', kind='manual', title='saved', files=['report.json'])
    idx = module.Index(timeline=[historical, live], working_tree={'untracked': list(live.files)})
    idx.stats = module._stats(idx)
    return idx


def test_disk_cache_excludes_transient_paths_and_preserves_history(tmp_path):
    idx = fixture()
    before = deepcopy(idx.to_dict())
    path = tmp_path/'index.json'
    module._write_cache(idx, path)
    raw = path.read_bytes()
    assert b'runs/temporary/' not in raw
    cached = json.loads(raw)
    assert 'working_tree' not in cached
    assert [t['id'] for t in cached['timeline']] == ['past']
    assert cached['timeline'][0] == before['timeline'][0]
    assert cached['stats']['events'] == 1
    assert cached['stats']['by_kind'] == {'manual': 1}
    assert idx.to_dict() == before


def test_cached_load_rebuilds_same_current_working_state(tmp_path, monkeypatch):
    idx = fixture()
    stamp = {'version': module.INDEX_VERSION, 'head': 'same'}
    idx.fingerprint = stamp
    path = tmp_path/'cache'/'index.json'
    module._write_cache(idx, path)
    monkeypatch.setattr(module, 'fingerprint', lambda root: stamp)
    class Git:
        def __init__(self, root): pass
        def available(self): return True
        def working_tree(self):
            return {'branch':'main','head':'abc','short':'abc','modified':[],
                    'untracked':['runs/current.json']}
    monkeypatch.setattr(module, 'GitReader', Git)
    expected = deepcopy(idx)
    module._attach_working_tree(expected, Git(tmp_path), expected.timeline)
    module._sort_timeline(expected.timeline)
    expected.stats = module._stats(expected)
    actual = module.load_index(tmp_path, cache_dir=path.parent)
    assert actual.from_cache
    assert actual.working_tree == expected.working_tree
    assert [t.to_dict() for t in actual.timeline] == [t.to_dict() for t in expected.timeline]
    assert actual.stats == expected.stats
