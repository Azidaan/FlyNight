"""Validate records and apply only the four generated notes reviewed on a copy."""
import hashlib
import json
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)

def call(script,args):
    saved=sys.argv
    try:
        sys.argv=[str(script),*map(str,args)]
        try:runpy.run_path(str(ROOT/script),run_name='__main__')
        except SystemExit as ex:
            if ex.code not in (0,None):raise
    finally:sys.argv=saved

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def files(folder):
    return {p.relative_to(folder):sha(p) for p in folder.rglob('*') if p.is_file()
            and '.obsidian' not in p.relative_to(folder).parts}

if 'catalog' in sys.argv:
    call('scripts/validate_project_history.py',[])
elif 'tests' in sys.argv or 'cache-test' in sys.argv:
    import pytest
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
        suites=['tests/test_project_history.py','tests/test_obsidian_fundamental_overview.py',
                'tests/test_obsidian_checked_hypotheses.py','tests/test_obsidian_tracking_hypotheses.py',
                'tests/test_obsidian_transfer_outcome.py']
        if 'cache-test' in sys.argv:
            suites=['tests/test_project_history.py::test_r16_14_cache_invalidated_by_fingerprint']
        rc=pytest.main(['-q',*suites,'--basetemp',str(Path(tmp)/'tests')])
        if rc:sys.exit(rc)
elif 'graph' in sys.argv:
    from scripts.audit_fundamental_obsidian import audit
    vault=ROOT/'ObsidianFlyVis'
    before=files(vault)
    targets=[next(vault.glob(s)).relative_to(vault) for s in [
        'Опорные задачи/Задача 3 — *.md','Опорные задачи/Задача 4 — Восстановить *.md',
        'Проблемы проекта/С1 — *.md','Проблемы проекта/С2 — *.md']]
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
        preview=Path(tmp)/'ObsidianFlyVis'
        shutil.copytree(vault,preview)
        call('scripts/build_obsidian_vault.py',['--vault',preview])
        check=audit(preview)
        assert check['passed']
        # Historical files may be concurrently edited or removed by the author.
        # Publish generated output only for the requested task and problem notes.
        assert all(sha(vault/p)==before[p] for p in targets)
        for p in targets:
            shutil.copyfile(preview/p,vault/p)
        actual=audit(vault)
        assert actual['passed'] and actual['hashes']==check['hashes']
    after=files(vault)
    assert before.keys()==after.keys() and {p for p in before if before[p]!=after[p]}==set(targets)
    protected=json.loads((ROOT/'runs/identity_assignment_20261108/graph_protected_before.json').read_text(encoding='utf8'))
    assert all(after.get(Path(p))==v for p,v in protected.items())
    result=dict(passed=True,preview_matches_live=True,changed=[p.as_posix() for p in targets],
                visible=actual['note_count'],protected=len(protected),hypotheses=actual['hypothesis_node_count'],
                removed_by_update=0,missing_links=actual['missing_targets'])
    with (HERE/'graph_audit.json').open('x',encoding='utf8') as out:
        json.dump(result,out,ensure_ascii=False,indent=2)
