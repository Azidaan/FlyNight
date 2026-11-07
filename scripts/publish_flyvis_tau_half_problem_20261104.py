"""Обновить только существующую проблему С7 через проверенную копию графа."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/flyvis_tau_half_20261104'
sys.path.insert(0, str(ROOT))
from scripts.obsidian_system_problems import load, problem_path
from scripts.obsidian_fundamental_overview import build
from scripts.audit_fundamental_obsidian import audit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((OUT / 'chain_result.json').read_text(encoding='utf-8'))['complete']
    vault = ROOT / 'ObsidianFlyVis'
    backup, stage = OUT / 'vault_before', OUT / 'vault_checked'
    assert not backup.exists() and not stage.exists()
    shutil.copytree(vault, backup)
    shutil.copytree(vault, stage)
    before = {p.relative_to(vault).as_posix(): digest(p) for p in vault.rglob('*') if p.is_file()}
    target = problem_path(next(p for p in load()['problems'] if p['id'] == 'С7')) + '.md'
    preserve = [name for name in before if name.endswith('.md') and name != target]
    assert build(stage, preserve) == 0
    for name in ('.obsidian/graph.json', '.graph-visible.json', '.generated.json'):
        shutil.copy2(backup / name, stage / name)
    checked = audit(stage)
    assert checked['note_count'] == 55 and checked['hypothesis_node_count'] == 25
    assert checked['problem_node_count'] == 10 and not checked['external_hypothesis_edges']
    assert set(before) == {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    changed = [name for name, value in before.items() if digest(stage / name) != value]
    assert changed == [target], changed
    assert all(digest(vault / name) == value for name, value in before.items())
    shutil.copy2(stage / target, vault / target)
    published = audit(vault)
    assert published['hashes'] == checked['hashes']
    assert all(digest(vault / name) == value for name, value in before.items() if name != target)
    published.update(changed_files=changed, added_files=[], deleted_files=[], moved_files=[],
                     protected_hypotheses_byte_exact=True, graph_settings_byte_exact=True)
    (OUT / 'graph_audit.json').write_text(json.dumps(published, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': published['passed'], 'changed_files': changed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
