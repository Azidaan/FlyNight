"""Recover the exact previous generated note into an archive, using its saved hash."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json
from scripts import build_obsidian_vault as generator

RUN = ROOT / 'runs/z13_shared_masa_hota_v1_20261015/full'
catalog_path = ROOT / 'experiments/project_history.json'
catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
preservation = json.loads((ROOT / 'runs/z13_spatial_hota_v1_20261014/vault_preservation.json').read_text(encoding='utf-8'))
name = 'ObsidianFlyVis\\09 Запланировано\\fly_spatial_hota_policy_z13_v1.md'
expected = preservation['protected_existing_edits'][name]
isolated = RUN / 'graph_recovery'
isolated.mkdir(exist_ok=True)
captured = {}


class CaptureVault:
    def __init__(self, path):
        self.path = path
        self.written = []

    def write(self, rel, text):
        self.written.append(rel.replace('\\', '/'))
        captured[rel.replace('\\', '/')] = text
        if rel == '09 Запланировано/fly_spatial_hota_policy_z13_v1.md':
            captured['text'] = text

    def copy(self, src, rel):
        self.written.append(rel.replace('\\', '/'))


original_read = Path.read_text
original_write = Path.write_text
original_vault = generator.Vault
original_settings = generator.graph_settings
current_json = ''


def read_source(path, *args, **kwargs):
    if path.resolve() == catalog_path.resolve():
        return current_json
    try:
        rel = path.resolve().relative_to(isolated.resolve()).as_posix()
    except ValueError:
        rel = None
    if rel in captured:
        return captured[rel]
    return original_read(path, *args, **kwargs)


def write_generated(path, text, *args, **kwargs):
    try:
        rel = path.resolve().relative_to(isolated.resolve()).as_posix()
    except ValueError:
        return original_write(path, text, *args, **kwargs)
    captured[rel] = text
    return len(text)


def main():
    global current_json
    # The previous catalog is recoverable from revisions; enumerate only the
    # saved evidence-prefix length/state, accepting an exact prior hash only.
    old = copy.deepcopy(catalog)
    card = next(c for c in old['experiments'] if c['id'] == 'fly_spatial_hota_policy_z13_v1')
    for r in reversed(card.get('revisions', [])):
        if r.get('reason', '').startswith('Полный фиксированный опыт завершён'):
            card[r['field']] = r['old']
    card['revisions'] = [r for r in card.get('revisions', []) if not r.get('reason', '').startswith('Полный фиксированный опыт завершён')]
    if not card['revisions']:
        card.pop('revisions')
    for field in ('done', 'obtained', 'interpretation', 'decision', 'metrics'):
        card.pop(field, None)
    for c in card['acceptance'][1:]:
        c['met'] = None
        c.pop('evidence', None)
    evidence = card['evidence'][:]
    try:
        Path.read_text = read_source
        Path.write_text = write_generated
        generator.Vault = CaptureVault
        generator.graph_settings = lambda *a, **k: None
        for state in ('running', 'ready', 'planned'):
            card['state'] = state
            for count in range(len(evidence), 0, -1):
                card['evidence'] = evidence[:count]
                current_json = json.dumps(old, ensure_ascii=False)
                sys.argv = ['build_obsidian_vault.py', '--vault', str(isolated)]
                assert generator.main() == 0
                text = captured['text']
                data = text.replace('\n', os.linesep).encode('utf-8')
                if hashlib.sha256(data).hexdigest() != expected:
                    continue
                archive = isolated / 'prior_fly_spatial_hota_policy_z13_v1.md'
                assert not archive.exists()
                archive.write_bytes(data)
                assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
                atomic_json(RUN / 'graph_recovery.json', {'exact_old_hash_recovered': True,
                    'old_path': name, 'sha256': expected, 'archive': archive.relative_to(ROOT).as_posix(),
                    'method': 'Official generator rendering of previous catalog; byte equality to preexisting saved hash.',
                    'state': state, 'evidence_prefix_count': count,
                    'cause': 'Preservation wrapper protected writes but not generator cleanup unlink. Wrapper now protects unlink too.',
                    'current_result_note': 'ObsidianFlyVis/01 Муха/Эксперименты/fly_spatial_hota_policy_z13_v1.md'})
                print('RECOVERED EXACT', expected, archive)
                return
        raise RuntimeError('No exact previous note found; do not claim recovery')
    finally:
        Path.read_text = original_read
        Path.write_text = original_write
        generator.Vault = original_vault
        generator.graph_settings = original_settings


if __name__ == '__main__':
    main()
