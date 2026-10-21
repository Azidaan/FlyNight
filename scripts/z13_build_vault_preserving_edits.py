"""Run the official graph generator while preserving existing local graph edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def snapshot_local_edits(root):
    """Include staged changes, local deletions and untracked notes, using NUL paths."""
    paths = set()
    for args in (['diff', '--name-only', '-z', 'HEAD', '--', 'ObsidianFlyVis'],
                 ['ls-files', '--others', '--exclude-standard', '-z', '--', 'ObsidianFlyVis']):
        paths.update(p for p in subprocess.check_output(
            ['git', '--no-optional-locks', *args], cwd=root).decode('utf-8').split('\0') if p)
    vault = (root / 'ObsidianFlyVis').resolve()
    protected = {}
    for rel in paths:
        path = (root / rel).resolve()
        path.relative_to(vault)
        protected[path] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return protected


def verify_local_edits(protected):
    for path, digest in protected.items():
        if digest is None:
            assert not path.exists(), f'Locally deleted note was recreated: {path}'
        else:
            assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, path


def owned_refresh(root, receipt, backup):
    """Explicit agent-owned generated notes only, guarded by exact current hashes."""
    if receipt is None:
        return {}
    receipt = receipt.resolve(); receipt.relative_to(root.resolve())
    vault = (root / 'ObsidianFlyVis').resolve()
    records = json.loads(receipt.read_text(encoding='utf-8'))
    allowed = {}
    for rel, digest in records.items():
        path = (root / rel).resolve(); sub = path.relative_to(vault)
        if path.suffix != '.md' or not path.is_file():
            raise ValueError('Only existing explicitly owned generated notes: ' + rel)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Owned note changed after receipt: ' + rel)
        allowed[path] = digest
    # Validate the whole set before preserving any original and before generation.
    for path in allowed:
        target = backup / path.relative_to(vault)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Never overwrite a preservation backup: ' + str(target))
        target.write_bytes(path.read_bytes())
    return allowed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path,
                    default=ROOT / 'runs/z13_spatial_hota_v1_20261014/vault_preservation.json')
    ap.add_argument('--refresh-owned', type=Path,
                    help='Explicit current SHA256 receipt for our generated notes; other edits stay protected')
    args = ap.parse_args()
    out = args.report.resolve()
    out.relative_to(ROOT.resolve())
    from scripts import build_obsidian_vault as generator
    protected = snapshot_local_edits(ROOT)
    refresh = owned_refresh(ROOT, args.refresh_owned, out.with_suffix('.owned_backup'))
    protected = {p: h for p, h in protected.items() if p not in refresh}
    original = Path.write_text
    original_read = Path.read_text
    original_unlink = Path.unlink
    original_copy = generator.shutil.copy2
    skipped = []
    skipped_deletes = []
    skipped_copies = []
    manifest = (ROOT / 'ObsidianFlyVis' / '.generated.json').resolve()
    def reading(path, *args, **kwargs):
        text = original_read(path, *args, **kwargs)
        if path.resolve() == manifest and refresh:
            old = json.loads(text)
            # Supplement generator's in-memory manifest, never the user's file.
            old = sorted(set(old) | {p.relative_to(manifest.parent).as_posix() for p in refresh})
            return json.dumps(old, ensure_ascii=False)
        return text
    def preserving(path, text, *args, **kwargs):
        if path.resolve() in protected:
            skipped.append(str(path.relative_to(ROOT)))
            return len(text)
        return original(path, text, *args, **kwargs)
    def preserving_unlink(path, *args, **kwargs):
        if path.resolve() in protected:
            skipped_deletes.append(str(path.relative_to(ROOT)))
            return None
        return original_unlink(path, *args, **kwargs)
    def preserving_copy(src, dst, *args, **kwargs):
        path = Path(dst)
        if path.resolve() in protected:
            skipped_copies.append(str(path.relative_to(ROOT)))
            return str(dst)
        return original_copy(src, dst, *args, **kwargs)
    Path.write_text = preserving
    Path.read_text = reading
    Path.unlink = preserving_unlink
    generator.shutil.copy2 = preserving_copy
    previous_argv = sys.argv
    previous_optional_locks = os.environ.get('GIT_OPTIONAL_LOCKS')
    try:
        os.environ['GIT_OPTIONAL_LOCKS'] = '0'
        sys.argv = ['build_obsidian_vault.py']
        code = generator.main()
    finally:
        if previous_optional_locks is None:
            os.environ.pop('GIT_OPTIONAL_LOCKS', None)
        else:
            os.environ['GIT_OPTIONAL_LOCKS'] = previous_optional_locks
        Path.write_text = original
        Path.read_text = original_read
        Path.unlink = original_unlink
        generator.shutil.copy2 = original_copy
        sys.argv = previous_argv
        verify_local_edits(protected)
    assert code == 0
    report = {'protected_existing_edits': {str(p.relative_to(ROOT)): h for p, h in protected.items()},
              'skipped_writes': skipped, 'skipped_deletes': skipped_deletes,
              'skipped_copies': skipped_copies,
              'refreshed_owned_notes': {str(p.relative_to(ROOT)): h for p, h in refresh.items()},
              'official_generator_completed': True}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Preserved existing edited files:', len(protected))


if __name__ == '__main__':
    main()
