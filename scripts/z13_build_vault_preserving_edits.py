"""Run the official graph generator while preserving existing local graph edits."""
import argparse
import hashlib
import json
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
            ['git', *args], cwd=root).decode('utf-8').split('\0') if p)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path,
                    default=ROOT / 'runs/z13_spatial_hota_v1_20261014/vault_preservation.json')
    args = ap.parse_args()
    out = args.report.resolve()
    out.relative_to(ROOT.resolve())
    from scripts import build_obsidian_vault as generator
    protected = snapshot_local_edits(ROOT)
    original = Path.write_text
    original_unlink = Path.unlink
    original_copy = generator.shutil.copy2
    skipped = []
    skipped_deletes = []
    skipped_copies = []
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
    Path.unlink = preserving_unlink
    generator.shutil.copy2 = preserving_copy
    previous_argv = sys.argv
    try:
        sys.argv = ['build_obsidian_vault.py']
        code = generator.main()
    finally:
        Path.write_text = original
        Path.unlink = original_unlink
        generator.shutil.copy2 = original_copy
        sys.argv = previous_argv
        verify_local_edits(protected)
    assert code == 0
    report = {'protected_existing_edits': {str(p.relative_to(ROOT)): h for p, h in protected.items()},
              'skipped_writes': skipped, 'skipped_deletes': skipped_deletes,
              'skipped_copies': skipped_copies,
              'official_generator_completed': True}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Preserved existing edited files:', len(protected))


if __name__ == '__main__':
    main()
