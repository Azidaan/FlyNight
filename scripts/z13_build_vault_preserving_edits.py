"""Run the official graph generator while preserving existing local graph edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from scripts import build_obsidian_vault as generator
    paths = subprocess.check_output(['git', '-c', 'core.quotepath=false', 'diff', '--name-only',
                                     '--', 'ObsidianFlyVis'], cwd=ROOT, text=True, encoding='utf-8').splitlines()
    protected = {(ROOT / p).resolve(): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                 for p in paths if (ROOT / p).is_file()}
    original = Path.write_text
    original_unlink = Path.unlink
    skipped = []
    skipped_deletes = []
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
    Path.write_text = preserving
    Path.unlink = preserving_unlink
    try:
        sys.argv = ['build_obsidian_vault.py']
        code = generator.main()
    finally:
        Path.write_text = original
        Path.unlink = original_unlink
    assert code == 0
    assert all(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in protected.items())
    report = {'protected_existing_edits': {str(p.relative_to(ROOT)): h for p, h in protected.items()},
              'skipped_writes': skipped, 'skipped_deletes': skipped_deletes,
              'official_generator_completed': True}
    out = ROOT / 'runs/z13_spatial_hota_v1_20261014/vault_preservation.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Preserved existing edited files:', len(protected))


if __name__ == '__main__':
    main()
