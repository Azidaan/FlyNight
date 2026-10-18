"""Verify documentary research references without importing experiment code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads((FOLDER / 'baseline_manifest.json').read_text(encoding='utf-8'))
    checks = {}
    for rel, expected in manifest['sha256'].items():
        path = (ROOT / rel).resolve()
        path.relative_to(ROOT.resolve())
        actual = digest(path)
        checks[rel] = {'expected': expected, 'actual': actual, 'unchanged': expected == actual}
    assert all(item['unchanged'] for item in checks.values()), checks
    result = {
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'checks': checks,
        'all_unchanged': True,
        'documentary_scope': 'No training, feature extraction, video processing, or experimental evaluation',
        'artifacts': {p.name: digest(p) for p in FOLDER.glob('*.md')},
    }
    report = json.loads((ROOT / 'runs/z13_target_center_dev_20261018/report.json').read_text(encoding='utf-8'))
    old = report['summary']['D_prior']['useful_mean_video']
    new = report['summary']['D_center']['useful_mean_video']
    result['existing_result_only'] = {
        'source': 'runs/z13_target_center_dev_20261018/report.json',
        'baseline_locator': 'summary.D_prior.useful_mean_video',
        'candidate_locator': 'summary.D_center.useful_mean_video',
        'baseline': old, 'candidate': new,
        'relative_percent': 100 * (new - old) / old,
        'not_hota': True,
    }
    (FOLDER / 'verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Unchanged references: {len(checks)}/{len(checks)}; documentary artifacts verified.')


if __name__ == '__main__':
    main()
