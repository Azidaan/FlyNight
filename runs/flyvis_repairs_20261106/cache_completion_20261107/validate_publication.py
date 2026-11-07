"""Validate isolated ledger reads against real local evidence; never swap disk files."""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--overlay', required=True)
ap.add_argument('--mode', choices=('validator', 'tests'), required=True)
ap.add_argument('--stage', choices=('after', 'baseline'), default='after')
ap.add_argument('--group', choices=('main','cache'), default='main')
args = ap.parse_args()
overlay = (ROOT / args.overlay).resolve()
overlay.relative_to((ROOT / 'runs/flyvis_repairs_20261106').resolve())
names = ('dissertation/ЕДИНЫЙ_ПЛАН.md', 'experiments/project_history.json')
mapping = {(ROOT / name).resolve(): overlay / args.stage / name for name in names}
mapping[(ROOT / 'experiments/metrics_glossary.json').resolve()] = overlay / 'baseline/experiments/metrics_glossary.json'
before = {path: path.read_bytes() for path in mapping}
candidate_before = {path: path.read_bytes() for path in mapping.values()}
hits = {str(path.relative_to(ROOT)): 0 for path in mapping}
original_open = Path.open

def redirected_open(self, mode='r', *pos, **kw):
    resolved = self.resolve()
    if resolved in mapping:
        if any(flag in mode for flag in 'wax+'):
            raise AssertionError('Ledger validation attempted a production write: ' + str(self))
        hits[str(resolved.relative_to(ROOT))] += 1
        return original_open(mapping[resolved], mode, *pos, **kw)
    return original_open(self, mode, *pos, **kw)

report_path = overlay / f'validation_{args.stage}_{args.mode}_{args.group}_stable.json'
assert not report_path.exists(), 'Preserve earlier validation evidence'
Path.open = redirected_open
code = 0
try:
    if args.mode == 'validator':
        sys.argv = ['scripts/validate_project_history.py']
        try:
            runpy.run_module('scripts.validate_project_history', run_name='__main__')
        except SystemExit as exc:
            code = exc.code or 0
    else:
        import pytest
        selected = ('test_r16_14_cache_invalidated_by_fingerprint' if args.group == 'cache' else 'not test_r16_14_cache_invalidated_by_fingerprint')
        code = pytest.main([str(ROOT / 'tests/test_project_history.py'), str(ROOT / 'tests/test_project_history_cache_transient.py'), '-k', selected, '-q', '-p', 'no:cacheprovider'])
finally:
    Path.open = original_open
    unchanged = all(path.read_bytes() == data for path, data in before.items())
    candidates_unchanged = all(path.read_bytes() == data for path, data in candidate_before.items())
    result = dict(mode=args.mode, stage=args.stage, exit_code=int(code), read_redirections=hits,
        production_ledger_and_glossary_unchanged=unchanged,
        publication_candidates_unchanged=candidates_unchanged,
        concurrent_working_ledger_changes=[str(path.relative_to(ROOT)) for path, data in before.items() if path.read_bytes()!=data],
        input_sha256={str(path.relative_to(ROOT)): hashlib.sha256(target.read_bytes()).hexdigest() for path, target in mapping.items()},
        boundary='Only reads of the prepared ledger and HEAD glossary are redirected in this process; actual local artifacts and read-only Git supply evidence. No on-disk production replacement or staging.')
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    # Other chats may append working records. These tests read only the pinned
    # publication candidates; their mutation, unlike unrelated working drift,
    # invalidates this run. Writes to mapped production paths remain forbidden.
    assert candidates_unchanged
    assert all(hits.values()), 'Expected ledger reads did not pass through the overlay'
raise SystemExit(code)
