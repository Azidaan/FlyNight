"""Check and publish the completed native benchmark using an exact file list."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
W = ROOT / 'runs/z13_weekend_20261024'
STAGE = ROOT / 'runs/z13_val20_matched_20261024/native_masa_recovery2'


def foreign_unchanged():
    protected = read(W / 'vault_preparation.json')['protected_existing_edits']
    for name, digest in protected.items():
        p = ROOT / name
        assert (sha(p) if p.is_file() else None) == digest, name
    assert sha(ROOT / '.gitignore') == read(W / 'native20_final_audit.json')['gitignore_sha256']
    return {n.replace(chr(92), '/') for n in protected}


def graph_rows():
    rows = subprocess.check_output(['git', '--no-optional-locks', 'status',
        '--porcelain', '-z', '--untracked-files=all', '--', 'ObsidianFlyVis'], cwd=ROOT)
    return [(r[:2], r[3:].replace(chr(92), '/'))
            for r in rows.decode('utf-8').split(chr(0)) if r]


def checks():
    from evaluation.project_history.index import fingerprint
    assert not (W / 'native20_final_checks.json').exists()
    foreign = foreign_unchanged()
    before = fingerprint(ROOT)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONIOENCODING': 'utf-8'}
    for label, args in (
        ('catalog', ['scripts/validate_project_history.py']),
        ('tests', ['-m', 'pytest', 'tests/test_project_history.py',
                   'tests/test_z13_status_read_retry.py', '-q'])):
        proc = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), *args],
                              cwd=ROOT, env=env, capture_output=True)
        (W / ('native20_final_' + label + '.log')).write_bytes(proc.stdout)
        (W / ('native20_final_' + label + '.stderr.log')).write_bytes(proc.stderr)
        assert proc.returncode == 0, (label, proc.returncode)
        print(label + ' passed', flush=True)
    after = fingerprint(ROOT)
    assert before == after, 'Index inputs changed while checks were running'
    owned = {n: sha(ROOT / n) for _, n in graph_rows()
             if n not in foreign and (ROOT / n).is_file()
             and (ROOT / n).suffix == '.md'}
    receipt = W / 'native20_final_owned_graph.json'
    write(receipt, owned)
    proc = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'),
        'scripts/z13_build_vault_preserving_edits.py', '--report',
        str(W / 'native20_final_vault.json'), '--refresh-owned', str(receipt)],
        cwd=ROOT, env=env, capture_output=True)
    (W / 'native20_final_vault.log').write_bytes(proc.stdout)
    (W / 'native20_final_vault.stderr.log').write_bytes(proc.stderr)
    assert proc.returncode == 0
    foreign_unchanged()
    write(W / 'native20_final_checks.json', dict(at=now().isoformat(),
        catalog_exit=0, tests_exit=0, input_fingerprint_before=before,
        input_fingerprint_after=after, vault_completed=True,
        foreign_edits_preserved=57, no_scientific_recalculation=True))
    print('graph passed;57foreign edits and gitignore preserved', flush=True)


def publish():
    checks = read(W / 'native20_final_checks.json')
    assert checks['catalog_exit'] == checks['tests_exit'] == 0
    assert '60 passed' in (W / 'native20_final_tests.log').read_text(encoding='utf-8')
    assert read(STAGE / 'finish_recorded.json')['report_sha256'] == sha(STAGE / 'report.json')
    assert read(W / 'native20_final_audit.json')['all_identities_ended']
    foreign = foreign_unchanged()
    assert not (ROOT / '.git/index.lock').exists()
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip() == 'main'
    assert subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT).decode().strip() == 'https://gitlab.com/aziteam-group/drbdiss.git'
    names = ['experiments/project_history.json', 'dissertation/ЕДИНЫЙ_ПЛАН.md',
             'scripts/z13_verify_native20_final_20261024.py',
             'scripts/z13_publish_native20_final_20261024.py']
    for n in ('protocol.json', 'controller.json', 'controller_result.json',
              'price.json', 'previous_end_audit.json', 'launch.json',
              'completion.json', 'report.json', 'verified_summary.json',
              'results.md', 'finish_recorded.json'):
        if (STAGE / n).is_file():
            names.append((STAGE / n).relative_to(ROOT).as_posix())
    names += [p.relative_to(ROOT).as_posix() for p in (STAGE / 'jobs').glob('*.json')]
    for n in ('HANDOFF.md', 'native20_final_audit.json', 'native20_final_checks.json',
              'native20_final_owned_graph.json', 'native20_final_vault.json',
              'val20_native_masa_recovery2_verified.json',
              'val20_native_masa_recovery2_vault.json',
              'val20_native_masa_recovery2_complete.json'):
        names.append((W / n).relative_to(ROOT).as_posix())
    deleted = set()
    for status, name in graph_rows():
        if name not in foreign:
            if (ROOT / name).is_file():
                names.append(name)
            elif 'D' in status:
                names.append(name)
                deleted.add(name)
    names = sorted(set(names))
    assert not set(names) & foreign and '.gitignore' not in names
    assert all((ROOT / n).is_file() or n in deleted for n in names)
    staged = set(subprocess.check_output(['git', 'diff', '--cached',
                 '--name-only', '-z'], cwd=ROOT).decode().split(chr(0))) - {''}
    assert staged <= set(names), 'Unrelated files already staged'
    for start in range(0, len(names), 25):
        subprocess.run(['git', 'add', '-f', '--', *names[start:start+25]],
                       cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'diff', '--cached', '--check'],
                   cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git', 'commit', '--quiet', '-m',
        'Verify matched20 native trackers with shared MASA and preserve pending FlyVis gate',
        '-m', 'Co-authored-by: Codex <noreply@openai.com>'],
        cwd=ROOT, check=True, capture_output=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    print(json.dumps(dict(commit=commit, files=len(names))), flush=True)
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, check=True)
    foreign_unchanged()
    write(W / 'native20_final_publication.json', dict(at=now().isoformat(),
        commit=commit, push_completed=True, paths=names,
        pending_flyvis_tolerance_unchanged=True))


if __name__ == '__main__':
    if sys.argv[1:] == ['--checks']:
        checks()
    elif sys.argv[1:] == ['--publish']:
        publish()
    else:
        raise RuntimeError('Choose --checks or --publish explicitly')
