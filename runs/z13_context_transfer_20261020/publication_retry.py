"""Publish the verified start with an explicit file boundary after authorized lock recovery."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent


def write(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git(*args, **kwargs):
    return subprocess.check_output(['git', *args], cwd=ROOT, **kwargs)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    hold = RUN / 'publication_approval_required.json'
    assert not hold.exists() or not json.loads(hold.read_text(encoding='utf-8'))['blocked'], 'Scientific publication approval is pending'
    auth = json.loads((RUN / 'publication_authorization.json').read_text(encoding='utf-8'))
    assert auth['destination'] == 'https://gitlab.com/aziteam-group/drbdiss.git' and auth['branch'] == 'main'
    assert git('remote','get-url','origin').decode().strip() == auth['destination']
    assert git('branch','--show-current').decode().strip() == auth['branch']
    assert not (ROOT / '.git/index.lock').exists()
    assert not git('diff', '--cached', '--name-only', '-z'), 'Preexisting staged files'
    for rel, digest in json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))['sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    preservation = json.loads((RUN / 'vault_preservation.json').read_text(encoding='utf-8'))
    for rel, digest in preservation['protected_existing_edits'].items():
        path = (ROOT / rel).resolve()
        path.relative_to((ROOT / 'ObsidianFlyVis').resolve())
        assert (not path.exists()) if digest is None else (hashlib.sha256(path.read_bytes()).hexdigest() == digest), rel
    own = ['scripts/z13_context_transfer_20261020.py', 'scripts/z13_context_transfer_progress.ps1',
           'scripts/z13_record_context_transfer_20261020.py', 'scripts/z13_context_transfer_service_20261020.py',
           'experiments/project_history.json', 'dissertation/ЕДИНЫЙ_ПЛАН.md',
           'ObsidianFlyVis/01 Муха/Эксперименты/z13_context_transfer_v1.md']
    # Explicitly named scientific records only: no recursive collection, logs, caches or binaries.
    records = ['publication_retry.py', 'publication_retry_intent.json', 'publication_scope_recovery.json',
               'protocol.json', 'full_authorization.json', 'controller.json', 'controller_result.json',
               'technical.json', 'technical_controller_result.json', 'train_rationale.json',
               'all_heads_fitted.json', 'validation_result.json', 'index_lock_recovery.json',
               'administrative_sources.json', 'recording_recovery.json', 'start_recorded.json',
               'finish_recorded.json', 'report.json', 'publication_authorization.json',
               'publication_approval_required.json', 'publication_state_audit.json', 'publication_state_audit.py',
               'C/fit.json', 'R1/fit.json',
               'C/done.json', 'R1/done.json', 'C/predictions.json', 'R1/predictions.json',
               'administrative_previous/z13_record_context_transfer_20261020.py',
               'administrative_previous/record_before_explicit_publication.py']
    write('publication_retry_intent.json', dict(reason='Author explicitly authorized these scientific records for existing GitLab main',
          protocol_hashes_verified=True, protected_graph_verified=True, new_calculation_started=False))
    paths = sorted(set(rel for rel in own if (ROOT / rel).is_file()) |
                   {str((RUN / rel).relative_to(ROOT).as_posix()) for rel in records if (RUN / rel).is_file()})
    for rel in paths:
        (ROOT / rel).resolve().relative_to(ROOT.resolve())
    write('publication_retry_scope.json', dict(paths=paths))
    paths.append((RUN / 'publication_retry_scope.json').relative_to(ROOT).as_posix())
    subprocess.run(['git', 'add', '-f', '--', *paths], cwd=ROOT, check=True)
    staged = set(git('diff', '--cached', '--name-only', '-z').decode('utf-8').strip('\0').split('\0'))
    assert staged and staged <= set(paths), 'Staged files crossed publication boundary'
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', 'Save FlyVis CPU predictions and record aggregation failure',
                    '-m', 'Co-authored-by: Codex <codex@openai.com>'], cwd=ROOT, check=True)
    commit = git('rev-parse', 'HEAD').decode().strip()
    result = subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, capture_output=True, text=True, timeout=90)
    write('publication_retry_result.json', dict(commit=commit, published=result.returncode == 0,
          exit_code=result.returncode, message=(result.stdout + result.stderr)[-3000:]))
    print(json.dumps(dict(commit=commit, published=result.returncode == 0, exit_code=result.returncode)))
    if result.returncode:
        raise RuntimeError('Commit saved; push failed, see publication_retry_result.json')


if __name__ == '__main__':
    main()
