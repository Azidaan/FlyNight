"""Publish only this completed research and generated, unprotected graph files."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_flyvis_deep_2026-10-25'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf8').strip()


def main():
    assert not (OUT / 'publication.json').exists()
    protection = read(OUT / 'preservation.json')
    audit = read(OUT / 'provenance.json')
    def unchanged():
        for name, h in protection['graph'].items():
            p = ROOT / name
            assert (sha(p) if p.is_file() else None) == h, name
        assert sha(ROOT / '.gitignore') == protection['gitignore_sha256']
        for name, h in audit['sha256'].items():
            assert sha(ROOT / name) == h, name
        assert sha(OUT / 'СВОДКА.md') == audit['report_sha256']
    unchanged()
    assert read(OUT / 'validation.json')['project_history_tests'] == {'passed': 58, 'failed': 0}
    assert read(OUT / 'vault_preservation.json')['official_generator_completed']
    assert not (ROOT / '.git/index.lock').exists()
    assert not git('diff', '--cached', '--name-only')
    assert git('branch', '--show-current') == 'main'
    assert git('remote', 'get-url', 'origin') == 'https://gitlab.com/aziteam-group/drbdiss.git'
    assert git('rev-parse', 'HEAD') == protection['head']
    names = ['experiments/project_history.json', 'dissertation/ЕДИНЫЙ_ПЛАН.md',
        'runs/z13_weekend_20261024/HANDOFF.md', 'scripts/z13_record_deep_research_20261025.py',
        Path(__file__).relative_to(ROOT).as_posix()]
    names += [(OUT / n).relative_to(ROOT).as_posix() for n in
        ('СВОДКА.md', 'provenance.json', 'preservation.json', 'validation.json', 'vault_preservation.json')]
    rows = subprocess.check_output(['git', '--no-optional-locks', 'status', '--porcelain', '-z',
        '--untracked-files=all', '--', 'ObsidianFlyVis'], cwd=ROOT).decode('utf8').split('\0')
    names += [row[3:].replace('\\', '/') for row in rows if row
        and row[3:].replace('\\', '/') not in protection['graph']
        and (ROOT / row[3:]).is_file()]
    names = sorted(set(names))
    assert not set(names).intersection(protection['graph']) and '.gitignore' not in names
    for n in names:
        assert (ROOT / n).is_file(), n
    for i in range(0, len(names), 15):
        subprocess.run(['git', '-c', 'core.autocrlf=input', 'add', '-f', '--', *names[i:i+15]], cwd=ROOT, check=True)
    staged = set(git('diff', '--cached', '--name-only', '-z').split('\0')) - {''}
    assert staged and staged <= set(names), staged - set(names)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
    unchanged()
    subprocess.run(['git', 'commit', '-m', 'Audit tracking literature against FlyVis proposal and identity failures',
        '-m', 'Co-authored-by: Codex <noreply@openai.com>'], cwd=ROOT, check=True)
    commit = git('rev-parse', 'HEAD')
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, check=True)
    unchanged()
    (OUT / 'publication.json').write_text(json.dumps(dict(
        at=datetime.now(timezone(timedelta(hours=3))).isoformat(), commit=commit,
        push_completed=True, paths=names), ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print('Published:', commit)


if __name__ == '__main__':
    main()
