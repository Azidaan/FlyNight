"""Reading history must not refresh the repository index or take its lock."""
import os
import subprocess
from evaluation.project_history.git_reader import GitReader


def test_status_does_not_refresh_index_after_stat_only_change(tmp_path,monkeypatch):
    monkeypatch.delenv('GIT_OPTIONAL_LOCKS',raising=False)
    repo=tmp_path/'repo';repo.mkdir()
    def git(*args):
        subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=True)
    git('init');file=repo/'tracked.txt';file.write_text('unchanged\n')
    git('add','tracked.txt')
    git('-c','user.name=ReadOnlyTest','-c','user.email=test@example.invalid','commit','-m','fixture')
    index=repo/'.git/index';before=index.read_bytes();stamp=index.stat().st_mtime_ns
    s=file.stat();os.utime(file,ns=(s.st_atime_ns,s.st_mtime_ns+2_000_000_000))
    state=GitReader(repo).working_tree()
    assert state['is_clean']
    assert index.read_bytes()==before
    assert index.stat().st_mtime_ns==stamp
    assert not (repo/'.git/index.lock').exists()
