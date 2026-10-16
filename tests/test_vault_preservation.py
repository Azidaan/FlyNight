"""Protect user notes, including staged edits and intentional local deletions."""
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts import z13_build_vault_preserving_edits as preservation


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def test_preserving_build_keeps_all_local_states(tmp_path, monkeypatch):
    import scripts

    git(tmp_path, 'init', '-q')
    vault = tmp_path / 'ObsidianFlyVis'
    vault.mkdir()
    staged = vault / 'изменённая заметка.md'
    deleted = vault / 'deleted.md'
    copied = vault / 'copy.md'
    for path in (staged, deleted, copied):
        path.write_text('original', encoding='utf-8')
    git(tmp_path, 'add', 'ObsidianFlyVis')
    git(tmp_path, '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
        'commit', '-qm', 'initial')
    staged.write_text('staged user edit', encoding='utf-8')
    git(tmp_path, 'add', '--', str(staged.relative_to(tmp_path)))
    deleted.unlink()
    copied.write_text('unstaged user edit', encoding='utf-8')
    untracked = vault / 'untracked.md'
    untracked.write_text('new user note', encoding='utf-8')
    protected = preservation.snapshot_local_edits(tmp_path)
    assert len(protected) == 4
    assert protected[deleted.resolve()] is None
    source = tmp_path / 'source.md'
    source.write_text('replacement', encoding='utf-8')
    output = vault / 'generated.md'

    def generate():
        for path in (staged, deleted, untracked):
            path.write_text('replacement', encoding='utf-8')
        staged.unlink()
        shutil.copy2(source, copied)
        output.write_text('generated content', encoding='utf-8')
        return 0

    fake = SimpleNamespace(main=generate, shutil=shutil)
    monkeypatch.setattr(scripts, 'build_obsidian_vault', fake, raising=False)
    monkeypatch.setattr(preservation, 'ROOT', tmp_path)
    monkeypatch.setattr(preservation.sys, 'argv', ['preserve', '--report',
                                               str(tmp_path / 'report.json')])
    preservation.main()
    preservation.verify_local_edits(protected)
    assert output.read_text(encoding='utf-8') == 'generated content'
    assert (tmp_path / 'report.json').is_file()
    deleted.write_text('accidentally recreated', encoding='utf-8')
    with pytest.raises(AssertionError, match='recreated'):
        preservation.verify_local_edits(protected)
