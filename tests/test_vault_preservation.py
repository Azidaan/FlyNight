"""Protect user notes, including staged edits and intentional local deletions."""
import shutil
import json
import hashlib
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


def test_owned_refresh_keeps_foreign_edits_and_registers_only_explicit_note(tmp_path, monkeypatch):
    import scripts
    git(tmp_path, 'init', '-q')
    vault=tmp_path/'ObsidianFlyVis';vault.mkdir()
    owned=vault/'old_plan.md';owned.write_text('our old generated plan',encoding='utf-8')
    foreign=vault/'user.md';foreign.write_text('original',encoding='utf-8')
    manifest=vault/'.generated.json';manifest.write_text('[]',encoding='utf-8')
    git(tmp_path,'add','.');git(tmp_path,'-c','user.name=Test','-c','user.email=test@example.com','commit','-qm','initial')
    foreign.write_text('user edit',encoding='utf-8')
    receipt=tmp_path/'owned.json';receipt.write_text(json.dumps({'ObsidianFlyVis/old_plan.md':hashlib.sha256(owned.read_bytes()).hexdigest()}),encoding='utf-8')
    def generate():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        assert old==['old_plan.md']
        for name in old:(vault/name).unlink()
        foreign.write_text('should be blocked',encoding='utf-8')
        return 0
    monkeypatch.setattr(scripts,'build_obsidian_vault',SimpleNamespace(main=generate,shutil=shutil),raising=False)
    monkeypatch.setattr(preservation,'ROOT',tmp_path)
    monkeypatch.setattr(preservation.sys,'argv',['preserve','--report',str(tmp_path/'result.json'),'--refresh-owned',str(receipt)])
    preservation.main()
    assert not owned.exists() and foreign.read_text(encoding='utf-8')=='user edit'
    assert manifest.read_text(encoding='utf-8')=='[]'
    assert (tmp_path/'result.owned_backup/old_plan.md').read_text(encoding='utf-8')=='our old generated plan'


def test_owned_refresh_rejects_changed_or_outside_note(tmp_path):
    vault=tmp_path/'ObsidianFlyVis';vault.mkdir();note=vault/'own.md';note.write_text('changed',encoding='utf-8')
    receipt=tmp_path/'owned.json';receipt.write_text(json.dumps({'ObsidianFlyVis/own.md':'wrong'}),encoding='utf-8')
    with pytest.raises(ValueError,match='changed after receipt'):preservation.owned_refresh(tmp_path,receipt,tmp_path/'backup')
    receipt.write_text(json.dumps({'outside.md':'wrong'}),encoding='utf-8')
    with pytest.raises(ValueError):preservation.owned_refresh(tmp_path,receipt,tmp_path/'backup')
