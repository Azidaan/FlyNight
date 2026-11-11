"""Generate a checked copy first; refuse to overwrite existing manual edits."""
from common import *
import subprocess, shutil

def main():
    guard()
    assert read(OUT/'after_preflight.json')['passed'] and read(OUT/'recording.json')['passed']
    from scripts.build_obsidian_vault import main as build
    from scripts.audit_obsidian_levels import audit
    vault = ROOT/'ObsidianFlyVis'
    preview = OUT/'graph_preview'
    assert not preview.exists()
    shutil.copytree(vault, preview, ignore=shutil.ignore_patterns('plugins', '.trash', 'workspace.json', 'workspace-mobile.json'))
    before = {p.relative_to(vault).as_posix(): sha(p) for p in vault.rglob('*.md')}
    protected = {n:h for n,h in before.items() if 'Гипотез' in n}
    dirty = subprocess.check_output(['git','diff','--name-only','-z','--','ObsidianFlyVis'], cwd=ROOT).decode('utf8').strip('\0').split('\0')
    dirty = {Path(n).relative_to('ObsidianFlyVis').as_posix() for n in dirty if n}
    settings = (vault/'.obsidian/graph.json').read_bytes()
    sys.argv = ['build_obsidian_vault.py','--vault',str(preview)]
    assert build() == 0
    # Preserve the author's UI preferences byte for byte. The full graph audit
    # below still requires the intended filter and colors.
    (preview/'.obsidian/graph.json').write_bytes(settings)
    for n,h in protected.items():
        assert sha(preview/n) == h, 'Protected hypothesis changed: '+n
    conflicts = [n for n in dirty if (vault/n).is_file() and (preview/n).is_file() and sha(vault/n) != sha(preview/n)]
    assert not conflicts, ('Existing manual edits would change', conflicts)
    result = audit(preview)
    assert result['passed']
    after = {p.relative_to(preview).as_posix():sha(p) for p in preview.rglob('*.md')}
    changed = [n for n,h in after.items() if before.get(n) != h]
    # An explicitly deleted tracked file must stay deleted.
    assert not [n for n in changed if n in dirty and not (vault/n).exists()]
    for n in changed:
        assert (sha(vault/n) if (vault/n).exists() else None) == before.get(n), 'Concurrent note edit: '+n
    for n in changed:
        dest = vault/n
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((preview/n).read_bytes())
    (vault/'.graph-visible.json').write_bytes((preview/'.graph-visible.json').read_bytes())
    assert (vault/'.obsidian/graph.json').read_bytes() == settings
    assert audit(vault) == result
    assert all(sha(vault/n) == h for n,h in protected.items())
    node = read(vault/'.graph-visible.json')['nodes']['exp:'+ID]
    assert node['color'] == 'без-положительного-итога'
    save('graph_check.json', dict(passed=True, preview=result, live=result, protected_notes=len(protected),
                                  protected_hypotheses=protected, existing_manual_edits_unchanged=True,
                                  changed=changed, new_card_note=node['path']))
    verify()
    print('GRAPH_CHECKED',len(changed),result['note_count'])

if __name__ == '__main__':
    main()
