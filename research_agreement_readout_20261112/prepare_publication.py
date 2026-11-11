"""Prepare only this task's evidence and generated notes, excluding prior drafts."""
from common import *
import subprocess, shutil

def git(*args, **kw):
    return subprocess.check_output(['git', *args], cwd=ROOT, **kw)

def main():
    guard()
    verify()
    assert read(OUT/'graph_check.json')['passed']
    assert not git('diff','--cached','--name-only').strip()
    assert git('rev-parse','HEAD').decode().strip() == read(OUT/'protocol.json')['head']
    old = read(OUT/'before/project_history.json')
    current = read(ROOT/'experiments/project_history.json')
    without = dict(current)
    without['experiments'] = [c for c in current['experiments'] if c['id'] != ID]
    assert without == old, 'Concurrent experiment edits need separation'
    marker = '## Журнал изменений\n'
    before = (OUT/'before/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
    after = (ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
    bp, bt = before.split(marker, 1)
    ap, at = after.split(marker, 1)
    assert bp == ap and at.endswith(bt)
    additions = at[:-len(bt)]
    published = git('show','HEAD:dissertation/ЕДИНЫЙ_ПЛАН.md').decode('utf8').replace('\r\n','\n')
    published = published.replace(marker, marker+additions, 1)
    (OUT/'publication_plan.md').write_text(published, encoding='utf8')
    vault = OUT/'publication_vault'
    assert not vault.exists()
    shutil.copytree(OUT/'graph_preview', vault)
    import scripts.obsidian_level_graph as generator
    import scripts.audit_obsidian_levels as auditor
    class RootProxy:
        def __truediv__(self, name):
            return OUT/'publication_plan.md' if str(name) == 'dissertation/ЕДИНЫЙ_ПЛАН.md' else ROOT/name
        def __getattr__(self, name):
            return getattr(ROOT, name)
        def __fspath__(self):
            return str(ROOT)
    generator.ROOT = auditor.ROOT = RootProxy()
    generator.local_file = lambda name,label: '['+label+']('+(ROOT/name).as_uri()+')'
    generator.build(vault)
    result = auditor.audit(vault)
    assert result['passed']
    for n,h in read(OUT/'graph_check.json')['protected_hypotheses'].items():
        assert sha(vault/n) == h
    paths = git('ls-tree','-r','--name-only','-z','HEAD','--','ObsidianFlyVis').decode().strip('\0').split('\0')
    paths = [n for n in paths if n.endswith('.md')]
    raw = git('cat-file','--batch', input=('\n'.join('HEAD:'+n for n in paths)+'\n').encode())
    pos = 0
    previous = {}
    for n in paths:
        end = raw.index(b'\n',pos)
        _, kind, size = raw[pos:end].decode().split()
        assert kind == 'blob'
        pos = end+1
        previous[n] = raw[pos:pos+int(size)]
        pos += int(size)+1
    selected = {}
    new = read(vault/'.graph-visible.json')
    for node in new['nodes'].values():
        name = 'ObsidianFlyVis/'+node['path']
        value = (vault/node['path']).read_bytes()
        if value.replace(b'\r\n',b'\n') != previous.get(name,b'').replace(b'\r\n',b'\n'):
            assert node['section'] != 'Гипотезы'
            selected[name] = str(vault/node['path'])
    selected['ObsidianFlyVis/.graph-visible.json'] = str(vault/'.graph-visible.json')
    selected['dissertation/ЕДИНЫЙ_ПЛАН.md'] = str(OUT/'publication_plan.md')
    for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json']:
        selected[n] = str(ROOT/n)
    for f in OUT.iterdir():
        if f.is_file() and (f.suffix in ['.py','.md','.json','.log'] or f.name == '.gitignore') and f.name not in ['prepared.json','publication_plan.md','status.json','worktree_before.json']:
            selected[f.relative_to(ROOT).as_posix()] = str(f)
    save('prepared.json', dict(passed=True, base_head=read(OUT/'protocol.json')['head'], publication_graph=result,
                               files={n:dict(source=v,sha256=sha(v)) for n,v in sorted(selected.items())},
                               scope='Own bounded scientific result and clean derived graph; all previous user drafts, settings and hypotheses excluded.'))
    print('PUBLICATION_PREPARED',len(selected))

if __name__ == '__main__':
    main()
