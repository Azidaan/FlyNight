"""Audit the exact proposed published graph, including archived old paths."""
from common import *
import subprocess

def git(*args, **kw):
    return subprocess.check_output(['git',*args],cwd=ROOT,**kw)

def main():
    guard()
    p = read(OUT/'prepared.json')
    assert not git('diff','--cached','--name-only').strip()
    vault = OUT/'publication_vault'
    old = json.loads(git('show','HEAD:ObsidianFlyVis/.graph-visible.json'))
    new = read(vault/'.graph-visible.json')
    archived = []
    for n in set(old['files'])-set(new['files']):
        source = vault/n
        assert source.exists() and 'граф-проекта/архив' in source.read_text(encoding='utf8')
        name = 'ObsidianFlyVis/'+n
        p['files'][name] = dict(source=str(source),sha256=sha(source))
        archived.append(n)
    from scripts.audit_obsidian_levels import audit
    import scripts.audit_obsidian_levels as auditor
    projected = OUT/'projected_vault'
    projected.mkdir(exist_ok=True)
    paths = git('ls-tree','-r','--name-only','-z','HEAD','--','ObsidianFlyVis/Карта проекта').decode().strip('\0').split('\0')
    paths = [n for n in paths if n.endswith('.md')]
    raw = git('cat-file','--batch',input=('\n'.join('HEAD:'+n for n in paths)+'\n').encode())
    pos = 0
    for n in paths:
        end = raw.index(b'\n',pos)
        _,kind,size = raw[pos:end].decode().split()
        assert kind == 'blob'
        pos = end+1
        data = raw[pos:pos+int(size)]
        pos += int(size)+1
        path = projected/Path(n).relative_to('ObsidianFlyVis')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    config = projected/'.obsidian/graph.json'
    config.parent.mkdir(exist_ok=True)
    config.write_bytes(git('show','HEAD:ObsidianFlyVis/.obsidian/graph.json'))
    for n,item in p['files'].items():
        if n.startswith('ObsidianFlyVis/'):
            path = projected/Path(n).relative_to('ObsidianFlyVis')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(Path(item['source']).read_bytes())
    # The previous published snapshot contains historical active notes absent
    # from its own manifest. Include only the generator's exact tag-only repair;
    # never carry unrelated text edits into this publication.
    from scripts.obsidian_level_graph import ACTIVE
    known = set(new['files'])
    historical_repairs = []
    for path in (projected/'Карта проекта').rglob('*.md'):
        text = path.read_text(encoding='utf8')
        parts = text.split('---\n',2)
        n = path.relative_to(projected).as_posix()
        if len(parts)==3 and ACTIVE in parts[1] and n not in known:
            source = vault/n
            assert source.read_text(encoding='utf8') == text.replace(ACTIVE,'граф-проекта/архив',1), 'Nontrivial historical text change: '+n
            path.write_bytes(source.read_bytes())
            p['files']['ObsidianFlyVis/'+n] = dict(source=str(source),sha256=sha(source))
            historical_repairs.append(n)
    class RootProxy:
        def __truediv__(self,name):
            return OUT/'publication_plan.md' if str(name)=='dissertation/ЕДИНЫЙ_ПЛАН.md' else ROOT/name
        def __getattr__(self,name):
            return getattr(ROOT,name)
        def __fspath__(self):
            return str(ROOT)
    auditor.ROOT = RootProxy()
    result = audit(projected)
    assert result['passed']
    save('publication_audit.json',dict(passed=True,archived_paths_included=archived,historical_tag_only_repairs=historical_repairs,projected_graph=result,
                                      scope='Exact projected publication audited; historical user drafts excluded.'))
    for name in ['publication_audit.py','publication_audit.json','stage_publication.py','.gitignore']:
        path=OUT/name
        p['files'][path.relative_to(ROOT).as_posix()]=dict(source=str(path),sha256=sha(path))
    p['projected_graph_verified']=True
    save('prepared_v2.json',p)
    print('PROJECTED_PUBLICATION_CHECKED',len(p['files']),len(archived))

if __name__=='__main__':
    main()
