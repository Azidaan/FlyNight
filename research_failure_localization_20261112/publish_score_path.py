"""Prepare reviewed graph derivatives and stage only this follow-up audit."""
from common import *
import subprocess,shutil
import scripts.obsidian_level_graph as gen
import scripts.audit_obsidian_levels as aud
D=OUT/'score_path'
def git(*args,**kwargs):return subprocess.check_output(['git',*args],cwd=ROOT,**kwargs)
def put(n,v):
    with (D/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
if sys.argv[1]=='stage':
    p=read(D/'publication.json');assert git('rev-parse','HEAD').decode().strip()==p['head'];assert not git('diff','--cached','--name-only').strip();entries=[]
    for name,value in p['files'].items():
        path=Path(value['source']);assert sha(path)==value['sha256'],name
        oid=git('hash-object','-w','--stdin',input=path.read_bytes().replace(b'\r\n',b'\n')).decode().strip();entries.append(('100644 '+oid+'\t'+name+'\0').encode())
    git('update-index','-z','--index-info',input=b''.join(entries));git('diff','--cached','--check');print('STAGED',len(entries));sys.exit()
assert read(D/'result.json')['passed'];assert not git('diff','--cached','--name-only').strip()
vault=ROOT/'ObsidianFlyVis';live=D/'graph_live';pub=D/'graph_publication';baseline=D/'graph_baseline'
settings=(vault/'.obsidian/graph.json').read_bytes()
protected={p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*.md') if 'Гипотез' in p.as_posix()}
for dest in [baseline,live,pub]:
    assert not dest.exists();shutil.copytree(vault,dest,ignore=shutil.ignore_patterns('plugins','.trash','workspace.json','workspace-mobile.json'))
class Proxy:
    def __init__(self,overrides):self.overrides=overrides
    def __truediv__(self,n):return self.overrides.get(str(n),ROOT/n)
    def __getattr__(self,n):return getattr(ROOT,n)
    def __fspath__(self):return str(ROOT)
gen.local_file=lambda n,label:'['+label+']('+(ROOT/n).as_uri()+')'
gen.ROOT=Proxy({'experiments/project_history.json':D/'project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md':D/'ЕДИНЫЙ_ПЛАН.md'})
load=gen.system_problems;gen.system_problems=lambda:read(D/'system_problems.json');gen.build(baseline)
gen.ROOT=ROOT;gen.system_problems=load;gen.build(live);live_audit=aud.audit(live);assert live_audit['passed']
changed=[]
for n in read(live/'.graph-visible.json')['nodes'].values():
    rel=n['path'];target=vault/rel;new=live/rel
    if not target.exists() or target.read_bytes()!=new.read_bytes():
        if target.exists():assert target.read_bytes()==(baseline/rel).read_bytes(),('Dirty note differs from generated baseline',rel)
        changed.append(rel)
for rel,h in protected.items():assert sha(live/rel)==h
for rel in changed:
    target=vault/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((live/rel).read_bytes())
(vault/'.graph-visible.json').write_bytes((live/'.graph-visible.json').read_bytes())
assert (vault/'.obsidian/graph.json').read_bytes()==settings
before=(D/'ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');after=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');marker='## Журнал изменений\n';bp,bt=before.split(marker,1);ap,at=after.split(marker,1);assert bp==ap and at.endswith(bt);extra=at[:-len(bt)]
headplan=git('show','HEAD:dissertation/ЕДИНЫЙ_ПЛАН.md').decode().replace('\r\n','\n');(D/'publication_plan.md').write_text(headplan.replace(marker,marker+extra,1),encoding='utf8')
gen.ROOT=aud.ROOT=Proxy({'dissertation/ЕДИНЫЙ_ПЛАН.md':D/'publication_plan.md'});gen.build(pub);pub_audit=aud.audit(pub);assert pub_audit['passed']
selected={}
for node in read(pub/'.graph-visible.json')['nodes'].values():
    name='ObsidianFlyVis/'+node['path'];path=pub/node['path']
    result=subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,capture_output=True)
    if result.returncode!=0 or result.stdout.replace(b'\r\n',b'\n')!=path.read_bytes().replace(b'\r\n',b'\n'):selected[name]=str(path)
for rel,h in protected.items():assert sha(pub/rel)==h
selected['ObsidianFlyVis/.graph-visible.json']=str(pub/'.graph-visible.json');selected['dissertation/ЕДИНЫЙ_ПЛАН.md']=str(D/'publication_plan.md')
for n in ['experiments/project_history.json','experiments/system_problems.json']:selected[n]=str(ROOT/n)
for n in ['score_path_audit.py','record_score_path.py','publish_score_path.py']:selected[(OUT/n).relative_to(ROOT).as_posix()]=str(OUT/n)
for n in ['.gitignore','protocol.json','result.json','РАЗБОР_ОЦЕНКИ.md','catalog_validation_fixed.log','tests.log','literature.json']:selected[(D/n).relative_to(ROOT).as_posix()]=str(D/n)
put('graph_check.json',dict(passed=True,live=live_audit,publication=pub_audit,changed=changed,protected_hypotheses=protected,settings_unchanged=True))
selected[(D/'graph_check.json').relative_to(ROOT).as_posix()]=str(D/'graph_check.json')
put('publication.json',dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),files={n:dict(source=v,sha256=sha(v)) for n,v in selected.items()}))
print('PREPARED',len(selected),'GRAPH_CHANGED',len(changed))
