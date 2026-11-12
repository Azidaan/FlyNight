"""Prepare reviewed graph derivatives and stage only this follow-up audit."""
from common import *
import subprocess,shutil
import scripts.obsidian_level_graph as gen
import scripts.audit_obsidian_levels as aud
D=OUT
def git(*args,**kwargs):return subprocess.check_output(['git',*args],cwd=ROOT,**kwargs)
def put(n,v):
    with (D/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
if sys.argv[1]=='stage':
    p=read(D/'publication.json');assert git('rev-parse','HEAD').decode().strip()==p['head'];assert not git('diff','--cached','--name-only').strip();entries=[]
    for name,value in p['files'].items():
        path=Path(value['source']);assert sha(path)==value['sha256'],name
        oid=git('hash-object','-w','--stdin',input=path.read_bytes().replace(b'\r\n',b'\n')).decode().strip();entries.append(('100644 '+oid+'\t'+name+'\0').encode())
    git('update-index','-z','--index-info',input=b''.join(entries));git('diff','--cached','--check');print('STAGED',len(entries));sys.exit()
assert read(D/'audit.json')['passed'];assert not git('diff','--cached','--name-only').strip()
vault=ROOT/'ObsidianFlyVis';live=D/'graph_live_v2';pub=D/'graph_publication_v2';baseline=D/'graph_baseline_v2'
settings=(vault/'.obsidian/graph.json').read_bytes()
protected={p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*.md') if 'Гипотез' in p.as_posix()}
old_manifest=read(vault/'.graph-visible.json');orphans={}
previous_manifest=json.loads(git('show','HEAD^:ObsidianFlyVis/.graph-visible.json'))
for file in (vault/'Карта проекта').rglob('*.md'):
    text=file.read_text(encoding='utf8');front=text.split('---\n',2);rel=file.relative_to(vault).as_posix()
    if len(front)!=3 or gen.ACTIVE not in front[1] or rel in old_manifest['files']:continue
    candidates=[node for node in previous_manifest['nodes'].values() if node['path']==rel];assert len(candidates)==1,rel
    assert git('show','HEAD^:ObsidianFlyVis/'+rel)==file.read_bytes().replace(b'\r\n',b'\n'),rel
    orphans[rel]=dict(node=candidates[0],sha256=sha(file))
# Give the generator its verified historical records so it archives their tags.
# No scientific text or protected hypothesis is rewritten.
original_init=gen.Tree.__init__
def initialize_with_history(self,target):
    original_init(self,target)
    for rel,value in orphans.items():self.old['verified_archival_path:'+rel]=value['node']
gen.Tree.__init__=initialize_with_history
for dest in [baseline,live,pub]:
    assert not dest.exists();shutil.copytree(vault,dest,ignore=shutil.ignore_patterns('plugins','.trash','workspace.json','workspace-mobile.json'))
class Proxy:
    def __init__(self,overrides):self.overrides=overrides
    def __truediv__(self,n):return self.overrides.get(str(n),ROOT/n)
    def __getattr__(self,n):return getattr(ROOT,n)
    def __fspath__(self):return str(ROOT)
gen.local_file=lambda n,label:'['+label+']('+(ROOT/n).as_uri()+')'
gen.ROOT=Proxy({'experiments/project_history.json':D/'before'/'project_history.json','experiments/useful_literature.json':D/'before'/'useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md':D/'before'/'ЕДИНЫЙ_ПЛАН.md'})
load=gen.system_problems;gen.system_problems=lambda:read(D/'before'/'system_problems.json');gen.build(baseline)
gen.ROOT=ROOT;gen.system_problems=load;gen.build(live);live_audit=aud.audit(live);assert live_audit['passed']
changed=[]
for new in (live/'Карта проекта').rglob('*.md'):
    rel=new.relative_to(live).as_posix();target=vault/rel
    if not target.exists() or target.read_bytes()!=new.read_bytes():
        if target.exists():assert target.read_bytes()==(baseline/rel).read_bytes() or (rel in orphans and sha(target)==orphans[rel]['sha256']),('Dirty note differs from generated baseline',rel)
        changed.append(rel)
for rel,h in protected.items():assert sha(live/rel)==h
for rel in changed:
    target=vault/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((live/rel).read_bytes())
(vault/'.graph-visible.json').write_bytes((live/'.graph-visible.json').read_bytes())
assert (vault/'.obsidian/graph.json').read_bytes()==settings
before=(D/'before'/'ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');after=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');marker='## Журнал изменений\n';bp,bt=before.split(marker,1);ap,at=after.split(marker,1);assert bp==ap and at.endswith(bt);extra=at[:-len(bt)]
headplan=git('show','HEAD:dissertation/ЕДИНЫЙ_ПЛАН.md').decode().replace('\r\n','\n');(D/'publication_plan.md').write_text(headplan.replace(marker,marker+extra,1),encoding='utf8')
gen.ROOT=aud.ROOT=Proxy({'dissertation/ЕДИНЫЙ_ПЛАН.md':D/'publication_plan.md'});gen.build(pub);pub_audit=aud.audit(pub);assert pub_audit['passed']
selected={}
node_paths={node['path'] for node in read(pub/'.graph-visible.json')['nodes'].values()}|{node['path'] for node in old_manifest['nodes'].values()}|set(orphans)
nodes=[dict(path=path) for path in sorted(node_paths)]
names=['ObsidianFlyVis/'+node['path'] for node in nodes]
batch=git('cat-file','--batch',input=('\n'.join('HEAD:'+n for n in names)+'\n').encode());position=0
for node,name in zip(nodes,names):
    end=batch.index(b'\n',position);header=batch[position:end].decode();position=end+1
    if header.endswith(' missing'):previous=b''
    else:
        _,kind,size=header.split();assert kind=='blob';size=int(size);previous=batch[position:position+size];position+=size+1
    path=pub/node['path']
    if previous.replace(b'\r\n',b'\n')!=path.read_bytes().replace(b'\r\n',b'\n'):selected[name]=str(path)
for rel,h in protected.items():assert sha(pub/rel)==h
selected['ObsidianFlyVis/.graph-visible.json']=str(pub/'.graph-visible.json');selected['dissertation/ЕДИНЫЙ_ПЛАН.md']=str(D/'publication_plan.md')
for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/obsidian_experiment_topics.json','experiments/useful_literature.json']:selected[n]=str(ROOT/n)
for n in ['common.py','register.py','worker.py','audit.py','audit_secondary_v2.py','preflight_correction.py','crop_observation.py','finish.py','publish.py']:selected[(OUT/n).relative_to(ROOT).as_posix()]=str(OUT/n)
for n in ['.gitignore','protocol.json','protocol_registered_v1.json','crop_observation.json','loaded_model.json','baseline_gate.json','replay.json','scores.json','audit.json','execution_receipt.json','worker.log','РЕЗУЛЬТАТ.md','catalog_validation.log','tests.log']:selected[(D/n).relative_to(ROOT).as_posix()]=str(D/n)
aud.ROOT=ROOT
assert aud.audit(vault)['passed']
put('graph_check.json',dict(passed=True,live=live_audit,publication=pub_audit,changed=changed,protected_hypotheses=protected,settings_unchanged=True,repaired_verified_historical_orphans=orphans,repair='Apply archived-note changes as well as current manifest nodes. Prior helper omitted these derived tag changes. Scientific source text preserved.'))
selected[(D/'graph_check.json').relative_to(ROOT).as_posix()]=str(D/'graph_check.json')
put('publication.json',dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),files={n:dict(source=v,sha256=sha(v)) for n,v in selected.items()}))
print('PREPARED',len(selected),'GRAPH_CHANGED',len(changed))
