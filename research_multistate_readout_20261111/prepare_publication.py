"""Prepare checked scientific records excluding pre-existing foreign drafts."""
from common import *
start()
import shutil
import subprocess
import scripts.obsidian_level_graph as generator
import scripts.audit_obsidian_levels as auditor


def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,**kw)
def head(name):return git('show','HEAD:'+name)
def save(path,raw):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)

checks=read(OUT/'final_checks.json');assert checks['passed'];verify()
assert git('rev-parse','HEAD').decode().strip()==read(OUT/'protocol.json')['head']
assert not git('diff','--cached','--name-only').strip()
initial=read(OUT/'before.json')
base=read(OUT/'before/experiments/project_history.json');current=read(ROOT/'experiments/project_history.json');identity=read(OUT/'recording.json')['card']
without=dict(current);without['experiments']=[c for c in current['experiments'] if c['id']!=identity]
assert without==base,'Foreign experiment cards changed'
oldplan=(OUT/'before/dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
now=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');marker='## Журнал изменений'
assert oldplan.split(marker)[0]==now.split(marker)[0]
assert now.split(marker)[1].endswith(oldplan.split(marker)[1].lstrip())
published=head('dissertation/ЕДИНЫЙ_ПЛАН.md').decode().replace('\r\n','\n')
prefix='\n\n'+(OUT/'result_entry.md').read_text(encoding='utf8').strip()+'\n\n'+(OUT/'registration_entry.md').read_text(encoding='utf8').strip()
published=published.replace(marker,marker+prefix,1)
save(OUT/'publication_plan.md',published.encode())
vault=OUT/'publication_vault';assert not vault.exists();shutil.copytree(OUT/'preview',vault)


class RootProxy:
    def __truediv__(self,name):return OUT/'publication_plan.md' if str(name)=='dissertation/ЕДИНЫЙ_ПЛАН.md' else ROOT/name
    def __getattr__(self,name):return getattr(ROOT,name)
    def __fspath__(self):return str(ROOT)


generator.ROOT=auditor.ROOT=RootProxy()
generator.local_file=lambda name,label:'['+label+']('+(ROOT/name).as_uri()+')'
generator.build(vault);graph=auditor.audit(vault)
assert graph['passed'] and graph['journal_entries']==checks['live']['journal_entries']-2
old=read(ROOT/'ObsidianFlyVis/.graph-visible.json');new=read(vault/'.graph-visible.json')
for name,digest in initial['protected_hypothesis_graph'].items():assert sha(vault/Path(name).relative_to('ObsidianFlyVis'))==digest
# One batch fetch of published notes avoids hundreds of separate git processes.
paths=git('ls-tree','-r','--name-only','-z','HEAD','--','ObsidianFlyVis').decode().strip('\0').split('\0')
paths=[p for p in paths if p.endswith('.md')]
raw=git('cat-file','--batch',input=('\n'.join('HEAD:'+p for p in paths)+'\n').encode());pos=0;previous={}
for name in paths:
    end=raw.index(b'\n',pos);_,kind,size=raw[pos:end].decode().split();assert kind=='blob';pos=end+1
    previous[name]=raw[pos:pos+int(size)];pos+=int(size)+1
selected={}
for node in new['nodes'].values():
    name='ObsidianFlyVis/'+node['path'];value=(vault/node['path']).read_bytes()
    if value.replace(b'\r\n',b'\n')!=previous.get(name,b'').replace(b'\r\n',b'\n'):
        assert node['section']!='Гипотезы';selected[name]=str(vault/node['path'])
selected['ObsidianFlyVis/.graph-visible.json']=str(vault/'.graph-visible.json')
for name in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json']:
    selected[name]=str(ROOT/name)
selected['dissertation/ЕДИНЫЙ_ПЛАН.md']=str(OUT/'publication_plan.md')
for file in OUT.iterdir():
    if not file.is_file() or file.name.startswith(('launch.','final_checks.launch.','recovery1.launch.')):continue
    if file.name in ('prepared.json','before.json','publication_plan.md','status.json','executor.json','launch.json'):continue
    if file.suffix in ('.py','.md','.json','.log') or file.name=='.gitignore':selected[file.relative_to(ROOT).as_posix()]=str(file)
assert not any('/Гипотезы/' in name or name=='AGENTS.md' or '/plugins/' in name for name in selected)
write('prepared.json',dict(passed=True,base_head=git('rev-parse','HEAD').decode().strip(),publication_graph=graph,
    scope='Only fixed two-state readout comparison, required scientific metadata and derived non-hypothesis graph notes. No raw arrays, foreign plan entries, hypotheses, plugins or drafts.',
    files={name:dict(source=source,sha256=sha(Path(source))) for name,source in sorted(selected.items())}))
print('PREPARED',len(selected))
